"""
Hierarchical badness-based prior construction.

A three-level prior is fitted on the train split:

1. **Global prior** — softmax over per-expert *badness*
   (``c_fn · FNR + c_fp · FPR + tier_cost``) with uniform mixing.
2. **Family prior** — per support-family Dirichlet-shrunk mixture of the raw
   per-family prior with the global prior.
3. **Group prior** — per (family, sub-cluster) bucket, shrunk against both
   parent levels and capped by an anti-collapse :func:`_adaptive_clip_max`.

The final ``[G, M]`` prior tensor is consumed by the GSDP loss in
:mod:`m2p.losses`.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .configs import PriorRegConfig


# ─────────────────────────────────────────────────────────────────────
# Mask parsing
# ─────────────────────────────────────────────────────────────────────

def parse_action_mask(x, M: int) -> np.ndarray:
    """Coerce ``x`` (str/list/array) into a length-``M`` 0/1 expert mask."""
    if isinstance(x, np.ndarray):
        arr = x.astype(float).ravel()
    elif isinstance(x, (list, tuple)):
        arr = np.asarray(x, dtype=float).ravel()
    elif isinstance(x, str):
        s = x.strip().strip("[]").replace(",", " ")
        arr = np.fromstring(s, sep=" ", dtype=float)
    else:
        raise TypeError(f"Unsupported mask type: {type(x)}")

    if arr.size == M + 1:
        arr = arr[1:]  # drop AI slot

    if arr.size != M:
        raise ValueError(
            f"Mask length {arr.size} does not match M={M} (or M+1 with AI)."
        )

    return (arr > 0).astype(np.float32)


def prepare_expert_mask(df: pd.DataFrame, mask_col: str, M: int) -> np.ndarray:
    """Returns ``[N, M]`` expert-only availability matrix."""
    return np.vstack(
        [parse_action_mask(x, M) for x in df[mask_col].tolist()]
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Numpy probability helpers
# ─────────────────────────────────────────────────────────────────────

def _mask_and_normalize_np(p: np.ndarray, mask: np.ndarray,
                           eps: float = 1e-8) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=np.float64)
    p = np.clip(p, 0.0, None) * mask

    if p.sum() <= eps:
        if mask.sum() <= eps:
            return np.zeros_like(p, dtype=np.float64)
        return mask / mask.sum()

    return p / p.sum()


def _clip_and_renormalize_np(p: np.ndarray, mask: np.ndarray,
                             clip_max: float = 0.4,
                             eps: float = 1e-8) -> np.ndarray:
    """
    Clip active entries to ``[0, clip_max]`` and renormalise.

    Excess mass from clipped entries is redistributed proportionally to the
    unclipped active entries; converges in at most ``k`` iterations.
    """
    p = np.asarray(p, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=np.float64)
    active = mask > 0

    if active.sum() == 0:
        return p

    for _ in range(int(active.sum()) + 1):
        excess = np.maximum(p - clip_max, 0.0) * active.astype(float)
        total_excess = excess.sum()
        if total_excess <= eps:
            break
        p = np.minimum(p, clip_max) * active.astype(float)
        uncapped = active & (p < clip_max - eps)
        if uncapped.sum() > 0:
            p[uncapped] += total_excess * (p[uncapped] / p[uncapped].sum())
        else:
            p[active] += total_excess / active.sum()
            break

    return _mask_and_normalize_np(p, mask, eps=eps)


def _scores_to_prior(badness: np.ndarray, active_mask: np.ndarray,
                     n_eval: Optional[np.ndarray] = None,
                     min_eval: int = 1, tau_bad: float = 1.0,
                     capacity: Optional[Sequence[float]] = None,
                     uniform_mix: float = 0.10,
                     eps: float = 1e-8) -> np.ndarray:
    """``softmax(−τ · badness)`` with uniform mixing over the valid support."""
    badness = np.asarray(badness, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=np.float64)

    valid = np.isfinite(badness) & (active_mask > 0)
    if n_eval is not None:
        n_eval = np.asarray(n_eval)
        valid &= n_eval >= min_eval
    if valid.sum() == 0:
        valid = active_mask > 0

    p = np.zeros_like(badness, dtype=np.float64)
    if valid.sum() == 0:
        return p

    scores = -tau_bad * badness[valid]
    if capacity is not None:
        cap = np.asarray(capacity, dtype=np.float64)[valid]
        scores = scores + np.log(np.clip(cap, eps, None))

    scores = scores - scores.max()
    p_valid = np.exp(scores)
    p[valid] = p_valid / np.clip(p_valid.sum(), eps, None)

    if uniform_mix > 0:
        u = np.zeros_like(p)
        u[valid] = 1.0 / valid.sum()
        p = (1.0 - uniform_mix) * p + uniform_mix * u

    return _mask_and_normalize_np(p, valid.astype(np.float64), eps=eps)


# ─────────────────────────────────────────────────────────────────────
# Per-expert badness statistics
# ─────────────────────────────────────────────────────────────────────

def _compute_expert_stats(df_sub: pd.DataFrame,
                          expert_cols: Sequence[str],
                          expert_mask_sub: np.ndarray,
                          y_col: str = "y_true",
                          alpha: float = 1.0, beta: float = 1.0,
                          c_fn: float = 2.0, c_fp: float = 1.0,
                          tier_cost: Optional[Sequence[float]] = None,
                          ) -> Dict[str, np.ndarray]:
    """
    Beta-smoothed FNR/FPR and badness for each expert on ``df_sub``.

    badness_j = ``c_fn · FNR_j + c_fp · FPR_j + tier_cost_j``
    """
    M = len(expert_cols)
    y = df_sub[y_col].to_numpy().astype(int)

    badness     = np.full(M, np.nan, dtype=np.float64)
    fnr         = np.full(M, np.nan, dtype=np.float64)
    fpr         = np.full(M, np.nan, dtype=np.float64)
    n_eval      = np.zeros(M, dtype=np.int64)
    active_rate = np.zeros(M, dtype=np.float64)

    if tier_cost is None:
        tier_cost = np.zeros(M, dtype=np.float64)
    elif isinstance(tier_cost, dict):
        tier_cost = np.array(
            [tier_cost.get(col, 0.0) for col in expert_cols], dtype=np.float64,
        )
    else:
        tier_cost = np.asarray(tier_cost, dtype=np.float64)

    for j, col in enumerate(expert_cols):
        yh_all = df_sub[col].to_numpy()
        valid_label = pd.notna(yh_all) & (yh_all != -1)

        active = expert_mask_sub[:, j] > 0
        keep = active & valid_label

        active_rate[j] = float(active.mean()) if active.size > 0 else 0.0
        n_eval[j] = int(keep.sum())

        if n_eval[j] == 0:
            continue

        yh = yh_all[keep].astype(int)
        yt = y[keep]

        tp = int(((yt == 1) & (yh == 1)).sum())
        fn = int(((yt == 1) & (yh == 0)).sum())
        fp = int(((yt == 0) & (yh == 1)).sum())
        tn = int(((yt == 0) & (yh == 0)).sum())

        fnr[j] = (fn + alpha) / (tp + fn + alpha + beta)
        fpr[j] = (fp + alpha) / (tn + fp + alpha + beta)
        badness[j] = c_fn * fnr[j] + c_fp * fpr[j] + tier_cost[j]

    return {"badness": badness, "fnr": fnr, "fpr": fpr,
            "n_eval": n_eval, "active_rate": active_rate}


# ─────────────────────────────────────────────────────────────────────
# Hierarchical priors
# ─────────────────────────────────────────────────────────────────────

def compute_global_prior(df: pd.DataFrame,
                         expert_cols: Sequence[str],
                         cfg: PriorRegConfig,
                         ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """``p_global = (1 − u) · softmax(−τ · badness) + u · U``."""
    tr = df[df[cfg.split_col] == cfg.train_value].copy()
    M = len(expert_cols)
    tr_mask = prepare_expert_mask(tr, cfg.mask_col, M)

    stats = _compute_expert_stats(
        tr, expert_cols, tr_mask,
        y_col=cfg.y_col, alpha=cfg.alpha, beta=cfg.beta,
        c_fn=cfg.c_fn, c_fp=cfg.c_fp, tier_cost=cfg.tier_cost,
    )
    active_any = (tr_mask.sum(axis=0) > 0).astype(np.float64)

    prior = _scores_to_prior(
        stats["badness"], active_mask=active_any,
        n_eval=stats["n_eval"], min_eval=5,
        tau_bad=cfg.tau_bad, capacity=cfg.capacity,
        uniform_mix=cfg.global_uniform_mix,
    )

    meta = dict(stats)
    meta["n_train"] = np.array([len(tr)])
    meta["active_any"] = active_any
    return prior, meta


def compute_family_prior(df: pd.DataFrame,
                         expert_cols: Sequence[str],
                         global_prior: np.ndarray,
                         cfg: PriorRegConfig,
                         ) -> Tuple[Dict[int, np.ndarray],
                                    Dict[int, Dict[str, np.ndarray]]]:
    """``p_family = λ · p_raw + (1 − λ) · p_global``."""
    tr = df[df[cfg.split_col] == cfg.train_value].copy()
    M = len(expert_cols)
    tr_mask = prepare_expert_mask(tr, cfg.mask_col, M)

    priors: Dict[int, np.ndarray] = {}
    meta: Dict[int, Dict[str, np.ndarray]] = {}

    all_families = sorted(
        df[cfg.family_col].dropna().astype(int).unique().tolist()
    )
    tr_fams = tr[cfg.family_col].astype(int).to_numpy()

    for fam in all_families:
        fam = int(fam)
        rows = np.where(tr_fams == fam)[0]

        if rows.size > 0:
            sub = tr.iloc[rows]
            sub_mask = tr_mask[rows]
            stats = _compute_expert_stats(
                sub, expert_cols, sub_mask,
                y_col=cfg.y_col, alpha=cfg.alpha, beta=cfg.beta,
                c_fn=cfg.c_fn, c_fp=cfg.c_fp, tier_cost=cfg.tier_cost,
            )
            fam_active = (sub_mask.sum(axis=0) > 0).astype(np.float64)
            p_raw = _scores_to_prior(
                stats["badness"], active_mask=fam_active,
                n_eval=stats["n_eval"], min_eval=3,
                tau_bad=cfg.tau_bad, capacity=cfg.capacity,
                uniform_mix=cfg.family_uniform_mix,
            )
            n = len(sub)
            lam_fam = n / (n + cfg.family_n0)
            p = lam_fam * p_raw + (1.0 - lam_fam) * global_prior
            p = _mask_and_normalize_np(p, fam_active)

            if cfg.family_uniform_mix > 0:
                u = fam_active / np.clip(fam_active.sum(), 1e-8, None)
                p = (1.0 - cfg.family_uniform_mix) * p + cfg.family_uniform_mix * u
                p = _mask_and_normalize_np(p, fam_active)

            priors[fam] = p
            meta[fam] = {
                **stats,
                "n_train": np.array([n]),
                "active_any": fam_active,
                "lambda_family": np.array([lam_fam]),
            }
        else:
            full_sub = df[df[cfg.family_col].astype(int) == fam]
            fam_mask = prepare_expert_mask(full_sub, cfg.mask_col, M)
            fam_active = (fam_mask.sum(axis=0) > 0).astype(np.float64)
            p = _mask_and_normalize_np(global_prior, fam_active)
            if cfg.family_uniform_mix > 0:
                u = fam_active / np.clip(fam_active.sum(), 1e-8, None)
                p = (1.0 - cfg.family_uniform_mix) * p + cfg.family_uniform_mix * u
                p = _mask_and_normalize_np(p, fam_active)

            priors[fam] = p
            meta[fam] = {
                "n_train": np.array([0]),
                "active_any": fam_active,
                "lambda_family": np.array([0.0]),
            }

    return priors, meta


def _adaptive_clip_max(k_active: int, anchors: Dict[int, float]) -> float:
    """Piecewise-linear interpolation of clip_max by active expert count."""
    if not anchors:
        return 0.0
    if k_active in anchors:
        return anchors[k_active]
    ks = sorted(anchors.keys())
    if k_active <= ks[0]:
        return anchors[ks[0]]
    if k_active >= ks[-1]:
        return anchors[ks[-1]]
    for i in range(len(ks) - 1):
        if ks[i] <= k_active <= ks[i + 1]:
            k_lo, k_hi = ks[i], ks[i + 1]
            t = (k_active - k_lo) / (k_hi - k_lo)
            return anchors[k_lo] + t * (anchors[k_hi] - anchors[k_lo])
    return anchors[ks[-1]]


def compute_group_prior(df: pd.DataFrame,
                        expert_cols: Sequence[str],
                        family_priors: Dict[int, np.ndarray],
                        global_prior: np.ndarray,
                        cfg: PriorRegConfig,
                        ) -> Tuple[Dict[int, np.ndarray],
                                   Dict[int, Dict[str, np.ndarray]],
                                   torch.Tensor,
                                   Dict[int, int]]:
    """
    Group-level prior:

    .. code-block:: text

        p_group = λ_grp · p_raw + λ_fam · p_family + λ_glo · p_global

    Then clipped to ``[0, _adaptive_clip_max(k_active, cfg.clip_max_anchors)]``
    and renormalised.
    """
    M = len(expert_cols)
    tr = df[df[cfg.split_col] == cfg.train_value].copy()
    tr_mask = prepare_expert_mask(tr, cfg.mask_col, M)

    all_groups = sorted(
        df[cfg.group_col].dropna().astype(int).unique().tolist()
    )
    group_priors: Dict[int, np.ndarray] = {}
    meta: Dict[int, Dict[str, np.ndarray]] = {}

    raw_priors_list = []
    raw_id_to_row: Dict[int, int] = {}

    full_group_to_family = (
        df.groupby(cfg.group_col)[cfg.family_col]
        .agg(lambda s: int(pd.Series.mode(s).iloc[0]))
        .to_dict()
    )
    tr_groups = tr[cfg.group_col].astype(int).to_numpy()

    for row_idx, g in enumerate(all_groups):
        g = int(g)
        raw_id_to_row[g] = row_idx
        tr_idx = np.where(tr_groups == g)[0]

        if tr_idx.size > 0:
            sub = tr.iloc[tr_idx]
            sub_mask = tr_mask[tr_idx]
            fam = int(pd.Series.mode(sub[cfg.family_col]).iloc[0])

            stats = _compute_expert_stats(
                sub, expert_cols, sub_mask,
                y_col=cfg.y_col, alpha=cfg.alpha, beta=cfg.beta,
                c_fn=cfg.c_fn, c_fp=cfg.c_fp, tier_cost=cfg.tier_cost,
            )
            grp_active = (sub_mask.sum(axis=0) > 0).astype(np.float64)
            p_raw = _scores_to_prior(
                stats["badness"], active_mask=grp_active,
                n_eval=stats["n_eval"], min_eval=2,
                tau_bad=cfg.tau_bad, capacity=cfg.capacity,
                uniform_mix=cfg.group_uniform_mix,
            )
            raw_priors_list.append(p_raw.copy())

            lam_group  = len(sub) / (len(sub) + cfg.group_n0)
            lam_global = cfg.global_mix
            lam_family = max(0.0, 1.0 - lam_group - lam_global)

            p = (
                lam_group * p_raw
                + lam_family * family_priors[fam]
                + lam_global * global_prior
            )
            p = _mask_and_normalize_np(p, grp_active)

            if cfg.group_uniform_mix > 0:
                u = grp_active / np.clip(grp_active.sum(), 1e-8, None)
                p = (1.0 - cfg.group_uniform_mix) * p + cfg.group_uniform_mix * u
                p = _mask_and_normalize_np(p, grp_active)

            if cfg.clip_max_anchors:
                k_active = int(grp_active.sum())
                cm = _adaptive_clip_max(k_active, cfg.clip_max_anchors)
                if cm > 0:
                    p = _clip_and_renormalize_np(p, grp_active, clip_max=cm)

            group_priors[g] = p
            meta[g] = {
                **stats,
                "family": np.array([fam]),
                "n_train": np.array([len(sub)]),
                "active_any": grp_active,
                "lambda_group":  np.array([lam_group]),
                "lambda_family": np.array([lam_family]),
                "lambda_global": np.array([lam_global]),
            }
        else:
            fam = int(full_group_to_family[g])
            full_sub = df[df[cfg.group_col].astype(int) == g]
            full_mask = prepare_expert_mask(full_sub, cfg.mask_col, M)
            grp_active = (full_mask.sum(axis=0) > 0).astype(np.float64)

            p = (1.0 - cfg.global_mix) * family_priors[fam] + cfg.global_mix * global_prior
            p = _mask_and_normalize_np(p, grp_active)

            if cfg.group_uniform_mix > 0:
                u = grp_active / np.clip(grp_active.sum(), 1e-8, None)
                p = (1.0 - cfg.group_uniform_mix) * p + cfg.group_uniform_mix * u
                p = _mask_and_normalize_np(p, grp_active)

            if cfg.clip_max_anchors:
                k_active = int(grp_active.sum())
                cm = _adaptive_clip_max(k_active, cfg.clip_max_anchors)
                if cm > 0:
                    p = _clip_and_renormalize_np(p, grp_active, clip_max=cm)

            raw_priors_list.append(p.copy())
            group_priors[g] = p
            meta[g] = {
                "family": np.array([fam]),
                "n_train": np.array([0]),
                "active_any": grp_active,
                "lambda_group":  np.array([0.0]),
                "lambda_family": np.array([1.0 - cfg.global_mix]),
                "lambda_global": np.array([cfg.global_mix]),
            }

    p_raw_table = torch.tensor(
        np.stack(raw_priors_list, axis=0), dtype=torch.float32,
    )
    return group_priors, meta, p_raw_table, raw_id_to_row


# ─────────────────────────────────────────────────────────────────────
# Tensor packaging
# ─────────────────────────────────────────────────────────────────────

def build_prior_tensor(
    prior_dict: Dict[int, np.ndarray],
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """``Dict[id → vector]`` → ``(table [G, M], id_to_row)``."""
    ids = sorted(int(k) for k in prior_dict.keys())
    table = np.stack([prior_dict[i] for i in ids], axis=0)
    id_to_row = {gid: row for row, gid in enumerate(ids)}
    return torch.tensor(table, dtype=torch.float32), id_to_row


def build_all_priors(df: pd.DataFrame, expert_cols: Sequence[str],
                     cfg: Optional[PriorRegConfig] = None,
                     ) -> Dict[str, object]:
    """End-to-end ``global → family → group`` prior construction."""
    if cfg is None:
        cfg = PriorRegConfig()

    global_prior, global_meta = compute_global_prior(df, expert_cols, cfg)
    family_priors, family_meta = compute_family_prior(
        df, expert_cols, global_prior, cfg,
    )
    group_priors, group_meta, raw_group_table, raw_id_to_row = (
        compute_group_prior(df, expert_cols, family_priors, global_prior, cfg)
    )
    final_group_table, final_id_to_row = build_prior_tensor(group_priors)

    return {
        "global_prior":     global_prior,
        "global_meta":      global_meta,
        "family_priors":    family_priors,
        "family_meta":      family_meta,
        "group_priors":     group_priors,
        "group_meta":       group_meta,
        "raw_group_table":  raw_group_table,
        "raw_id_to_row":    raw_id_to_row,
        "final_group_table": final_group_table,
        "final_id_to_row":   final_id_to_row,
        "config":           cfg,
    }


__all__ = [
    "parse_action_mask", "prepare_expert_mask",
    "compute_global_prior", "compute_family_prior", "compute_group_prior",
    "build_prior_tensor", "build_all_priors",
]
