"""
MPD2-Router — full training pipeline in a single module.
========================================================

This file consolidates the entire router-training pipeline that was
originally distributed across nine helper modules (``configs``, ``costs``,
``data``, ``models``, ``priors``, ``losses``, ``augmented_lagrangian``,
``training``, ``evaluation``) into one self-contained Python file.

The contents are organised in the following sections:

  §1  Configuration dataclasses  (TrainConfig, PriorRegConfig, ALConfig)
  §2  Clinical costs             (TIER_COST, EXPERT_TIER, action_costs,
                                  l2d_objective, …)
  §3  Dataset                    (L2DDataset, prob1_from_logits_np)
  §4  Model                      (StructuralRisk, MLPBlock,
                                  OVAExpertHeadFixed, Router)
  §5  Hierarchical priors        (build_all_priors and helpers)
  §6  Routing-regularisation     (gsdp_loss, rank_majorization_js_loss,
                                  combined_routing_loss)
  §7  Augmented Lagrangian       (AugLag)
  §8  Evaluation & reporting     (evaluate, print_eval_report, …)
  §9  Training loop              (train_l2d_multi_expert)

Companion modules kept *separate* because they correspond to upstream pipeline
stages (not the router training itself):

* :mod:`m2p.feature_extraction` — frozen Swin-V2 logits / hidden states.
* :mod:`m2p.ood`                — OOD detectors fitted on REFUGE-train.
* :mod:`m2p.grouping`           — three-stage hierarchical bucketing.
* :mod:`m2p.adaptive_hpo`       — constraint-aware Bayesian HPO.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset


# ════════════════════════════════════════════════════════════════════
# §1  Configuration dataclasses
# ════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Optimiser, schedule, and clinical-cost asymmetry."""
    epochs:        int   = 150
    lr:            float = 1e-4
    patience:      int   = 18
    batch_size:    int   = 64
    gamma_tier:    float = 0.1
    c_fn:          float = 1.5
    c_fp:          float = 1.25
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu"
    min_delta:     float = 1e-4
    warmup_epochs: int   = 15


@dataclass
class ALConfig:
    """Augmented-Lagrangian dual-update knobs."""
    mu:                float            = 20.0
    lr_lambda:         float            = 0.01
    max_deferral_rate: Optional[float]  = 0.75
    max_avg_cost:      Optional[float]  = None
    gamma_tier:        float            = 1.0   # used in eval scoring


@dataclass
class PriorRegConfig:
    """Single source of truth for hierarchical-prior + routing-regularisation knobs."""

    tau_bad: float = 1.0

    global_uniform_mix: float = 0.50
    family_uniform_mix: float = 0.35
    group_uniform_mix:  float = 0.35

    family_n0: float = 25.0
    group_n0:  float = 30.0

    global_mix: float = 0.05

    c_fn: float = 1.8
    c_fp: float = 1.2

    clip_max_anchors: Dict[int, float] = field(
        default_factory=lambda: {5: 0.36, 7: 0.35, 12: 0.34}
    )

    alpha: float = 1.0
    beta:  float = 1.0

    tier_cost: Optional[Sequence[float]] = None
    capacity:  Optional[Sequence[float]] = None

    split_col:   str = "split"
    train_value: str = "train"
    mask_col:    str = "m_actions"
    y_col:       str = "y_true"
    family_col:  str = "support_family"
    group_col:   str = "group_id"

    gsdp_divergence: str = "kl"

    rank_js_margin:    float = 0.0
    rank_js_mode:      str   = "any_excess"
    rank_js_reduction: str   = "mean"

    w_gsdp:    float = 1.0
    w_rank_js: float = 1.0


# ════════════════════════════════════════════════════════════════════
# §2  Clinical costs and L2D objective
# ════════════════════════════════════════════════════════════════════

C_FN_BASE: float = 1.8   # base cost of a false negative
C_FP_BASE: float = 1.2   # base cost of a false positive


# Per-expert scarcity costs (lower = cheaper to consult).
TIER_COST: Dict[str, float] = {
    "chaksu_expert_1": 0.310,
    "refuge_expert_1": 0.303,
    "chaksu_expert_3": 0.300,
    "refuge_expert_2": 0.300,
    "chaksu_expert_2": 0.253,
    "refuge_expert_4": 0.240,
    "refuge_expert_6": 0.240,
    "refuge_expert_7": 0.222,
    "chaksu_expert_4": 0.180,
    "refuge_expert_3": 0.180,
    "refuge_expert_5": 0.147,
    "chaksu_expert_5": 0.130,
}


# Discrete tier assignment (1 = scarcest, 4 = most available).
EXPERT_TIER: Dict[str, int] = {
    "chaksu_expert_1": 1,
    "refuge_expert_1": 1,
    "chaksu_expert_3": 1,
    "refuge_expert_2": 2,
    "chaksu_expert_2": 2,
    "refuge_expert_4": 2,
    "refuge_expert_6": 3,
    "refuge_expert_7": 3,
    "chaksu_expert_4": 3,
    "refuge_expert_3": 4,
    "refuge_expert_5": 4,
    "chaksu_expert_5": 4,
}


def action_costs(tier_cost: Dict[str, float],
                 experts: List[str]) -> np.ndarray:
    """``costs[a]`` — tier cost incurred by selecting action *a* (0 = AI)."""
    costs = np.zeros((1 + len(experts),), dtype=np.float32)
    for j, e in enumerate(experts):
        costs[1 + j] = float(tier_cost[e])
    return costs


def expert_clinical_cost(y_true: torch.Tensor, y_exp: torch.Tensor,
                         c_fn: float, c_fp: float) -> torch.Tensor:
    """Per-expert oracle clinical cost (NaN expert labels are zeroed)."""
    y_exp_filled = torch.nan_to_num(y_exp, nan=0.0)
    yt = y_true.unsqueeze(1)
    fn = (yt == 1) & (y_exp_filled == 0)
    fp = (yt == 0) & (y_exp_filled == 1)
    return c_fn * fn.float() + c_fp * fp.float()


def ai_expected_clinical_cost(y_true: torch.Tensor, p_ai: torch.Tensor,
                              c_fn: float, c_fp: float) -> torch.Tensor:
    """Differentiable expected FN/FP cost of the AI action."""
    yt = (y_true > 0.5).float()
    return c_fn * yt * (1.0 - p_ai) + c_fp * (1.0 - yt) * p_ai


def per_action_clinical_cost(y_true: torch.Tensor, p_ai: torch.Tensor,
                             y_exp: torch.Tensor, c_fn: float,
                             c_fp: float) -> torch.Tensor:
    """``[B, 1+M]`` — col 0 is the AI cost, cols 1..M are expert costs."""
    L_ai  = ai_expected_clinical_cost(y_true, p_ai, c_fn, c_fp)
    L_exp = expert_clinical_cost(y_true, y_exp, c_fn, c_fp)
    return torch.cat([L_ai.unsqueeze(1), L_exp], dim=1)


def l2d_objective(pi: torch.Tensor, action_mask: torch.Tensor,
                  y_true: torch.Tensor, p_ai: torch.Tensor,
                  y_exp: torch.Tensor, tier_costs: torch.Tensor,
                  c_fn: float, c_fp: float):
    """Returns ``(exp_clin, exp_tier, defer_rate, pi_masked)``."""
    pi = pi * action_mask
    pi = pi / (pi.sum(dim=1, keepdim=True) + 1e-8)

    L_clin = per_action_clinical_cost(y_true, p_ai, y_exp, c_fn, c_fp)

    exp_clin   = (pi * L_clin).sum(dim=1).mean()
    exp_tier   = (pi * tier_costs.view(1, -1)).sum(dim=1).mean()
    defer_rate = 1.0 - pi[:, 0].mean()
    return exp_clin, exp_tier, defer_rate, pi


# ════════════════════════════════════════════════════════════════════
# §3  Dataset
# ════════════════════════════════════════════════════════════════════

def prob1_from_logits_np(logit0: np.ndarray, logit1: np.ndarray) -> np.ndarray:
    """Numerically-stable ``softmax(logit)[1]`` on numpy."""
    m = np.maximum(logit0, logit1)
    e0 = np.exp(logit0 - m)
    e1 = np.exp(logit1 - m)
    p1 = e1 / (e0 + e1 + 1e-12)
    return np.clip(p1, 1e-6, 1.0 - 1e-6).astype(np.float32)


class L2DDataset(Dataset):
    """
    14-tuple per sample::

        (vim_risk, quality_risk, uncertainty,
         p_ai,
         vcdr, acdr,
         logit_0, logit_1,
         y_true, y_exp,
         m_actions, expert_mask,
         ds_id, group_id)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        self.expert_cols: List[str] = [
            c for c in df.columns if c.startswith("y_") and c != "y_true"
        ]

        expert_vals = self.df[self.expert_cols].to_numpy(dtype=np.float32)
        m_experts = (~np.isnan(expert_vals)).astype(np.float32)
        self.m_actions = np.concatenate(
            [np.ones((len(self.df), 1), dtype=np.float32), m_experts], axis=1
        )
        self.expert_mask = self.m_actions[:, 1:]

        self.expert_names = [
            c.removeprefix("y_") if c.startswith("y_") else c
            for c in self.expert_cols
        ]
        self.action_names = ["AI"] + self.expert_names
        self.expert_name_to_id = {n: i for i, n in enumerate(self.expert_names)}
        self.expert_id_to_name = {i: n for i, n in enumerate(self.expert_names)}

        self.y_true       = self.df["y_true"].astype(np.float32).to_numpy()
        self.vim_risk     = self.df["vim_risk_z"].astype(np.float32).to_numpy()
        self.quality_risk = self.df["quality_risk"].astype(np.float32).to_numpy()
        self.uncertainty  = self.df["uncertainty"].astype(np.float32).to_numpy()
        self.vcdr         = self.df["vCDR"].astype(np.float32).to_numpy()
        self.acdr         = self.df["aCDR"].astype(np.float32).to_numpy()

        self.ds_map = {n: i for i, n in enumerate(sorted(df["dataset"].unique()))}
        self.ds_ids = np.array(
            [self.ds_map[d] for d in df["dataset"].values], dtype=np.int64,
        )

        self.group_ids = self.df["group_id"].astype(np.int64).to_numpy()
        self.y_exp     = self.df[self.expert_cols].to_numpy(dtype=np.float32)

        if "prob_1" in self.df.columns:
            self.p_ai = np.clip(
                self.df["prob_1"].astype(np.float32).to_numpy(),
                1e-6, 1 - 1e-6,
            )
        else:
            self.p_ai = prob1_from_logits_np(
                self.df["logit_0"].values.astype(np.float32),
                self.df["logit_1"].values.astype(np.float32),
            )

        self.logit_0 = self.df["logit_0"].astype(np.float32).to_numpy()
        self.logit_1 = self.df["logit_1"].astype(np.float32).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        return (
            torch.tensor(self.vim_risk[i]),
            torch.tensor(self.quality_risk[i]),
            torch.tensor(self.uncertainty[i]),
            torch.tensor(self.p_ai[i]),
            torch.tensor(self.vcdr[i]),
            torch.tensor(self.acdr[i]),
            torch.tensor(self.logit_0[i]),
            torch.tensor(self.logit_1[i]),
            torch.tensor(self.y_true[i]),
            torch.tensor(self.y_exp[i]),
            torch.tensor(self.m_actions[i]),
            torch.tensor(self.expert_mask[i]),
            torch.tensor(self.ds_ids[i]),
            torch.tensor(self.group_ids[i]),
        )


# ════════════════════════════════════════════════════════════════════
# §4  Model
# ════════════════════════════════════════════════════════════════════

class StructuralRisk(nn.Module):
    """Calibrated linear glaucoma-risk head over ``(vCDR, aCDR)``."""

    def __init__(self, trainable: bool = True):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        coef = [[7.50903265, 8.04041925]]
        bias = [-6.84406693]
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(coef, dtype=torch.float32))
            self.linear.bias.copy_(torch.tensor(bias, dtype=torch.float32))
        if not trainable:
            for p in self.linear.parameters():
                p.requires_grad = False

    def forward(self, vcdr, acdr):
        x = torch.stack([vcdr, acdr], dim=-1)
        z = self.linear(x).squeeze(-1)
        return z, torch.sigmoid(z)


class MLPBlock(nn.Module):
    """``Linear → ReLU → LayerNorm → Dropout`` building block."""

    def __init__(self, in_dim: int, out_dim: int, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.net(x)


class OVAExpertHeadFixed(nn.Module):
    """Two-stage one-vs-all expert head with differentiable allocation."""

    def __init__(self, fused_dim: int, hidden: int, n_experts: int,
                 p_drop: float = 0.1, hard: bool = True):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(p_drop),
        )
        self.gate_logits  = nn.Linear(hidden, n_experts)
        self.alloc_logits = nn.Linear(hidden, n_experts)
        self.hard = hard

    def forward(self, h, mask, tau: float = 1.0):
        z = self.trunk(h)

        # ---- Stage 1: OVA gating ----
        g_logits = self.gate_logits(z) / tau
        g_logits = g_logits.masked_fill(mask <= 0, -1e8)
        gate_probs = torch.sigmoid(g_logits)

        if self.training:
            u = torch.rand_like(g_logits).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = torch.log(u) - torch.log(1 - u)
            selection_soft = torch.sigmoid((g_logits + gumbel_noise) / tau)
            if self.hard:
                selection = (
                    (selection_soft > 0.5).float()
                    - selection_soft.detach()
                    + selection_soft
                )
            else:
                selection = selection_soft
        else:
            selection = (gate_probs > 0.5).float()

        selection = selection * mask
        dead = (selection.sum(-1, keepdim=True) == 0).float()
        selection = selection + dead * mask

        # ---- Stage 2: allocation ----
        a_logits = self.alloc_logits(z) / tau
        a_logits_masked = a_logits.masked_fill(mask <= 0, -1e8)
        alloc_probs = torch.softmax(a_logits_masked, dim=-1)

        q = alloc_probs * selection
        q = q / q.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return q, gate_probs, selection, a_logits


class Router(nn.Module):
    """MPD2-Router: deferral gate + OVA expert allocator on fused features."""

    def __init__(self, n_experts: int, hidden: int = 64,
                 hidden_expert: int = 128, p_drop: float = 0.1,
                 init_tau: float = 1.25):
        super().__init__()

        self.struct = StructuralRisk(trainable=True)

        self.risk_enc   = MLPBlock(3, hidden, p_drop=p_drop)
        self.struct_enc = MLPBlock(2, hidden, p_drop=p_drop)
        self.ai_enc     = MLPBlock(2, hidden, p_drop=p_drop)

        fused_dim = 3 * hidden

        self.defer_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(p_drop),
            nn.Linear(128, 1),
        )

        self.expert_head = OVAExpertHeadFixed(
            fused_dim=fused_dim,
            hidden=hidden_expert,
            n_experts=n_experts,
            p_drop=p_drop,
            hard=True,
        )

        self.log_tau = nn.Parameter(
            torch.log(torch.tensor(init_tau, dtype=torch.float32))
        )

    def forward(self, vim_risk, quality_risk, uncertainty, vcdr, acdr,
                p_ai, logit_0, logit_1, action_mask):
        z_struct, p_struct = self.struct(vcdr, acdr)
        struct_margin = 2.0 * p_struct - 1.0
        ai_struct_gap = torch.abs(p_ai - p_struct)

        h_risk   = self.risk_enc(
            torch.stack([vim_risk, quality_risk, uncertainty], dim=-1)
        )
        h_struct = self.struct_enc(
            torch.stack([struct_margin, ai_struct_gap], dim=-1)
        )
        h_ai     = self.ai_enc(torch.stack([logit_0, logit_1], dim=-1))

        h = torch.cat([h_risk, h_struct, h_ai], dim=-1)

        d = torch.sigmoid(self.defer_head(h)).squeeze(-1)

        expert_mask = action_mask[:, 1:]
        tau = torch.exp(self.log_tau).clamp(min=0.5)
        q, gate_probs, selection, expert_logits = self.expert_head(
            h, expert_mask, tau,
        )

        pi = torch.cat([(1 - d).unsqueeze(-1), d.unsqueeze(-1) * q], dim=-1)

        aux = {
            "d":             d,
            "q":             q,
            "expert_logits": expert_logits,
            "expert_mask":   expert_mask,
            "gate_probs":    gate_probs,
            "selection":     selection,
            "z_struct":      z_struct,
            "p_struct":      p_struct,
            "struct_margin": struct_margin,
            "ai_struct_gap": ai_struct_gap,
            "tau":           tau,
            "risk_feats":    torch.stack(
                [vim_risk, quality_risk, uncertainty, z_struct, ai_struct_gap],
                dim=-1,
            ),
        }
        return pi, expert_logits, aux


# ════════════════════════════════════════════════════════════════════
# §5  Hierarchical priors
# ════════════════════════════════════════════════════════════════════

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
    """``[N, M]`` expert-only availability matrix."""
    return np.vstack(
        [parse_action_mask(x, M) for x in df[mask_col].tolist()]
    ).astype(np.float32)


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
    """Clip active entries to ``[0, clip_max]`` and renormalise."""
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


def _compute_expert_stats(df_sub: pd.DataFrame,
                          expert_cols: Sequence[str],
                          expert_mask_sub: np.ndarray,
                          y_col: str = "y_true",
                          alpha: float = 1.0, beta: float = 1.0,
                          c_fn: float = 2.0, c_fp: float = 1.0,
                          tier_cost: Optional[Sequence[float]] = None,
                          ) -> Dict[str, np.ndarray]:
    """Beta-smoothed FNR/FPR + ``badness = c_fn·FNR + c_fp·FPR + tier_cost``."""
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
    """Piecewise-linear interpolation of ``clip_max`` by active expert count."""
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
    """``p_group = λ_grp · p_raw + λ_fam · p_family + λ_glo · p_global``, then clipped."""
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


# ════════════════════════════════════════════════════════════════════
# §6  Routing-regularisation losses
# ════════════════════════════════════════════════════════════════════

def safe_masked_prob_torch(p: torch.Tensor, mask: torch.Tensor,
                           eps: float = 1e-8) -> torch.Tensor:
    """Mask, normalise, and floor a probability tensor for safe log-divergence."""
    p = torch.clamp(p, min=0.0) * mask
    Z = p.sum(dim=-1, keepdim=True)

    zero_rows = Z.squeeze(-1) <= eps
    if zero_rows.any():
        p = p.clone()
        u = mask[zero_rows]
        u = u / u.sum(dim=-1, keepdim=True).clamp_min(1.0)
        p[zero_rows] = u
        Z = p.sum(dim=-1, keepdim=True)

    p = p / Z.clamp_min(eps)
    p = p.clamp_min(eps) * mask
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    return p


def kl_div_prob(q: torch.Tensor, p: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    q = q.clamp_min(eps)
    p = p.clamp_min(eps)
    return (q * (q.log() - p.log())).sum(dim=-1)


def js_div_prob(p: torch.Tensor, q: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """Row-wise Jensen–Shannon divergence."""
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1)
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def gsdp_loss(q: torch.Tensor, expert_mask: torch.Tensor,
              group_ids: torch.Tensor,
              group_prior_table: torch.Tensor,
              group_id_to_row: Dict[int, int],
              d: Optional[torch.Tensor] = None,
              divergence: str = "kl",
              eps: float = 1e-8) -> torch.Tensor:
    """GSDP on group-average conditional routing."""
    q = safe_masked_prob_torch(q, expert_mask, eps=eps)

    if d is None:
        w = torch.ones(q.size(0), device=q.device, dtype=q.dtype)
    else:
        w = d.detach().to(device=q.device, dtype=q.dtype)

    total_loss = q.new_zeros(())
    total_w    = q.new_zeros(())

    for g in torch.unique(group_ids).tolist():
        g = int(g)
        idx = group_ids == g
        wg = w[idx]
        group_mass = wg.sum()
        if group_mass <= eps:
            continue

        qg = q[idx]
        mg = expert_mask[idx]

        qbar = (wg[:, None] * qg).sum(dim=0) / group_mass.clamp_min(eps)
        active_union = ((wg[:, None] * mg).sum(dim=0) > 0).to(q.dtype)

        p = group_prior_table[group_id_to_row[g]].to(
            device=q.device, dtype=q.dtype,
        )

        qbar = safe_masked_prob_torch(
            qbar.unsqueeze(0), active_union.unsqueeze(0), eps=eps,
        )[0]
        p = safe_masked_prob_torch(
            p.unsqueeze(0), active_union.unsqueeze(0), eps=eps,
        )[0]

        if divergence.lower() == "kl":
            term = kl_div_prob(qbar.unsqueeze(0), p.unsqueeze(0), eps=eps)[0]
        elif divergence.lower() == "js":
            term = js_div_prob(qbar.unsqueeze(0), p.unsqueeze(0), eps=eps)[0]
        else:
            raise ValueError("divergence must be 'kl' or 'js'")

        total_loss = total_loss + group_mass * term
        total_w    = total_w + group_mass

    return total_loss / total_w.clamp_min(eps)


def truncated_geometric_prior(k: int, rho: float, device=None,
                              dtype=torch.float32) -> torch.Tensor:
    ranks = torch.arange(k, device=device, dtype=dtype)
    g = (1.0 - rho) * (rho ** ranks)
    return g / g.sum().clamp_min(1e-8)


def default_rho_by_k(max_k: int = 12) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k in range(2, max_k + 1):
        if k == 2:
            out[k] = 0.67
        elif k == 3:
            out[k] = 0.80
        elif k == 4:
            out[k] = 0.75
        else:
            out[k] = 0.72
    return out


def rank_majorization_js_loss(q: torch.Tensor, expert_mask: torch.Tensor,
                              d: Optional[torch.Tensor] = None,
                              rho_by_k: Optional[Dict[int, float]] = None,
                              margin: float = 0.0,
                              mode: str = "any_excess",
                              reduction: str = "mean",
                              eps: float = 1e-8) -> torch.Tensor:
    """Majorisation-gated JS against a truncated-geometric rank prior."""
    if q.ndim != 2 or expert_mask.ndim != 2:
        raise ValueError("q and expert_mask must be [B, M].")
    if q.shape != expert_mask.shape:
        raise ValueError("q and expert_mask must have the same shape.")

    B, M = q.shape
    device, dtype = q.device, q.dtype

    if rho_by_k is None:
        rho_by_k = default_rho_by_k(M)

    q = safe_masked_prob_torch(q, expert_mask, eps=eps)

    sample_w = (
        torch.ones(B, device=device, dtype=dtype)
        if d is None
        else d.detach().to(device=device, dtype=dtype)
    )

    k_active = (expert_mask > 0).sum(dim=-1).long()

    total_loss = torch.zeros((), device=device, dtype=dtype)
    total_w    = torch.zeros((), device=device, dtype=dtype)

    for k_t in torch.unique(k_active):
        k = int(k_t.item())
        if k < 2:
            continue
        idx = k_active == k
        if not idx.any():
            continue

        qk = q[idx]
        wk = sample_w[idx]

        rk, _ = torch.sort(qk, dim=-1, descending=True)
        rk = rk[:, :k]

        rho = float(rho_by_k.get(k, 0.72))
        gk = truncated_geometric_prior(k, rho, device=device, dtype=dtype)
        gk = gk.unsqueeze(0).expand_as(rk)

        cr = torch.cumsum(rk, dim=-1)[:, :-1]
        cg = torch.cumsum(gk, dim=-1)[:, :-1]
        diff = cr - cg

        if mode == "any_excess":
            gate = (diff > margin).any(dim=-1)
        elif mode == "all_prefixes":
            gate = (diff >= -margin).all(dim=-1)
        else:
            raise ValueError("mode must be 'any_excess' or 'all_prefixes'")

        js = js_div_prob(rk, gk, eps=eps)
        per_sample = gate.to(dtype) * js

        total_loss = total_loss + (wk * per_sample).sum()
        total_w    = total_w + wk.sum()

    if reduction == "sum":
        return total_loss
    if reduction == "mean":
        return total_loss / total_w.clamp_min(eps)
    raise ValueError("reduction must be 'mean' or 'sum'")


def combined_routing_loss(q: torch.Tensor, expert_mask: torch.Tensor,
                          group_ids: torch.Tensor,
                          group_prior_table: torch.Tensor,
                          group_id_to_row: Dict[int, int],
                          cfg: PriorRegConfig,
                          d: Optional[torch.Tensor] = None,
                          ) -> Dict[str, torch.Tensor]:
    """``cfg.w_gsdp · GSDP + cfg.w_rank_js · Rank-JS``."""
    l_gsdp = gsdp_loss(
        q, expert_mask, group_ids,
        group_prior_table, group_id_to_row,
        d=d, divergence=cfg.gsdp_divergence,
    )
    l_rank = rank_majorization_js_loss(
        q, expert_mask, d=d,
        margin=cfg.rank_js_margin,
        mode=cfg.rank_js_mode,
        reduction=cfg.rank_js_reduction,
    )
    total = cfg.w_gsdp * l_gsdp + cfg.w_rank_js * l_rank
    return {"total": total, "gsdp": l_gsdp, "rank_js": l_rank}


# ════════════════════════════════════════════════════════════════════
# §7  Augmented Lagrangian
# ════════════════════════════════════════════════════════════════════

class AugLag:
    """Stateful augmented-Lagrangian helper with non-negative dual variables."""

    def __init__(self, cfg: ALConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.lam_def  = torch.zeros((), device=device)
        self.lam_cost = torch.zeros((), device=device)

    def penalty(self, defer_rate: torch.Tensor,
                avg_cost: torch.Tensor) -> torch.Tensor:
        pen = torch.zeros((), device=self.device)

        if self.cfg.max_deferral_rate is not None:
            g = defer_rate - self.cfg.max_deferral_rate
            pen = pen + self.lam_def * g + 0.5 * self.cfg.mu * torch.clamp(g, min=0.0) ** 2

        if self.cfg.max_avg_cost is not None:
            g = avg_cost - self.cfg.max_avg_cost
            pen = pen + self.lam_cost * g + 0.5 * self.cfg.mu * torch.clamp(g, min=0.0) ** 2

        return pen

    @torch.no_grad()
    def update(self, defer_rate: torch.Tensor, avg_cost: torch.Tensor) -> None:
        if self.cfg.max_deferral_rate is not None:
            g = defer_rate - self.cfg.max_deferral_rate
            self.lam_def = torch.clamp(
                self.lam_def + self.cfg.lr_lambda * g, min=0.0,
            )

        if self.cfg.max_avg_cost is not None:
            g = avg_cost - self.cfg.max_avg_cost
            self.lam_cost = torch.clamp(
                self.lam_cost + self.cfg.lr_lambda * g, min=0.0,
            )


# ════════════════════════════════════════════════════════════════════
# §8  Evaluation and reporting
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(router, dl, tier_costs: torch.Tensor,
             cfg: TrainConfig, mcfg: PriorRegConfig,
             group_prior_table: torch.Tensor,
             group_id_to_row: Dict[int, int],
             al_cfg: ALConfig,
             device: torch.device,
             tag: str = "VAL") -> dict:
    """Held-out evaluation pass; returns full per-action / per-dataset breakdown."""
    router.eval()
    n_experts = len(dl.dataset.expert_cols)
    num_actions = 1 + n_experts

    N = 0
    sum_acc = sum_clin = 0.0
    sum_tier_hard = sum_tier_soft = 0.0
    sum_defer_hard = sum_defer_soft = 0.0
    sum_gsdp = sum_rank_js = 0.0

    action_count     = torch.zeros(num_actions)
    action_correct   = torch.zeros(num_actions)
    action_fn        = torch.zeros(num_actions)
    action_fp        = torch.zeros(num_actions)
    action_tp        = torch.zeros(num_actions)
    action_tn        = torch.zeros(num_actions)
    action_soft_mass = torch.zeros(num_actions)

    all_preds, all_trues, all_probs, all_actions = [], [], [], []
    all_ds, all_d = [], []

    for (vim, q, u, p_ai, vcdr, acdr, logit_0, logit_1, y_true, y_exp,
         m_actions, expert_mask, ds_ids, group_ids) in dl:

        vim, q, u = vim.to(device), q.to(device), u.to(device)
        vcdr, acdr = vcdr.to(device), acdr.to(device)
        p_ai = p_ai.to(device)
        logit_0, logit_1 = logit_0.to(device), logit_1.to(device)
        y_true, y_exp = y_true.to(device), y_exp.to(device)
        m_actions = m_actions.to(device)
        expert_mask = expert_mask.to(device)
        ds_ids, group_ids = ds_ids.to(device), group_ids.to(device)
        B = y_true.shape[0]

        pi, _, aux = router(
            vim, q, u, vcdr, acdr, p_ai, logit_0, logit_1, m_actions,
        )
        pi = pi * m_actions
        pi = pi / (pi.sum(dim=1, keepdim=True) + 1e-8)

        d = aux["d"].view(-1)
        q_route = aux["q"].view(B, n_experts)
        expert_mask_route = m_actions[:, 1:].to(q_route.dtype)

        all_d.append(d.cpu())
        all_ds.append(ds_ids.cpu())

        reg = combined_routing_loss(
            q=q_route, expert_mask=expert_mask_route,
            group_ids=group_ids,
            group_prior_table=group_prior_table,
            group_id_to_row=group_id_to_row,
            cfg=mcfg, d=d,
        )
        sum_gsdp    += reg["gsdp"].item() * B
        sum_rank_js += reg["rank_js"].item() * B

        defer_soft = 1.0 - pi[:, 0].mean()
        tier_soft  = (pi * tier_costs.view(1, -1)).sum(dim=1).mean()
        action_soft_mass += pi.sum(dim=0).cpu()

        a = torch.argmax(pi, dim=1)
        defer_hard = (a > 0).float().mean()
        tier_hard  = tier_costs[a].mean()

        y_true01 = (y_true > 0.5).long()
        pred = torch.empty_like(y_true01)
        pred_ai = (p_ai >= 0.5).long()
        y_exp_filled = torch.nan_to_num(y_exp, nan=0.0)

        ai_mask = (a == 0)
        ex_mask = (a > 0)
        pred[ai_mask] = pred_ai[ai_mask]
        if ex_mask.any():
            j = a[ex_mask] - 1
            pred[ex_mask] = y_exp_filled[ex_mask, j].long()

        # Soft probability for AUPRC: policy-weighted mixture
        p_pos = pi[:, 0] * p_ai + (pi[:, 1:] * y_exp_filled).sum(dim=1)
        all_probs.append(p_pos.cpu())

        acc = (pred == y_true01).float().mean()
        fn = (y_true01 == 1) & (pred == 0)
        fp = (y_true01 == 0) & (pred == 1)
        clin = (cfg.c_fn * fn.float() + cfg.c_fp * fp.float()).mean()

        for k in range(num_actions):
            mask_k = (a == k)
            count_k = mask_k.sum().item()
            action_count[k] += count_k
            if count_k > 0:
                action_correct[k] += (pred[mask_k] == y_true01[mask_k]).sum().item()
                action_fn[k] += ((y_true01[mask_k] == 1) & (pred[mask_k] == 0)).sum().item()
                action_fp[k] += ((y_true01[mask_k] == 0) & (pred[mask_k] == 1)).sum().item()
                action_tp[k] += ((y_true01[mask_k] == 1) & (pred[mask_k] == 1)).sum().item()
                action_tn[k] += ((y_true01[mask_k] == 0) & (pred[mask_k] == 0)).sum().item()

        all_preds.append(pred.cpu())
        all_trues.append(y_true01.cpu())
        all_actions.append(a.cpu())

        sum_acc        += acc.item() * B
        sum_clin       += clin.item() * B
        sum_tier_hard  += tier_hard.item() * B
        sum_tier_soft  += tier_soft.item() * B
        sum_defer_hard += defer_hard.item() * B
        sum_defer_soft += defer_soft.item() * B
        N += B

    all_d_cat   = torch.cat(all_d)
    all_preds_c = torch.cat(all_preds)
    all_trues_c = torch.cat(all_trues)
    all_probs_c = torch.cat(all_probs)

    q_mass = action_soft_mass[1:] / max(action_soft_mass[1:].sum().item(), 1e-8)
    top1_share = q_mass.max().item()
    top1_idx   = q_mass.argmax().item()

    y_np    = all_trues_c.numpy()
    p_np    = all_probs_c.numpy()
    pred_np = all_preds_c.numpy()
    auprc = (
        float(average_precision_score(y_np, p_np))
        if len(np.unique(y_np)) > 1 else float("nan")
    )
    mcc = float(matthews_corrcoef(y_np, pred_np))

    inv_expert_map = dl.dataset.expert_id_to_name
    action_names = ["AI"] + [
        inv_expert_map[i] for i in range(len(inv_expert_map))
    ]

    results = {
        "acc":              sum_acc / N,
        "clinical":         sum_clin / N,
        "tier_hard":        sum_tier_hard / N,
        "tier_soft":        sum_tier_soft / N,
        "defer_hard":       sum_defer_hard / N,
        "defer_soft":       sum_defer_soft / N,
        "gsdp":             sum_gsdp / N,
        "rank_js":          sum_rank_js / N,
        "auprc":            auprc,
        "mcc":              mcc,
        "N":                N,
        "tag":              tag,
        "action_count":     action_count,
        "action_correct":   action_correct,
        "action_fn":        action_fn,
        "action_fp":        action_fp,
        "action_tp":        action_tp,
        "action_tn":        action_tn,
        "action_soft_mass": action_soft_mass,
        "all_preds":        all_preds_c,
        "all_trues":        all_trues_c,
        "all_probs":        all_probs_c,
        "all_actions":      torch.cat(all_actions),
        "d_mean":           all_d_cat.mean().item(),
        "d_std":            all_d_cat.std().item(),
        "q_mass":           q_mass,
        "top1_expert":      top1_idx,
        "top1_share":       top1_share,
        "all_ds":           torch.cat(all_ds),
        "action_names":     action_names,
        "inv_expert_map":   inv_expert_map,
    }

    gamma_eval = getattr(cfg, "gamma_tier", 1.0)
    dmax = getattr(al_cfg, "max_deferral_rate", None)
    es_base = results["clinical"] + gamma_eval * results["tier_soft"]
    es_violation = (
        max(0.0, results["defer_soft"] - dmax) if dmax is not None else 0.0
    )
    results["es_base"]      = es_base
    results["es_violation"] = es_violation
    results["total_cost"]   = float(
        results["clinical"] + gamma_eval * results["tier_hard"]
    )

    print(
        f"[{tag}] acc={results['acc']:.4f} clin={results['clinical']:.4f} "
        f"auprc={auprc:.4f} mcc={mcc:.4f} "
        f"total_cost={results['total_cost']:.4f} "
        f"tier_h={results['tier_hard']:.4f} "
        f"def_h={results['defer_hard']:.4f} def_s={results['defer_soft']:.4f} "
        f"score={es_base:.4f} viol={es_violation:.4f} "
        f"d={results['d_mean']:.3f}±{results['d_std']:.3f} "
        f"top1=exp{top1_idx}({top1_share:.1%}) "
        f"gsdp={results['gsdp']:.4f} rank_js={results['rank_js']:.4f}"
    )
    return results


def diagnostic_checks(router, aux, q_route, expert_mask_route) -> None:
    """Lightweight gradient/utilisation telemetry; call after ``loss.backward()``."""
    with torch.no_grad():
        expert_grad_norm = sum(
            p.grad.norm().item()
            for p in router.expert_head.parameters() if p.grad is not None
        )
        defer_grad_norm = sum(
            p.grad.norm().item()
            for p in router.defer_head.parameters() if p.grad is not None
        )
        ratio = expert_grad_norm / max(defer_grad_norm, 1e-8)
        print(f"  [diag] grad norms — expert: {expert_grad_norm:.4f}, "
              f"defer: {defer_grad_norm:.4f}, ratio: {ratio:.2f}")

        gate_probs = aux["gate_probs"]
        avg_open = (gate_probs > 0.5).float().sum(dim=-1).mean().item()
        avg_available = expert_mask_route.sum(dim=-1).mean().item()
        select_ratio = avg_open / max(avg_available, 1e-8)
        print(f"  [diag] avg experts selected: {avg_open:.1f} / "
              f"{avg_available:.1f} ({select_ratio:.1%})")

        q_mass = (q_route * expert_mask_route).sum(dim=0)
        q_mass = q_mass / q_mass.sum().clamp(min=1e-8)
        sorted_mass, _ = q_mass.sort()
        n = len(sorted_mass)
        idx = torch.arange(1, n + 1, dtype=torch.float32, device=sorted_mass.device)
        gini = (
            2 * (idx * sorted_mass).sum() / (n * sorted_mass.sum())
            - (n + 1) / n
        ).item()
        print(f"  [diag] q_mass Gini: {gini:.3f}, "
              f"top-3: {sorted_mass[-3:].tolist()}")


def print_eval_report(results: dict, dl=None) -> None:
    """Global metrics + per-action breakdown."""
    tag = results["tag"]
    N = results["N"]

    if dl is not None:
        action_names = dl.dataset.action_names
    else:
        action_names = results.get(
            "action_names",
            ["AI"] + [
                f"Exp{k}"
                for k in range(1, int(results["all_actions"].max().item()) + 1)
            ],
        )
    num_actions = len(action_names)

    all_preds = results["all_preds"]
    all_trues = results["all_trues"]
    all_probs = results.get("all_probs", None)

    global_tp = ((all_trues == 1) & (all_preds == 1)).sum().item()
    global_fn = ((all_trues == 1) & (all_preds == 0)).sum().item()
    global_fp = ((all_trues == 0) & (all_preds == 1)).sum().item()
    global_tn = ((all_trues == 0) & (all_preds == 0)).sum().item()

    precision = global_tp / max(global_tp + global_fp, 1)
    recall    = global_tp / max(global_tp + global_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    specificity = global_tn / max(global_tn + global_fp, 1)

    trues_np = all_trues.cpu().numpy()
    preds_np = all_preds.cpu().numpy()
    mcc = matthews_corrcoef(trues_np, preds_np)
    auprc = (
        average_precision_score(trues_np, all_probs.cpu().numpy())
        if all_probs is not None and len(np.unique(trues_np)) > 1
        else float("nan")
    )

    score_val = results["es_base"] if results["es_base"] is not None else results["total_cost"]
    print(f"\n{'='*70}")
    print(f"[{tag}] FULL EVALUATION REPORT  (N={N})")
    print(f"{'='*70}")
    print(f"\n--- Global Metrics ---")
    print(f"  Accuracy:      {results['acc']:.4f}")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  F1:            {f1:.4f}")
    print(f"  Specificity:   {specificity:.4f}")
    print(f"  MCC:           {mcc:.4f}")
    print(f"  AUPRC:         {auprc:.4f}")
    print(f"  Clinical cost: {results['clinical']:.4f}  "
          f"(FN={global_fn}, FP={global_fp})")

    print(f"\n--- Deferral & Cost ---")
    print(f"  Defer (hard): {results['defer_hard']:.4f}   "
          f"Defer (soft): {results['defer_soft']:.4f}")
    print(f"  Tier  (hard): {results['tier_hard']:.4f}   "
          f"Tier  (soft): {results['tier_soft']:.4f}")
    print(f"  Total score:  {score_val:.4f}")

    print(f"\n--- Global Confusion Matrix ---")
    print(f"               Pred=0    Pred=1")
    print(f"  True=0 (TN)  {global_tn:>6}    (FP) {global_fp:>6}")
    print(f"  True=1 (FN)  {global_fn:>6}    (TP) {global_tp:>6}")

    action_count     = results["action_count"]
    action_correct   = results["action_correct"]
    action_tp_a      = results["action_tp"]
    action_fn_a      = results["action_fn"]
    action_fp_a      = results["action_fp"]
    action_soft_mass = results["action_soft_mass"]

    print(f"\n--- Per-Action Breakdown ---")
    print(f"  {'Action':<12} {'Count':>6} {'%Routed':>8} {'Acc':>7} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'FN':>5} {'FP':>5} {'SoftMass':>9}")
    print(f"  {'-'*80}")

    for k in range(num_actions):
        cnt = int(action_count[k].item())
        pct = 100.0 * cnt / N if N > 0 else 0.0

        if cnt > 0:
            acc_k  = action_correct[k].item() / cnt
            tp_k   = action_tp_a[k].item()
            fn_k   = action_fn_a[k].item()
            fp_k   = action_fp_a[k].item()
            prec_k = tp_k / (tp_k + fp_k + 1e-8)
            rec_k  = tp_k / (tp_k + fn_k + 1e-8)
            f1_k   = 2 * prec_k * rec_k / (prec_k + rec_k + 1e-8)
        else:
            acc_k = prec_k = rec_k = f1_k = 0.0
            fn_k = fp_k = 0

        soft_pct = 100.0 * action_soft_mass[k].item() / N if N > 0 else 0.0
        print(f"  {action_names[k]:<12} {cnt:>6} {pct:>7.1f}% "
              f"{acc_k:>7.4f} {prec_k:>7.4f} {rec_k:>7.4f} {f1_k:>7.4f} "
              f"{int(fn_k):>5} {int(fp_k):>5} {soft_pct:>8.1f}%")

    print(f"{'='*70}\n")


def print_eval_report_dataset(results: dict, dl=None) -> None:
    """Same metrics broken down per source dataset (REFUGE/ORIGA/CHAKSU)."""
    all_targets = results["all_trues"]
    all_preds   = results["all_preds"]
    all_actions = results["all_actions"]
    all_probs   = results["all_probs"]
    all_ds      = results["all_ds"]

    if "action_names" in results:
        action_names = results["action_names"]
    elif dl is not None:
        action_names = dl.dataset.action_names
    else:
        num_act = int(all_actions.max().item()) + 1
        action_names = ["AI"] + [f"Exp{k}" for k in range(1, num_act)]
    num_actions = len(action_names)

    if dl is not None:
        inv_ds_map = {v: k for k, v in dl.dataset.ds_map.items()}
    else:
        inv_ds_map = {i: f"ds_{i}" for i in sorted(all_ds.unique().tolist())}

    for ds_idx, ds_name in inv_ds_map.items():
        mask_ds = (all_ds == ds_idx)
        N_ds = int(mask_ds.sum().item())
        if N_ds == 0:
            print(f"\n{'='*70}\n  {ds_name}  (N=0, skipped)")
            continue

        t_ds = all_targets[mask_ds]
        p_ds = all_preds[mask_ds]
        a_ds = all_actions[mask_ds]
        probs_ds = all_probs[mask_ds]

        acc_ds   = (p_ds == t_ds).float().mean().item()
        fn_ds    = ((t_ds == 1) & (p_ds == 0)).sum().item()
        fp_ds    = ((t_ds == 0) & (p_ds == 1)).sum().item()
        tp_ds    = ((t_ds == 1) & (p_ds == 1)).sum().item()
        tn_ds    = ((t_ds == 0) & (p_ds == 0)).sum().item()
        defer_ds = (a_ds > 0).float().mean().item()
        mcc_ds   = matthews_corrcoef(t_ds.cpu().numpy(), p_ds.cpu().numpy())
        auprc_ds = (
            average_precision_score(t_ds.cpu().numpy(), probs_ds.cpu().numpy())
            if len(t_ds.unique()) > 1 else float("nan")
        )

        print(f"\n{'='*70}")
        print(f"  {ds_name}  (N={N_ds}, acc={acc_ds:.4f}, MCC={mcc_ds:.4f}, "
              f"AUPRC={auprc_ds:.4f}, defer={defer_ds:.3f}, "
              f"FN={int(fn_ds)}, FP={int(fp_ds)}, "
              f"TP={int(tp_ds)}, TN={int(tn_ds)})")
        print(f"{'='*70}")

        print(f"  {'Action':<25} {'Count':>6} {'Share':>7} "
              f"{'Acc':>7} {'FN':>5} {'FP':>5} "
              f"{'TP':>5} {'TN':>5} {'Sens':>7} {'Spec':>7}")
        print(f"  {'-'*85}")

        for k in range(num_actions):
            mask_k = (a_ds == k)
            count_k = int(mask_k.sum().item())
            if count_k == 0:
                print(f"  {action_names[k]:<25} {0:>6} {0:>7.1%}")
                continue
            tk = t_ds[mask_k]
            pk = p_ds[mask_k]
            acc_k  = (pk == tk).float().mean().item()
            fn_k   = ((tk == 1) & (pk == 0)).sum().item()
            fp_k   = ((tk == 0) & (pk == 1)).sum().item()
            tp_k   = ((tk == 1) & (pk == 1)).sum().item()
            tn_k   = ((tk == 0) & (pk == 0)).sum().item()
            sens_k = tp_k / max(tp_k + fn_k, 1)
            spec_k = tn_k / max(tn_k + fp_k, 1)
            share_k = count_k / N_ds
            print(f"  {action_names[k]:<25} {count_k:>6} {share_k:>7.1%} "
                  f"{acc_k:>7.3f} {fn_k:>5} {fp_k:>5} "
                  f"{tp_k:>5} {tn_k:>5} {sens_k:>7.3f} {spec_k:>7.3f}")


# ════════════════════════════════════════════════════════════════════
# §9  Training loop
# ════════════════════════════════════════════════════════════════════

def _make_optimizer(router: Router, lr: float) -> torch.optim.Optimizer:
    """AdamW with separate parameter groups; ``log_tau`` has zero weight decay."""
    return torch.optim.AdamW([
        {"params": list(router.risk_enc.parameters()),    "lr": lr, "weight_decay": 1e-4},
        {"params": list(router.struct_enc.parameters()),  "lr": lr, "weight_decay": 1e-4},
        {"params": list(router.ai_enc.parameters()),      "lr": lr, "weight_decay": 1e-4},
        {"params": list(router.defer_head.parameters()),  "lr": lr, "weight_decay": 1e-4},
        {"params": list(router.expert_head.parameters()), "lr": lr, "weight_decay": 1e-4},
        {"params": [router.log_tau],                      "lr": lr, "weight_decay": 0.0},
        {"params": list(router.struct.parameters()),      "lr": lr, "weight_decay": 1e-4},
    ])


def train_l2d_multi_expert(df: pd.DataFrame,
                           cfg: Optional[TrainConfig] = None,
                           mcfg: Optional[PriorRegConfig] = None,
                           al_cfg: Optional[ALConfig] = None,
                           shuffle: bool = False):
    """
    End-to-end training driver.

    Returns ``(router, priors, all_q, all_pi)``; the router is reloaded with
    the best-scoring checkpoint before returning.
    """
    cfg = cfg or TrainConfig()
    mcfg = mcfg or PriorRegConfig()
    al_cfg = al_cfg or ALConfig()
    device = torch.device(cfg.device)

    EXPERT_COLS = [c for c in df.columns
                   if c.startswith("y_") and c != "y_true"]
    EXPERTS = [c.removeprefix("y_") for c in EXPERT_COLS]
    N_EXPERTS = len(EXPERT_COLS)

    df["m_experts"] = df.apply(
        lambda row: [int(not pd.isna(row[c])) for c in EXPERT_COLS], axis=1,
    )
    df["m_actions"] = df["m_experts"].apply(lambda m: [1] + m)

    costs_np = action_costs(TIER_COST, EXPERTS)
    tier_costs = torch.tensor(costs_np, dtype=torch.float32, device=device)

    priors = build_all_priors(df, EXPERT_COLS, cfg=mcfg)
    group_prior_table = priors["final_group_table"].to(device)
    group_id_to_row = priors["final_id_to_row"]

    df_tr = df[df["split"].astype(str) == "train"].reset_index(drop=True)
    df_va = (
        df[df["split"].astype(str) == "val"].reset_index(drop=True)
        if (df["split"].astype(str) == "val").any() else None
    )
    print(f"Train: {len(df_tr)}, Val: {len(df_va) if df_va is not None else 0}")

    ds_tr = L2DDataset(df_tr)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=shuffle)
    dl_va = None
    if df_va is not None and len(df_va) > 0:
        ds_va = L2DDataset(df_va)
        dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False)

    router = Router(N_EXPERTS, hidden=128).to(device)
    opt = _make_optimizer(router, lr=cfg.lr)
    al = AugLag(al_cfg, device=device)

    best_score = float("inf")
    best_epoch = 0
    best_state = None
    patience_ctr = 0

    all_pi: list = []
    all_q: list = []

    for ep in range(1, cfg.epochs + 1):
        router.train()
        total_loss = n_epoch = 0
        defer_sum = tier_sum = gsdp_sum = rank_js_sum = 0.0

        for batch_idx, (vim, q, u, p_ai, vcdr, acdr, logit_0, logit_1,
                        y_true, y_exp, m_actions, expert_mask, ds_ids,
                        group_ids) in enumerate(dl_tr):

            vim, q, u = vim.to(device), q.to(device), u.to(device)
            vcdr, acdr = vcdr.to(device), acdr.to(device)
            p_ai = p_ai.to(device)
            logit_0, logit_1 = logit_0.to(device), logit_1.to(device)
            y_true, y_exp = y_true.to(device), y_exp.to(device)
            m_actions = m_actions.to(device)
            expert_mask = expert_mask.to(device)
            ds_ids, group_ids = ds_ids.to(device), group_ids.to(device)
            B = y_true.size(0)

            pi, expert_logits, aux = router(
                vim, q, u, vcdr, acdr, p_ai, logit_0, logit_1, m_actions,
            )
            aux = dict(aux)
            expert_mask = m_actions[:, 1:].to(expert_logits.dtype)
            aux["expert_logits"] = expert_logits
            aux["expert_mask"]   = expert_mask

            d = aux["d"].view(-1)
            q_route = aux["q"].view(B, N_EXPERTS)
            expert_mask_route = m_actions[:, 1:].to(q_route.dtype)

            all_q.append(q_route.detach().cpu())
            all_pi.append(pi.detach().cpu())

            exp_clin, exp_tier, defer_rate, _ = l2d_objective(
                pi=pi, action_mask=m_actions,
                y_true=y_true, p_ai=p_ai, y_exp=y_exp,
                tier_costs=tier_costs,
                c_fn=cfg.c_fn, c_fp=cfg.c_fp,
            )
            reg = combined_routing_loss(
                q=q_route, expert_mask=expert_mask_route,
                group_ids=group_ids,
                group_prior_table=group_prior_table,
                group_id_to_row=group_id_to_row,
                cfg=mcfg, d=d,
            )
            pen = al.penalty(defer_rate, exp_tier)
            loss = (
                exp_clin
                + cfg.gamma_tier * exp_tier
                + pen
                + mcfg.w_gsdp * reg["gsdp"]
                + mcfg.w_rank_js * reg["rank_js"]
            )

            opt.zero_grad()
            loss.backward()
            if batch_idx % 50 == 0:
                diagnostic_checks(router, aux, q_route, expert_mask_route)
            opt.step()

            total_loss  += loss.item() * B
            defer_sum   += defer_rate.detach().item() * B
            tier_sum    += exp_tier.detach().item() * B
            gsdp_sum    += reg["gsdp"].item() * B
            rank_js_sum += reg["rank_js"].item() * B
            n_epoch     += B

        defer_mean = torch.tensor(defer_sum / max(n_epoch, 1), device=device)
        tier_mean  = torch.tensor(tier_sum  / max(n_epoch, 1), device=device)
        al.update(defer_mean, tier_mean)
        tau_now = torch.exp(router.log_tau).item()

        print(
            f"[Ep {ep:03d}] loss={total_loss / max(n_epoch, 1):.4f} "
            f"defer={defer_mean.item():.3f} tier={tier_mean.item():.4f} "
            f"gsdp={gsdp_sum / max(n_epoch, 1):.4f} "
            f"rank_js={rank_js_sum / max(n_epoch, 1):.4f} "
            f"tau={tau_now:.3f} lam_def={al.lam_def.item():.3f}"
        )

        if dl_va is not None:
            val_results = evaluate(
                router, dl_va, tier_costs, cfg, mcfg,
                group_prior_table, group_id_to_row, al_cfg, device, tag="VAL",
            )
            score = val_results["es_base"]
            if score < best_score - cfg.min_delta:
                if ep > cfg.warmup_epochs:
                    best_score = score
                    best_epoch = ep
                    patience_ctr = 0
                    best_state = copy.deepcopy(router.state_dict())
            elif ep > cfg.warmup_epochs:
                patience_ctr += 1
                if patience_ctr >= cfg.patience:
                    print(
                        f"Early stopping at epoch {ep}, best epoch {best_epoch}, "
                        f"score={best_score:.4f}"
                    )
                    router.load_state_dict(best_state)
                    break

    if dl_va is not None and best_state is not None:
        router.load_state_dict(best_state)
        print(f"Loaded best model from epoch {best_epoch}, "
              f"score={best_score:.4f}")

    return router, priors, all_q, all_pi


__all__ = [
    # §1 configs
    "TrainConfig", "ALConfig", "PriorRegConfig",
    # §2 costs / objective
    "C_FN_BASE", "C_FP_BASE", "TIER_COST", "EXPERT_TIER",
    "action_costs", "expert_clinical_cost", "ai_expected_clinical_cost",
    "per_action_clinical_cost", "l2d_objective",
    # §3 dataset
    "L2DDataset", "prob1_from_logits_np",
    # §4 model
    "StructuralRisk", "MLPBlock", "OVAExpertHeadFixed", "Router",
    # §5 priors
    "parse_action_mask", "prepare_expert_mask",
    "compute_global_prior", "compute_family_prior", "compute_group_prior",
    "build_prior_tensor", "build_all_priors",
    # §6 losses
    "safe_masked_prob_torch", "kl_div_prob", "js_div_prob",
    "gsdp_loss", "truncated_geometric_prior", "default_rho_by_k",
    "rank_majorization_js_loss", "combined_routing_loss",
    # §7 augmented Lagrangian
    "AugLag",
    # §8 evaluation / reporting
    "evaluate", "diagnostic_checks",
    "print_eval_report", "print_eval_report_dataset",
    # §9 training
    "train_l2d_multi_expert",
]
