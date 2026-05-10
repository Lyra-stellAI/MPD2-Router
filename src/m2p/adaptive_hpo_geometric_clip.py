"""
Autonomous Adaptive Hyperparameter Optimization for MPD²-Router
================================================================

Updated version for lbl+ova / MPD²-Router training with:
  1. clip_ceiling + clip_slack geometric clip caps for group priors,
  2. corrected HPO search-space keys,
  3. fixed config materialization,
  4. more flexible notebook integration for dataset/router signatures,
  5. constraint-aware validation selection for pruning / early stopping.

This file assumes the notebook already provides:
  - action_costs
  - build_all_priors
  - L2DDataset
  - Router
  - l2d_objective
  - combined_routing_loss
  - AugLag
  - evaluate

Usage:
    from adaptive_hpo_geometric_clip import run_hpo
    best_cfg, study = run_hpo(...)
"""

from __future__ import annotations
import pandas as pd 
import copy
import inspect
import json
import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import torch
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────
# §1  Geometric clip schedule
# ────────────────────────────────────────────────────────────────────

def default_rho_by_k(max_k: int = 32) -> Dict[int, float]:
    """Match the notebook's default truncated-geometric rank prior."""
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


def truncated_geometric_top_mass(k: int, rho: float, eps: float = 1e-8) -> float:
    """
    Top mass of the truncated geometric prior over k active experts:

        g_{k,1} = (1 - rho) / (1 - rho^k)

    This is the natural reference point for the maximum routed expert share.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    rho = float(np.clip(rho, eps, 1.0 - eps))
    denom = max(1.0 - rho**k, eps)
    return float((1.0 - rho) / denom)


def clip_max_from_geom(
    k: int,
    *,
    clip_ceiling: float,
    clip_slack: float,
    rho_by_k: Optional[Dict[int, float]] = None,
    clip_floor: float = 1e-6,
) -> float:
    """
    Geometric anti-collapse cap with a global ceiling:

        clip_max(k) = min(clip_ceiling, g_{k,1}(rho_k) + clip_slack)

    Notes
    -----
    - This does NOT force monotonicity across k.
    - Different support sizes may share the same ceiling, e.g. 0.35.
    - We still keep a tiny feasibility floor to avoid degenerate caps.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if rho_by_k is None:
        rho_by_k = default_rho_by_k(max_k=k)

    rho = float(rho_by_k.get(k, 0.72))
    geom_top = truncated_geometric_top_mass(k, rho)
    cap = min(float(clip_ceiling), geom_top + float(clip_slack))
    cap = max(cap, float(clip_floor))
    return float(cap)


def build_geometric_clip_anchors(
    anchor_ks: Sequence[int] = (5, 7, 12),
    *,
    clip_ceiling: float = 0.35,
    clip_slack: float = 0.03,
    rho_by_k: Optional[Dict[int, float]] = None,
    clip_floor: float = 1e-6,
) -> Dict[int, float]:
    """
    Materialize anchor caps for notebook compatibility.

    The underlying rule is:
        clip_max(k) = min(clip_ceiling, g_{k,1}(rho_k) + clip_slack)

    This preserves the user's intended logic:
    - a shared upper bound like 0.35 may apply to multiple support sizes,
    - support-size dependence is weak and contextual,
    - no monotone anchor post-processing is imposed here.
    """
    if not anchor_ks:
        return {}

    ks = sorted(int(k) for k in anchor_ks)
    if rho_by_k is None:
        rho_by_k = default_rho_by_k(max_k=max(ks))

    return {
        k: clip_max_from_geom(
            k,
            clip_ceiling=float(clip_ceiling),
            clip_slack=float(clip_slack),
            rho_by_k=rho_by_k,
            clip_floor=float(clip_floor),
        )
        for k in ks
    }


# ────────────────────────────────────────────────────────────────────
# §2  Search space
# ────────────────────────────────────────────────────────────────────

@dataclass
class SearchSpace:
    # --- TrainConfig ---
    lr: Tuple = ("float", 3e-5, 3e-4, True)
    gamma_tier: Tuple = ("float", 0.10, 1.20, True)

    # --- PriorRegConfig ---
    tau_bad: Tuple = ("float", 0.3, 1.2, False)
    w_gsdp: Tuple = ("float", 0.03, 1.0, True)
    w_rank_js: Tuple = ("float", 0.03, 1.0, True)

    global_uniform_mix: Tuple = ("float", 0.5, 0.0, False)
    family_uniform_mix: Tuple = ("float", 0.10, 0.55, False)
    group_uniform_mix: Tuple = ("float", 0.1, 0.55, False)

    family_n0: Tuple = ("float", 10.0, 40.0, False)
    group_n0: Tuple = ("float", 15.0, 60.0, False)
    global_mix: Tuple = ("float", 0.00, 0.08, False)

    # Geometric anti-collapse cap: clip_max(k) = min(clip_ceiling, g_{k,1}(rho_k) + clip_slack)
    clip_ceiling: Tuple = ("float", 0.32, 0.38, False)
    clip_slack: Tuple = ("float", 0.00, 0.06, False)

    # --- ALConfig ---
    al_mu: Tuple = ("float", 10.0, 35.0, False)
    al_lr_lambda: Tuple = ("float", 0.03, 0.20, True)


def sample_config(trial: optuna.Trial, ss: SearchSpace) -> Dict[str, Any]:
    """Sample a complete configuration from the HPO search space."""
    params: Dict[str, Any] = {}

    for name, spec in vars(ss).items():
        kind = spec[0]
        if kind == "float":
            low, high = spec[1], spec[2]
            log = spec[3] if len(spec) > 3 else False
            params[name] = trial.suggest_float(name, low, high, log=log)
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "cat":
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown search-space type for {name}: {kind}")

    return params


def params_to_configs(
    params: Dict[str, Any],
    *,
    train_costs: Tuple[float, float] = (2.0, 1.5),
    prior_costs: Optional[Tuple[float, float]] = None,
    batch_size: int = 64,
    warmup_epochs: int = 15,
    epochs: int = 150,
    patience: int = 18,
    min_delta: float = 1e-4,
    max_deferral_rate: float = 0.70,
    max_avg_cost: Optional[float] = None,
    clip_anchor_ks: Sequence[int] = (5, 7, 12),
    rho_by_k: Optional[Dict[int, float]] = None,
    clip_floor: float = 1e-6,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Convert flat Optuna params into train / prior-reg / AL config dicts.

    Clinical costs are fixed by study design by default and are not HPO knobs.
    """
    if prior_costs is None:
        prior_costs = train_costs

    clip_anchors = build_geometric_clip_anchors(
        anchor_ks=clip_anchor_ks,
        clip_ceiling=float(params["clip_ceiling"]),
        clip_slack=float(params["clip_slack"]),
        rho_by_k=rho_by_k,
        clip_floor=float(clip_floor),
    )

    train_cfg = {
        "lr": float(params["lr"]),
        "gamma_tier": float(params["gamma_tier"]),
        "c_fn": float(train_costs[0]),
        "c_fp": float(train_costs[1]),
        "batch_size": int(batch_size),
        "warmup_epochs": int(warmup_epochs),
        "epochs": int(epochs),
        "patience": int(patience),
        "min_delta": float(min_delta),
    }

    mcfg = {
        "tau_bad": float(params["tau_bad"]),
        "w_gsdp": float(params["w_gsdp"]),
        "w_rank_js": float(params["w_rank_js"]),
        "global_uniform_mix": float(params["global_uniform_mix"]),
        "family_uniform_mix": float(params["family_uniform_mix"]),
        "group_uniform_mix": float(params["group_uniform_mix"]),
        "family_n0": float(params["family_n0"]),
        "group_n0": float(params["group_n0"]),
        "global_mix": float(params["global_mix"]),
        "c_fn": float(prior_costs[0]),
        "c_fp": float(prior_costs[1]),
        "clip_max_anchors": clip_anchors,
        "clip_ceiling": float(params["clip_ceiling"]),
        "clip_slack": float(params["clip_slack"]),
        "clip_floor": float(clip_floor),
    }

    al_cfg = {
        "mu": float(params["al_mu"]),
        "lr_lambda": float(params["al_lr_lambda"]),
        "max_deferral_rate": float(max_deferral_rate),
        "max_avg_cost": max_avg_cost,
    }

    return train_cfg, mcfg, al_cfg


# ────────────────────────────────────────────────────────────────────
# §3  Multi-objective scalarization
# ────────────────────────────────────────────────────────────────────

@dataclass
class ObjectiveWeights:
    w_clinical: float = 0.35
    w_mcc: float = 0.15
    w_auprc: float = 0.10
    w_tier: float = 0.05
    w_constraint: float = 0.05
    rho: float = 0.05

    def scalarize(
        self,
        clinical: float,
        mcc: float,
        auprc: float,
        tier: float,
        constraint_viol: float,
        utopia: Optional[Dict[str, float]] = None,
    ) -> float:
        objectives = {
            "clinical": clinical,
            "mcc": -mcc,
            "auprc": -auprc,
            "tier": tier,
            "constraint": constraint_viol,
        }

        weights = {
            "clinical": self.w_clinical,
            "mcc": self.w_mcc,
            "auprc": self.w_auprc,
            "tier": self.w_tier,
            "constraint": self.w_constraint,
        }

        if utopia is None:
            utopia = {k: 0.0 for k in objectives}

        weighted_gaps: List[float] = []
        augmentation = 0.0
        for k in objectives:
            gap = objectives[k] - utopia.get(k, 0.0)
            wg = weights[k] * gap
            weighted_gaps.append(wg)
            augmentation += wg

        return float(max(weighted_gaps) + self.rho * augmentation)


class UtopiaTracker:
    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        self.best: Dict[str, float] = {}
        self.ema: Dict[str, float] = {}
        self.n_updates = 0

    def update(self, metrics: Dict[str, float]) -> Dict[str, float]:
        obj = {
            "clinical": metrics.get("clinical", 1.0),
            "mcc": -metrics.get("mcc", 0.0),
            "auprc": -metrics.get("auprc", 0.0),
            "tier": metrics.get("tier_soft", 0.5),
            "constraint": metrics.get("es_violation", 0.0),
        }

        for k, v in obj.items():
            v = float(v)
            if k not in self.best or v < self.best[k]:
                self.best[k] = v
            if k not in self.ema:
                self.ema[k] = v
            else:
                self.ema[k] = self.ema_alpha * v + (1.0 - self.ema_alpha) * self.ema[k]

        self.n_updates += 1
        if self.n_updates < 10:
            return dict(self.best)
        return {k: 0.5 * self.best[k] + 0.5 * self.ema[k] for k in self.best}


# ────────────────────────────────────────────────────────────────────
# §4  Internal notebook-compatibility helpers
# ────────────────────────────────────────────────────────────────────

class _DictNamespace:
    def __init__(self, d: Dict[str, Any]):
        self.__dict__.update(d)

    def __repr__(self) -> str:
        return f"Namespace({self.__dict__})"


def _make_prior_reg_config(d: Dict[str, Any]):
    defaults = {
        "alpha": 1.0,
        "beta": 1.0,
        "tier_cost": None,
        "capacity": None,
        "split_col": "split",
        "train_value": "train",
        "mask_col": "m_actions",
        "y_col": "y_true",
        "family_col": "support_family",
        "group_col": "group_id",
        "gsdp_divergence": "kl",
        "rank_js_margin": 0.0,
        "rank_js_mode": "any_excess",
        "rank_js_reduction": "mean",
    }
    return _DictNamespace({**defaults, **d})


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().numpy().tolist()
    return str(o)


def _construct_dataset(dataset_cls, df, expert_cols):
    try:
        return dataset_cls(df, expert_cols)
    except TypeError:
        return dataset_cls(df)


def _construct_router(router_cls, n_experts: int, hidden: int = 128):
    ctor_attempts = [
        lambda: router_cls(n_experts, hidden=hidden),
        lambda: router_cls(hidden=hidden),
        lambda: router_cls(n_experts),
        lambda: router_cls(),
    ]
    last_err = None
    for fn in ctor_attempts:
        try:
            return fn()
        except TypeError as e:
            last_err = e
    raise last_err if last_err is not None else RuntimeError("Router construction failed.")


def _make_optimizer(router: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    seen = set()
    param_groups: List[Dict[str, Any]] = []

    def add_module(name: str, weight_decay: float = 1e-4):
        if not hasattr(router, name):
            return
        module = getattr(router, name)
        if not hasattr(module, "parameters"):
            return
        params = []
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
        if params:
            param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})

    for name in ("risk_enc", "struct_enc", "ai_enc", "defer_head", "expert_head", "struct"):
        add_module(name, weight_decay=1e-4)

    if hasattr(router, "log_tau") and isinstance(getattr(router, "log_tau"), torch.nn.Parameter):
        p = getattr(router, "log_tau")
        if p.requires_grad and id(p) not in seen:
            param_groups.append({"params": [p], "lr": lr, "weight_decay": 0.0})
            seen.add(id(p))

    if not param_groups:
        params = [p for p in router.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("Router has no trainable parameters.")
        param_groups = [{"params": params, "lr": lr, "weight_decay": 1e-4}]

    return torch.optim.AdamW(param_groups)


def _router_forward(router, vim, q, u, vcdr, acdr, p_ai, logit_0, logit_1, action_mask):
    attempts = [
        lambda: router(vim, q, u, vcdr, acdr, p_ai, logit_0, logit_1, action_mask),
        lambda: router(vim, q, u, vcdr, acdr, logit_0, logit_1, p_ai, action_mask),
    ]
    last_err = None
    for fn in attempts:
        try:
            out = fn()
            if isinstance(out, tuple) and len(out) == 3:
                return out
            if isinstance(out, tuple) and len(out) == 2:
                pi, aux = out
                expert_logits = aux.get("expert_logits", None) if isinstance(aux, dict) else None
                return pi, expert_logits, aux
            raise RuntimeError("Unexpected router forward return signature.")
        except TypeError as e:
            last_err = e
    raise last_err if last_err is not None else RuntimeError("Router forward failed.")


def _to_numpy_1d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def _selection_score(val_results: Dict[str, float], violation_weight: float = 10.0) -> float:
    es_base = float(val_results.get("es_base", np.inf))
    es_violation = max(0.0, float(val_results.get("es_violation", 0.0)))
    return es_base + violation_weight * es_violation


# ────────────────────────────────────────────────────────────────────
# §5  Trial executor
# ────────────────────────────────────────────────────────────────────

class TrialExecutor:
    def __init__(
        self,
        df,
        expert_cols: List[str],
        experts: List[str],
        tier_cost_dict: Dict[str, float],
        n_experts: int,
        action_costs_fn,
        build_all_priors_fn,
        L2DDataset_cls,
        Router_cls,
        l2d_objective_fn,
        combined_routing_loss_fn,
        AugLag_cls,
        evaluate_fn,
        seed: int = 42,
        selection_violation_weight: float = 10.0,
    ):
        self.df = df
        self.expert_cols = expert_cols
        self.experts = experts
        self.tier_cost_dict = tier_cost_dict
        self.n_experts = n_experts
        self.action_costs_fn = action_costs_fn
        self.build_all_priors_fn = build_all_priors_fn
        self.L2DDataset_cls = L2DDataset_cls
        self.Router_cls = Router_cls
        self.l2d_objective_fn = l2d_objective_fn
        self.combined_routing_loss_fn = combined_routing_loss_fn
        self.AugLag_cls = AugLag_cls
        self.evaluate_fn = evaluate_fn
        self.seed = seed
        self.selection_violation_weight = selection_violation_weight

    def run_trial(
        self,
        trial: optuna.Trial,
        train_cfg_dict: Dict[str, Any],
        mcfg_dict: Dict[str, Any],
        al_cfg_dict: Dict[str, Any],
    ) -> Dict[str, float]:
        trial_seed = self.seed + trial.number
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_cfg = _DictNamespace(train_cfg_dict)
        train_cfg.device = str(device)
        mcfg = _make_prior_reg_config(mcfg_dict)
        al_cfg = _DictNamespace(al_cfg_dict)

        df = self.df.copy()
        EXPERT_COLS = [c for c in df.columns if c.startswith("y_") and c != "y_true"]

        df["m_experts"] = df.apply(lambda row: [int(not pd.isna(row[c])) for c in EXPERT_COLS], axis=1)
        df["m_actions"] = df["m_experts"].apply(lambda m: [1] + m)
        
        costs_np = self.action_costs_fn(self.tier_cost_dict, self.experts)
        tier_costs = torch.tensor(costs_np, dtype=torch.float32, device=device)

        priors = self.build_all_priors_fn(df, self.expert_cols, cfg=mcfg)
        group_prior_table = priors["final_group_table"].to(device)
        group_id_to_row = priors["final_id_to_row"]

        if torch.isnan(group_prior_table).any():
            raise optuna.TrialPruned("NaN in group_prior_table — prior construction failed")

        df_tr = df[df["split"].astype(str) == "train"].reset_index(drop=True)
        df_va = df[df["split"].astype(str) == "val"].reset_index(drop=True)
        if len(df_va) == 0:
            raise optuna.TrialPruned("No validation data")

        ds_tr = _construct_dataset(self.L2DDataset_cls, df_tr, self.expert_cols)
        ds_va = _construct_dataset(self.L2DDataset_cls, df_va, self.expert_cols)

        dl_tr = torch.utils.data.DataLoader(
            ds_tr,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(trial_seed),
        )
        dl_va = torch.utils.data.DataLoader(ds_va, batch_size=64, shuffle=False)

        router = _construct_router(self.Router_cls, self.n_experts, hidden=128).to(device)
        opt = _make_optimizer(router, lr=train_cfg.lr)
        al = self.AugLag_cls(al_cfg, device=device)

        best_score = float("inf")
        best_state = None
        best_epoch = 0
        patience_ctr = 0

        for ep in range(1, train_cfg.epochs + 1):
            router.train()
            n_epoch = 0
            defer_sum = 0.0
            tier_sum = 0.0

            for batch in dl_tr:
                (
                    vim, q, u, p_ai, vcdr, acdr, logit_0, logit_1,
                    y_true, y_exp, m_actions, expert_mask, ds_ids, group_ids,
                ) = batch

                vim, q, u = vim.to(device), q.to(device), u.to(device)
                vcdr, acdr = vcdr.to(device), acdr.to(device)
                p_ai = p_ai.to(device)
                logit_0, logit_1 = logit_0.to(device), logit_1.to(device)
                y_true, y_exp = y_true.to(device), y_exp.to(device)
                m_actions = m_actions.to(device)
                expert_mask = expert_mask.to(device)
                ds_ids, group_ids = ds_ids.to(device), group_ids.to(device)
                B = int(y_true.size(0))

                pi, expert_logits, aux = _router_forward(
                    router,
                    vim, q, u, vcdr, acdr,
                    p_ai, logit_0, logit_1, m_actions,
                )

                d = aux["d"].view(-1)
                q_route = aux["q"].view(B, self.n_experts)
                expert_mask_route = m_actions[:, 1:].to(q_route.dtype)

                exp_clin, exp_tier, defer_rate, pi_masked = self.l2d_objective_fn(
                    pi=pi,
                    action_mask=m_actions,
                    y_true=y_true,
                    p_ai=p_ai,
                    y_exp=y_exp,
                    tier_costs=tier_costs,
                    c_fn=train_cfg.c_fn,
                    c_fp=train_cfg.c_fp,
                )

                reg = self.combined_routing_loss_fn(
                    q=q_route,
                    expert_mask=expert_mask_route,
                    group_ids=group_ids,
                    group_prior_table=group_prior_table,
                    group_id_to_row=group_id_to_row,
                    cfg=mcfg,
                    d=d,
                )

                pen = al.penalty(defer_rate, exp_tier)
                loss = (
                    exp_clin
                    + train_cfg.gamma_tier * exp_tier
                    + pen
                    + mcfg.w_gsdp * reg["gsdp"]
                    + mcfg.w_rank_js * reg["rank_js"]
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=5.0)
                opt.step()

                # Early NaN detection — abort before wasting more epochs
                if torch.isnan(loss):
                    raise optuna.TrialPruned(
                        f"NaN loss at epoch {ep}, batch {n_epoch // B}. "
                        f"Likely unstable hyperparameters."
                    )

                defer_sum += float(defer_rate.detach().item()) * B
                tier_sum += float(exp_tier.detach().item()) * B
                n_epoch += B

            defer_mean = torch.tensor(defer_sum / max(n_epoch, 1), device=device)
            tier_mean = torch.tensor(tier_sum / max(n_epoch, 1), device=device)
            al.update(defer_mean, tier_mean)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val_results = self.evaluate_fn(
                    router,
                    dl_va,
                    tier_costs,
                    train_cfg,
                    mcfg,
                    group_prior_table,
                    group_id_to_row,
                    al_cfg,
                    device,
                    tag=f"T{trial.number}",
                )

            score = _selection_score(val_results, violation_weight=self.selection_violation_weight)
            trial.report(score, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if ep > train_cfg.warmup_epochs:
                if score < best_score - train_cfg.min_delta:
                    best_score = score
                    best_epoch = ep
                    patience_ctr = 0
                    best_state = copy.deepcopy(router.state_dict())
                else:
                    patience_ctr += 1
                    if patience_ctr >= train_cfg.patience:
                        break

        if best_state is not None:
            router.load_state_dict(best_state)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final = self.evaluate_fn(
                router,
                dl_va,
                tier_costs,
                train_cfg,
                mcfg,
                group_prior_table,
                group_id_to_row,
                al_cfg,
                device,
                tag=f"T{trial.number}-BEST",
            )

        # Guard against NaN metrics that would crash sklearn or scalarization
        for key in ("clinical", "mcc", "auprc", "acc"):
            val = final.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                raise optuna.TrialPruned(
                    f"NaN in final metric '{key}' — model diverged during training"
                )

        return {
            "clinical": float(final["clinical"]),
            "mcc": float(final["mcc"]),
            "auprc": float(final["auprc"]),
            "tier_soft": float(final["tier_soft"]),
            "defer_soft": float(final["defer_soft"]),
            "es_base": float(final["es_base"]),
            "es_violation": float(final["es_violation"]),
            "acc": float(final["acc"]),
            "best_epoch": int(best_epoch),
            **self._per_dataset_metrics(final, ds_va),
        }

    def _per_dataset_metrics(self, results, ds_va) -> Dict[str, float]:
        from sklearn.metrics import matthews_corrcoef as _mcc

        all_t = _to_numpy_1d(results["all_trues"])
        all_p = _to_numpy_1d(results["all_preds"])
        all_a = _to_numpy_1d(results["all_actions"])
        all_ds = _to_numpy_1d(results["all_ds"])

        inv_map = {v: k for k, v in getattr(ds_va, "ds_map", {}).items()}
        out: Dict[str, float] = {}

        for ds_idx in np.unique(all_ds).tolist():
            ds_idx_int = int(ds_idx)
            ds_name = inv_map.get(ds_idx_int, f"ds_{ds_idx_int}")
            mask = all_ds == ds_idx

            t = all_t[mask].astype(int)
            p = all_p[mask].astype(int)
            a = all_a[mask]
            n = int(mask.sum())
            if n == 0:
                continue

            tp = int(((t == 1) & (p == 1)).sum())
            fn = int(((t == 1) & (p == 0)).sum())
            fp = int(((t == 0) & (p == 1)).sum())
            tn = int(((t == 0) & (p == 0)).sum())

            acc = (tp + tn) / max(n, 1)
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            defer = float((a > 0).mean())
            mcc = float(_mcc(t, p)) if np.unique(t).size > 1 else 0.0

            prefix = str(ds_name).replace(" ", "_")
            out[f"{prefix}_acc"] = acc
            out[f"{prefix}_mcc"] = mcc
            out[f"{prefix}_sens"] = sens
            out[f"{prefix}_spec"] = spec
            out[f"{prefix}_defer"] = defer
            out[f"{prefix}_fn"] = float(fn)
            out[f"{prefix}_fp"] = float(fp)
            out[f"{prefix}_n"] = float(n)

        ds_accs = [v for k, v in out.items() if k.endswith("_acc")]
        out["acc_gap"] = float(max(ds_accs) - min(ds_accs)) if len(ds_accs) > 1 else 0.0
        return out


# ────────────────────────────────────────────────────────────────────
# §6  Optuna objective
# ────────────────────────────────────────────────────────────────────

def make_objective(
    executor: TrialExecutor,
    weights: ObjectiveWeights,
    utopia_tracker: UtopiaTracker,
    search_space: SearchSpace,
    ledger_path: Optional[str] = None,
    *,
    config_kwargs: Optional[Dict[str, Any]] = None,
):
    if config_kwargs is None:
        config_kwargs = {}

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = sample_config(trial, search_space)
        train_cfg, mcfg, al_cfg = params_to_configs(params, **config_kwargs)

        try:
            metrics = executor.run_trial(trial, train_cfg, mcfg, al_cfg)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned(f"Exception: {e}")

        utopia = utopia_tracker.update(metrics)
        score = weights.scalarize(
            clinical=metrics["clinical"],
            mcc=metrics["mcc"],
            auprc=metrics["auprc"],
            tier=metrics["tier_soft"],
            constraint_viol=metrics["es_violation"],
            utopia=utopia,
        )

        elapsed = time.time() - t0

        if ledger_path is not None:
            entry = {
                "trial": trial.number,
                "params": params,
                "train_cfg": train_cfg,
                "mcfg": mcfg,
                "al_cfg": al_cfg,
                "metrics": metrics,
                "scalar_score": score,
                "utopia": utopia,
                "elapsed_s": elapsed,
            }
            with open(ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=_json_default) + "\n")

        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("clip_max_anchors", mcfg.get("clip_max_anchors", {}))
        trial.set_user_attr("clip_ceiling", mcfg.get("clip_ceiling"))
        trial.set_user_attr("clip_slack", mcfg.get("clip_slack"))

        return float(score)

    return objective


# ────────────────────────────────────────────────────────────────────
# §7  Study construction & execution
# ────────────────────────────────────────────────────────────────────

def run_hpo(
    df,
    expert_cols: List[str],
    experts: List[str],
    tier_cost_dict: Dict[str, float],
    n_experts: int,
    action_costs_fn,
    build_all_priors_fn,
    L2DDataset_cls,
    Router_cls,
    l2d_objective_fn,
    combined_routing_loss_fn,
    AugLag_cls,
    evaluate_fn,
    *,
    n_trials: int = 80,
    n_jobs: int = 1,
    seed: int = 42,
    study_name: str = "mpd2_router_hpo",
    storage: Optional[str] = None,
    ledger_path: str = "hpo_ledger.jsonl",
    objective_weights: Optional[ObjectiveWeights] = None,
    train_costs: Tuple[float, float] = (2.0, 1.5),
    prior_costs: Optional[Tuple[float, float]] = None,
    batch_size: int = 64,
    warmup_epochs: int = 15,
    epochs: int = 150,
    patience: int = 18,
    min_delta: float = 1e-4,
    max_deferral_rate: float = 0.70,
    max_avg_cost: Optional[float] = None,
    clip_anchor_ks: Sequence[int] = (5, 7, 12),
    rho_by_k: Optional[Dict[int, float]] = None,
    clip_floor: float = 1e-6,
    selection_violation_weight: float = 10.0,
) -> Tuple[Dict[str, Any], optuna.Study]:
    if objective_weights is None:
        objective_weights = ObjectiveWeights()

    sampler = TPESampler(
        seed=seed,
        n_startup_trials=max(10, n_trials // 8),
        multivariate=True,
        group=True,
        constant_liar=True if n_jobs > 1 else False,
    )

    pruner = HyperbandPruner(
        min_resource=warmup_epochs,
        max_resource=epochs,
        reduction_factor=3,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    executor = TrialExecutor(
        df=df,
        expert_cols=expert_cols,
        experts=experts,
        tier_cost_dict=tier_cost_dict,
        n_experts=n_experts,
        action_costs_fn=action_costs_fn,
        build_all_priors_fn=build_all_priors_fn,
        L2DDataset_cls=L2DDataset_cls,
        Router_cls=Router_cls,
        l2d_objective_fn=l2d_objective_fn,
        combined_routing_loss_fn=combined_routing_loss_fn,
        AugLag_cls=AugLag_cls,
        evaluate_fn=evaluate_fn,
        seed=seed,
        selection_violation_weight=selection_violation_weight,
    )

    utopia_tracker = UtopiaTracker(ema_alpha=0.3)
    search_space = SearchSpace()

    config_kwargs = {
        "train_costs": train_costs,
        "prior_costs": prior_costs,
        "batch_size": batch_size,
        "warmup_epochs": warmup_epochs,
        "epochs": epochs,
        "patience": patience,
        "min_delta": min_delta,
        "max_deferral_rate": max_deferral_rate,
        "max_avg_cost": max_avg_cost,
        "clip_anchor_ks": clip_anchor_ks,
        "rho_by_k": rho_by_k,
        "clip_floor": clip_floor,
    }

    objective = make_objective(
        executor=executor,
        weights=objective_weights,
        utopia_tracker=utopia_tracker,
        search_space=search_space,
        ledger_path=ledger_path,
        config_kwargs=config_kwargs,
    )

    study.enqueue_trial({
        "lr": 1e-4,
        "gamma_tier": 1.0,
        "tau_bad": 1.0,
        "w_gsdp": 0.30,
        "w_rank_js": 0.30,
        "global_uniform_mix": 0.35,
        "family_uniform_mix": 0.30,
        "group_uniform_mix": 0.30,
        "family_n0": 25.0,
        "group_n0": 30.0,
        "global_mix": 0.05,
        "clip_ceiling": 0.35,
        "clip_slack": 0.03,
        "al_mu": 25.0,
        "al_lr_lambda": 0.10,
    })

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_metrics = {k: best_trial.user_attrs[k] for k in best_trial.user_attrs}

    logger.info(
        f"Best trial #{best_trial.number}: score={best_trial.value:.4f}\n"
        f"  Metrics: {json.dumps(best_metrics, indent=2, default=_json_default)}\n"
        f"  Params:  {json.dumps(best_params, indent=2, default=_json_default)}"
    )

    return best_params, study


# ────────────────────────────────────────────────────────────────────
# §8  Post-hoc analysis utilities
# ────────────────────────────────────────────────────────────────────

def pareto_front(study: optuna.Study, obj_keys: Optional[List[str]] = None) -> List[int]:
    if obj_keys is None:
        obj_keys = ["clinical", "mcc", "auprc", "tier_soft"]

    negate = {"mcc", "auprc", "acc"}
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return []

    points = []
    for t in trials:
        vec = []
        for k in obj_keys:
            v = t.user_attrs.get(k, float("inf"))
            if k in negate:
                v = -v
            vec.append(v)
        points.append(vec)

    points = np.array(points)
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_pareto[i] = False
                break

    return [trials[i].number for i in range(n) if is_pareto[i]]


def importance_analysis(study: optuna.Study) -> Dict[str, float]:
    try:
        from optuna.importance import get_param_importances
        return get_param_importances(study)
    except Exception as e:
        logger.warning(f"Importance analysis failed: {e}")
        return {}


def retrain_with_best(
    best_params: Dict[str, Any],
    df,
    expert_cols,
    train_fn,
    *,
    extra_epochs: int = 30,
    train_costs: Tuple[float, float] = (2.0, 1.5),
    prior_costs: Optional[Tuple[float, float]] = None,
    batch_size: int = 64,
    warmup_epochs: int = 15,
    epochs: int = 150,
    patience: int = 18,
    min_delta: float = 1e-4,
    max_deferral_rate: float = 0.70,
    max_avg_cost: Optional[float] = None,
    clip_anchor_ks: Sequence[int] = (5, 7, 12),
    rho_by_k: Optional[Dict[int, float]] = None,
    clip_floor: float = 1e-6,
):
    train_cfg_d, mcfg_d, al_cfg_d = params_to_configs(
        best_params,
        train_costs=train_costs,
        prior_costs=prior_costs,
        batch_size=batch_size,
        warmup_epochs=warmup_epochs,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        max_deferral_rate=max_deferral_rate,
        max_avg_cost=max_avg_cost,
        clip_anchor_ks=clip_anchor_ks,
        rho_by_k=rho_by_k,
        clip_floor=clip_floor,
    )
    train_cfg_d["epochs"] = int(train_cfg_d.get("epochs", epochs) + extra_epochs)
    train_cfg_d["patience"] = max(int(train_cfg_d.get("patience", patience)), 25)
    return train_cfg_d, mcfg_d, al_cfg_d


def print_study_summary(study: optuna.Study, top_k: int = 10):
    trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
    )

    print(f"\n{'='*80}")
    print(f"HPO Study Summary: {study.study_name}")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed:    {len(trials)}")
    print(f"  Best value:   {study.best_value:.4f} (trial #{study.best_trial.number})")
    print(f"{'='*80}")

    print(f"\nTop-{top_k} Trials:")
    print(
        f"  {'#':>4} {'Score':>8} {'Clin':>7} {'MCC':>7} {'AUPRC':>7} "
        f"{'Tier':>7} {'Defer':>7} {'Gap':>7}"
    )
    print(f"  {'-'*68}")

    for t in trials[:top_k]:
        ua = t.user_attrs
        print(
            f"  {t.number:>4} {t.value:>8.4f} "
            f"{ua.get('clinical', 0):>7.4f} "
            f"{ua.get('mcc', 0):>7.4f} "
            f"{ua.get('auprc', 0):>7.4f} "
            f"{ua.get('tier_soft', 0):>7.4f} "
            f"{ua.get('defer_soft', 0):>7.4f} "
            f"{ua.get('acc_gap', 0):>7.4f}"
        )

    imp = importance_analysis(study)
    if imp:
        print("\nParameter Importance (fANOVA):")
        for k, v in sorted(imp.items(), key=lambda x: -x[1])[:10]:
            bar = "█" * int(v * 40)
            print(f"  {k:<25} {v:>6.3f} {bar}")

    pf = pareto_front(study)
    if pf:
        print(f"\nPareto-optimal trials: {pf}")

    print(f"{'='*80}\n")


# ────────────────────────────────────────────────────────────────────
# §9  Convenience entry point
# ────────────────────────────────────────────────────────────────────

def run_hpo_from_notebook(
    df,
    tier_cost_dict,
    action_costs_fn,
    build_all_priors_fn,
    L2DDataset_cls,
    Router_cls,
    l2d_objective_fn,
    combined_routing_loss_fn,
    AugLag_cls,
    evaluate_fn,
    n_trials: int = 80,
    seed: int = 42,
    **kwargs,
):
    EXPERT_COLS = [c for c in df.columns if c.startswith("y_") and c != "y_true"]
    EXPERTS  = [c.removeprefix("y_") for c in EXPERT_COLS]
    N_EXPERTS = len(EXPERT_COLS)
    df["m_experts"] = df.apply(lambda row: [int(not pd.isna(row[c])) for c in EXPERT_COLS], axis=1)
    df["m_actions"] = df["m_experts"].apply(lambda m: [1] + m)
    
    best_params, study = run_hpo(
        df=df,
        expert_cols= EXPERT_COLS,
        experts= EXPERTS,
        tier_cost_dict=tier_cost_dict,
        n_experts=N_EXPERTS,
        action_costs_fn=action_costs_fn,
        build_all_priors_fn=build_all_priors_fn,
        L2DDataset_cls=L2DDataset_cls,
        Router_cls=Router_cls,
        l2d_objective_fn=l2d_objective_fn,
        combined_routing_loss_fn=combined_routing_loss_fn,
        AugLag_cls=AugLag_cls,
        evaluate_fn=evaluate_fn,
        n_trials=n_trials,
        seed=seed,
        **kwargs,
    )
    print_study_summary(study)
    return best_params, study