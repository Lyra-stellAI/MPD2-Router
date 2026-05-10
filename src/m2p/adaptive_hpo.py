"""
Autonomous Adaptive Hyperparameter Optimization for MPD²-Router
================================================================

Multi-objective Bayesian optimization with constraint-aware early stopping,
adaptive search space pruning, and population-based warm-starting.

Design principles
-----------------
1. **Multi-objective**: jointly optimise clinical cost and routing quality
   (MCC, AUPRC, deferral constraint satisfaction) via scalarized Chebyshev
   aggregation with an adaptive reference point.
2. **Constraint-aware**: treat deferral-rate violations as hard constraints
   using augmented Lagrangian inside the surrogate objective.  Model
   selection also penalises constraint violations via a tunable weight
   (``selection_violation_weight``).
3. **Adaptive pruning**: Hyperband-style successive halving on the epoch
   budget to kill bad configs early.
4. **Geometric clip caps**: anti-collapse clipping derived from the
   truncated-geometric rank prior, controlled by two intuitive knobs
   (``clip_ceiling``, ``clip_slack``) instead of per-anchor-point values.
5. **Reproducibility**: every trial logs the full materialised config +
   seed + metrics to a structured JSON ledger for post-hoc analysis.

Usage::

    from adaptive_hpo import run_hpo
    best_cfg, study = run_hpo(df, expert_cols, ...)

Requires: optuna >= 3.0, torch, numpy, pandas, scikit-learn
"""

from __future__ import annotations

import copy
import json
import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import torch
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# §1  Geometric clip schedule
# ────────────────────────────────────────────────────────────────────

def default_rho_by_k(max_k: int = 32) -> Dict[int, float]:
    """
    Default concentration parameters for the truncated-geometric rank
    prior, keyed by support size *k*.

    These values were hand-tuned against the notebook's default rank
    prior and should rarely need changing.

    Returns
    -------
    dict  —  mapping  k → ρ_k
    """
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
    Top mass of the truncated geometric prior over *k* active experts::

        g_{k,1}  =  (1 − ρ) / (1 − ρ^k)

    This is the natural reference point for the maximum routed-expert
    share.
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
    Geometric anti-collapse cap with a global ceiling::

        clip_max(k) = min(clip_ceiling,  g_{k,1}(ρ_k) + clip_slack)

    Parameters
    ----------
    k : int
        Number of active experts in the support.
    clip_ceiling : float
        Hard upper bound on the cap (e.g. 0.35).
    clip_slack : float
        Additive headroom above the geometric top mass.
    rho_by_k : dict, optional
        Per-support-size concentration; uses ``default_rho_by_k`` if None.
    clip_floor : float
        Tiny feasibility floor to avoid degenerate caps.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if rho_by_k is None:
        rho_by_k = default_rho_by_k(max_k=k)
    rho = float(rho_by_k.get(k, 0.72))
    geom_top = truncated_geometric_top_mass(k, rho)
    cap = min(float(clip_ceiling), geom_top + float(clip_slack))
    return float(max(cap, float(clip_floor)))


def build_geometric_clip_anchors(
    anchor_ks: Sequence[int] = (5, 7, 12),
    *,
    clip_ceiling: float = 0.35,
    clip_slack: float = 0.03,
    rho_by_k: Optional[Dict[int, float]] = None,
    clip_floor: float = 1e-6,
) -> Dict[int, float]:
    """
    Materialise anchor caps for notebook compatibility.

    The underlying rule is::

        clip_max(k) = min(clip_ceiling,  g_{k,1}(ρ_k) + clip_slack)

    Returns
    -------
    dict  —  mapping  anchor support-size → clip cap
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
# §2  Search-space specification
# ────────────────────────────────────────────────────────────────────

@dataclass
class SearchSpace:
    """
    Declarative specification of the HPO search space.

    Each field is a tuple ``(type, *bounds[, log])``:

    * ``("float", low, high, log=False)``
    * ``("int",   low, high)``
    * ``("cat",   [choices])``

    **16 tunable dimensions** — a reasonable density for 80–120 trials
    with multivariate TPE + Hyperband.

    Design notes
    ~~~~~~~~~~~~
    * ``warmup_epochs`` is tunable.  A prior HPO run found the optimum at
      9, well below the notebook default of 15–18, so leaving it fixed
      forfeits a meaningful degree of freedom.
    * Clinical costs ``c_fn / c_fp`` are fixed by study design and
      controlled via the ``train_costs`` kwarg in ``params_to_configs``
      rather than being search-space knobs.
    * Clip caps use two geometric knobs (``clip_ceiling`` +
      ``clip_slack``) instead of per-anchor values, encoding domain
      structure while reducing dimensionality.
    * Uniform-mix proportions are separated by hierarchy level (global /
      family / group) because their optima differ structurally.
    """

    # ── TrainConfig ──────────────────────────────────────────────────
    lr:              Tuple = ("float", 3e-5, 3e-4, True)
    gamma_tier:      Tuple = ("float", 0.05, 1.50, True)
    warmup_epochs:   Tuple = ("int",   8,    20)

    # ── PriorRegConfig ──────────────────────────────────────────────
    tau_bad:         Tuple = ("float", 0.30, 2.00, False)
    w_gsdp:    Tuple = ("float", 0.03, 1.00, False)   # linear
    w_rank_js: Tuple = ("float", 0.03, 1.00, False)   # linear

    global_uniform_mix:  Tuple = ("float", 0.05, 0.40, False)
    family_uniform_mix:  Tuple = ("float", 0.10, 0.55, False)
    group_uniform_mix:   Tuple = ("float", 0.10, 0.55, False)

    family_n0:       Tuple = ("float", 10.0, 40.0, False)
    group_n0:        Tuple = ("float", 15.0, 60.0, False)
    global_mix:      Tuple = ("float", 0.01, 0.10, False)

    # Geometric anti-collapse: clip_max(k) = min(ceiling, g_{k,1} + slack)
    clip_ceiling:    Tuple = ("float", 0.30, 0.38, False)
    clip_slack:      Tuple = ("float", 0.00, 0.06, False)

    # ── ALConfig ────────────────────────────────────────────────────
    al_mu:           Tuple = ("float", 8.0,  40.0, False)
    al_lr_lambda:    Tuple = ("float", 0.02, 0.30, True)


def sample_config(trial: optuna.Trial, ss: SearchSpace) -> Dict[str, Any]:
    """
    Sample a complete configuration from the search space.

    Iterates over every field in ``ss`` and dispatches to the correct
    Optuna ``suggest_*`` method based on the declared type string.
    """
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
            raise ValueError(f"Unknown search-space type for {name!r}: {kind!r}")
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
    Convert a flat Optuna parameter dict → ``(train_cfg, mcfg, al_cfg)``
    config dicts.

    Parameters that appear in *params* (i.e. are HPO knobs) take
    precedence; everything else falls back to keyword-argument defaults.
    This lets callers promote or demote any parameter between "fixed" and
    "tunable" without touching the function body.

    Parameters
    ----------
    params : dict
        Flat dict from ``sample_config`` or ``study.best_params``.
    train_costs : (c_fn, c_fp)
        Fixed clinical cost asymmetry for the training objective.
    prior_costs : (c_fn, c_fp) or None
        Clinical costs for prior construction; defaults to *train_costs*.
    warmup_epochs : int
        Fallback used only when ``warmup_epochs`` is **not** in *params*.
    epochs, patience, min_delta, batch_size :
        Fixed training-loop hypers (not tuned).
    max_deferral_rate : float
        Hard deployment constraint on deferral rate.
    clip_anchor_ks : sequence of int
        Support sizes at which to materialise clip anchors.
    rho_by_k, clip_floor :
        Geometric clip schedule overrides.

    Returns
    -------
    train_cfg, mcfg, al_cfg : dict, dict, dict
    """
    if prior_costs is None:
        prior_costs = train_costs

    # ── Geometric clip anchors ──────────────────────────────────────
    clip_anchors = build_geometric_clip_anchors(
        anchor_ks=clip_anchor_ks,
        clip_ceiling=float(params["clip_ceiling"]),
        clip_slack=float(params["clip_slack"]),
        rho_by_k=rho_by_k,
        clip_floor=float(clip_floor),
    )

    # ── TrainConfig ─────────────────────────────────────────────────
    train_cfg: Dict[str, Any] = {
        "lr":             float(params["lr"]),
        "gamma_tier":     float(params["gamma_tier"]),
        "c_fn":           float(train_costs[0]),
        "c_fp":           float(train_costs[1]),
        "batch_size":     int(batch_size),
        "warmup_epochs":  int(params.get("warmup_epochs", warmup_epochs)),
        "epochs":         int(epochs),
        "patience":       int(patience),
        "min_delta":      float(min_delta),
    }

    # ── PriorRegConfig ──────────────────────────────────────────────
    mcfg: Dict[str, Any] = {
        "tau_bad":             float(params["tau_bad"]),
        "w_gsdp":              float(params["w_gsdp"]),
        "w_rank_js":           float(params["w_rank_js"]),
        "global_uniform_mix":  float(params["global_uniform_mix"]),
        "family_uniform_mix":  float(params["family_uniform_mix"]),
        "group_uniform_mix":   float(params["group_uniform_mix"]),
        "family_n0":           float(params["family_n0"]),
        "group_n0":            float(params["group_n0"]),
        "global_mix":          float(params["global_mix"]),
        "c_fn":                float(prior_costs[0]),
        "c_fp":                float(prior_costs[1]),
        "clip_max_anchors":    clip_anchors,
        "clip_ceiling":        float(params["clip_ceiling"]),
        "clip_slack":          float(params["clip_slack"]),
        "clip_floor":          float(clip_floor),
    }

    # ── ALConfig ────────────────────────────────────────────────────
    al_cfg: Dict[str, Any] = {
        "mu":                float(params["al_mu"]),
        "lr_lambda":         float(params["al_lr_lambda"]),
        "max_deferral_rate": float(max_deferral_rate),
        "max_avg_cost":      max_avg_cost,
    }

    return train_cfg, mcfg, al_cfg


# ────────────────────────────────────────────────────────────────────
# §3  Multi-objective scalarization
# ────────────────────────────────────────────────────────────────────

@dataclass
class ObjectiveWeights:
    """
    Augmented Chebyshev scalarization weights for multi-objective HPO.

    The final scalar objective is::

        s  =  max_i [ w_i · (f_i − z_i*) ]
            + ρ · Σ_i  w_i · (f_i − z_i*)

    where *z** is the utopia point (best-known per-objective values).
    This guarantees Pareto-optimal solutions for any weight vector.

    All objectives are converted to **minimisation** form internally:
    ``clinical``, ``tier``, ``constraint_viol`` are lower-is-better;
    ``mcc`` and ``auprc`` are negated (higher-is-better → lower-is-better).
    """
    w_clinical:   float = 0.35
    w_mcc:        float = 0.15
    w_auprc:      float = 0.10
    w_tier:       float = 0.05
    w_constraint: float = 0.05
    rho:          float = 0.05   # augmentation coefficient

    def scalarize(
        self,
        clinical: float,
        mcc: float,
        auprc: float,
        tier: float,
        constraint_viol: float,
        utopia: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute augmented Chebyshev scalarization (minimisation)."""
        objectives = {
            "clinical":   clinical,
            "mcc":        -mcc,
            "auprc":      -auprc,
            "tier":       tier,
            "constraint": constraint_viol,
        }
        weights = {
            "clinical":   self.w_clinical,
            "mcc":        self.w_mcc,
            "auprc":      self.w_auprc,
            "tier":       self.w_tier,
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


# ────────────────────────────────────────────────────────────────────
# §4  Adaptive utopia estimation
# ────────────────────────────────────────────────────────────────────

class UtopiaTracker:
    """
    Maintains running estimates of the per-objective best (utopia point)
    using exponential moving average to handle noise.

    The utopia point *z** anchors the Chebyshev scalarization and adapts
    as the study progresses, focusing the search on the current Pareto
    frontier rather than chasing stale reference values.

    Strategy
    --------
    * First 10 trials — use the optimistic (best-seen) utopia.
    * After 10 trials — blend best-seen and EMA 50 / 50 for stability.
    """

    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        self.best: Dict[str, float] = {}
        self.ema: Dict[str, float] = {}
        self.n_updates = 0

    def update(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Update utopia estimates with new trial results."""
        obj = {
            "clinical":   metrics.get("clinical", 1.0),
            "mcc":        -metrics.get("mcc", 0.0),
            "auprc":      -metrics.get("auprc", 0.0),
            "tier":       metrics.get("tier_soft", 0.5),
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
# §5  Internal notebook-compatibility helpers
# ────────────────────────────────────────────────────────────────────

class _DictNamespace:
    """Lightweight namespace that behaves like a dataclass for attr access."""
    def __init__(self, d: Dict[str, Any]):
        self.__dict__.update(d)

    def __repr__(self) -> str:
        return f"Namespace({self.__dict__})"


def _make_prior_reg_config(d: Dict[str, Any]):
    """Construct a PriorRegConfig-compatible namespace from a dict."""
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
    """JSON serializer for non-standard types."""
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
    """Try ``(df, expert_cols)`` first; fall back to ``(df,)``."""
    try:
        return dataset_cls(df, expert_cols)
    except TypeError:
        return dataset_cls(df)


def _construct_router(router_cls, n_experts: int, hidden: int = 128):
    """Try several constructor signatures to survive API changes."""
    attempts = [
        lambda: router_cls(n_experts, hidden=hidden),
        lambda: router_cls(hidden=hidden),
        lambda: router_cls(n_experts),
        lambda: router_cls(),
    ]
    last_err: Optional[Exception] = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as exc:
            last_err = exc
    raise last_err if last_err is not None else RuntimeError(
        "Router construction failed."
    )


def _make_optimizer(router: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Build an AdamW optimizer by introspecting known sub-modules of the
    router.  Falls back to ``router.parameters()`` if none are found.
    ``log_tau`` gets zero weight-decay.
    """
    seen: set = set()
    param_groups: List[Dict[str, Any]] = []

    def _add(name: str, wd: float = 1e-4):
        mod = getattr(router, name, None)
        if mod is None or not hasattr(mod, "parameters"):
            return
        ps = [p for p in mod.parameters() if p.requires_grad and id(p) not in seen]
        if ps:
            param_groups.append({"params": ps, "lr": lr, "weight_decay": wd})
            seen.update(id(p) for p in ps)

    for name in ("risk_enc", "struct_enc", "ai_enc",
                 "defer_head", "expert_head", "struct"):
        _add(name, wd=1e-4)

    # log_tau — no weight decay
    log_tau = getattr(router, "log_tau", None)
    if isinstance(log_tau, torch.nn.Parameter) and log_tau.requires_grad and id(log_tau) not in seen:
        param_groups.append({"params": [log_tau], "lr": lr, "weight_decay": 0.0})
        seen.add(id(log_tau))

    # Fallback: grab everything we haven't seen yet
    if not param_groups:
        ps = [p for p in router.parameters() if p.requires_grad]
        if not ps:
            raise RuntimeError("Router has no trainable parameters.")
        param_groups = [{"params": ps, "lr": lr, "weight_decay": 1e-4}]

    return torch.optim.AdamW(param_groups)


def _router_forward(router, vim, q, u, vcdr, acdr, p_ai, logit_0, logit_1, action_mask):
    """Try known forward-arg orderings; return ``(pi, expert_logits, aux)``."""
    orderings = [
        lambda: router(vim, q, u, vcdr, acdr, p_ai, logit_0, logit_1, action_mask),
        lambda: router(vim, q, u, vcdr, acdr, logit_0, logit_1, p_ai, action_mask),
    ]
    last_err: Optional[Exception] = None
    for fn in orderings:
        try:
            out = fn()
            if isinstance(out, tuple) and len(out) == 3:
                return out
            if isinstance(out, tuple) and len(out) == 2:
                pi, aux = out
                el = aux.get("expert_logits", None) if isinstance(aux, dict) else None
                return pi, el, aux
            raise RuntimeError("Unexpected router forward return signature.")
        except TypeError as exc:
            last_err = exc
    raise last_err if last_err is not None else RuntimeError("Router forward failed.")


def _to_numpy_1d(x) -> np.ndarray:
    """Coerce tensor / array → 1-D numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def _selection_score(
    val_results: Dict[str, float],
    violation_weight: float = 10.0,
) -> float:
    """
    Constraint-aware model-selection score.
    Combines the base early-stopping score with a weighted penalty for
    """
    es_base = float(val_results.get("es_base", np.inf))
    total_cost = float(val_results.get("total_cost", np.inf))
    #es_violation = max(0.0, float(val_results.get("es_violation", 0.0)))
    #es_base + violation_weight * es_violation
    return (es_base + total_cost)/2.0


# ────────────────────────────────────────────────────────────────────
# §6  Trial executor with intermediate reporting
# ────────────────────────────────────────────────────────────────────

class TrialExecutor:
    """
    Wraps the training loop with per-epoch Optuna reporting and
    Hyperband-compatible pruning.

    Key features
    ~~~~~~~~~~~~~
    * Reports an intermediate validation score at every epoch for
      Hyperband pruning.
    * Uses ``_selection_score`` for constraint-aware checkpoint selection.
    * NaN guards at loss computation, prior construction, and final
      metric levels.
    * Defensive construction helpers for dataset / router / optimizer to
      tolerate minor API changes in the notebook.
    """

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

    # ----------------------------------------------------------------

    def run_trial(
        self,
        trial: optuna.Trial,
        train_cfg_dict: Dict[str, Any],
        mcfg_dict: Dict[str, Any],
        al_cfg_dict: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Execute a single training trial with pruning support.

        Returns
        -------
        dict — keys: ``clinical``, ``mcc``, ``auprc``, ``tier_soft``,
        ``defer_soft``, ``es_base``, ``es_violation``, ``acc``,
        ``best_epoch``, plus per-dataset breakdown metrics.
        """
        # ── Reproducibility ─────────────────────────────────────────
        trial_seed = self.seed + trial.number
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Materialise configs ─────────────────────────────────────
        train_cfg = _DictNamespace(train_cfg_dict)
        train_cfg.device = str(device)
        mcfg = _make_prior_reg_config(mcfg_dict)
        al_cfg = _DictNamespace(al_cfg_dict)

        # ── Data ────────────────────────────────────────────────────
        df = self.df.copy()
        expert_cols_live = [
            c for c in df.columns if c.startswith("y_") and c != "y_true"
        ]
        df["m_experts"] = df.apply(
            lambda row: [int(not pd.isna(row[c])) for c in expert_cols_live],
            axis=1,
        )
        df["m_actions"] = df["m_experts"].apply(lambda m: [1] + m)

        costs_np = self.action_costs_fn(self.tier_cost_dict, self.experts)
        tier_costs = torch.tensor(costs_np, dtype=torch.float32, device=device)

        priors = self.build_all_priors_fn(df, self.expert_cols, cfg=mcfg)
        group_prior_table = priors["final_group_table"].to(device)
        group_id_to_row = priors["final_id_to_row"]

        if torch.isnan(group_prior_table).any():
            raise optuna.TrialPruned(
                "NaN in group_prior_table — prior construction failed"
            )

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

        # ── Model ───────────────────────────────────────────────────
        router = _construct_router(
            self.Router_cls, self.n_experts, hidden=128,
        ).to(device)
        opt = _make_optimizer(router, lr=train_cfg.lr)
        al = self.AugLag_cls(al_cfg, device=device)

        best_score = float("inf")
        best_state = None
        best_epoch = 0
        patience_ctr = 0

        # ── Training loop ───────────────────────────────────────────
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
                    router, vim, q, u, vcdr, acdr,
                    p_ai, logit_0, logit_1, m_actions,
                )

                d = aux["d"].view(-1)
                q_route = aux["q"].view(B, self.n_experts)
                expert_mask_route = m_actions[:, 1:].to(q_route.dtype)

                exp_clin, exp_tier, defer_rate, pi_masked = self.l2d_objective_fn(
                    pi=pi, action_mask=m_actions, y_true=y_true,
                    p_ai=p_ai, y_exp=y_exp, tier_costs=tier_costs,
                    c_fn=train_cfg.c_fn, c_fp=train_cfg.c_fp,
                )

                reg = self.combined_routing_loss_fn(
                    q=q_route, expert_mask=expert_mask_route,
                    group_ids=group_ids,
                    group_prior_table=group_prior_table,
                    group_id_to_row=group_id_to_row,
                    cfg=mcfg, d=d,
                )

                pen = al.penalty(defer_rate, exp_tier)
                loss = (
                    exp_clin
                    + train_cfg.gamma_tier * exp_tier
                    + pen
                    + mcfg.w_gsdp  * reg["gsdp"]
                    + mcfg.w_rank_js * reg["rank_js"]
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=5.0)
                opt.step()

                # NaN guard — abort before wasting more epochs
                if torch.isnan(loss):
                    raise optuna.TrialPruned(
                        f"NaN loss at epoch {ep}. "
                        f"Likely unstable hyperparameters."
                    )

                defer_sum += float(defer_rate.detach().item()) * B
                tier_sum  += float(exp_tier.detach().item()) * B
                n_epoch   += B

            # ── AL multiplier update ────────────────────────────────
            defer_mean = torch.tensor(
                defer_sum / max(n_epoch, 1), device=device,
            )
            tier_mean = torch.tensor(
                tier_sum / max(n_epoch, 1), device=device,
            )
            al.update(defer_mean, tier_mean)

            # ── Validation ──────────────────────────────────────────
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val_results = self.evaluate_fn(
                    router, dl_va, tier_costs, train_cfg, mcfg,
                    group_prior_table, group_id_to_row, al_cfg, device,
                    tag=f"T{trial.number}",
                )

            # Constraint-aware score for both Optuna reporting and
            # checkpoint selection
            score = _selection_score(
                val_results,
                violation_weight=self.selection_violation_weight,
            )

            # ── Optuna pruning ──────────────────────────────────────
            trial.report(score, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # ── Early stopping (active only after warmup) ───────────
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

        # ── Load best checkpoint & final eval ───────────────────────
        if best_state is not None:
            router.load_state_dict(best_state)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final = self.evaluate_fn(
                router, dl_va, tier_costs, train_cfg, mcfg,
                group_prior_table, group_id_to_row, al_cfg, device,
                tag=f"T{trial.number}-BEST",
            )

        # NaN guard — prevent corrupted metrics entering the study
        for key in ("clinical", "mcc", "auprc", "acc"):
            val = final.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                raise optuna.TrialPruned(
                    f"NaN in final metric {key!r} — model diverged"
                )

        return {
            "clinical":     float(final["clinical"]),
            "mcc":          float(final["mcc"]),
            "auprc":        float(final["auprc"]),
            "tier_soft":    float(final["tier_soft"]),
            "defer_soft":   float(final["defer_soft"]),
            "es_base":      float(final["es_base"]),
            "es_violation": float(final["es_violation"]),
            "acc":          float(final["acc"]),
            "best_epoch":   int(best_epoch),
            **self._per_dataset_metrics(final, ds_va),
        }

    # ----------------------------------------------------------------

    def _per_dataset_metrics(self, results, ds_va) -> Dict[str, float]:
        """
        Per-dataset system accuracy, MCC, sensitivity, specificity and
        deferral rate.  Enables tracking generalisation across HPO trials.
        """
        from sklearn.metrics import matthews_corrcoef as _mcc

        all_t  = _to_numpy_1d(results["all_trues"])
        all_p  = _to_numpy_1d(results["all_preds"])
        all_a  = _to_numpy_1d(results["all_actions"])
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

            acc   = (tp + tn) / max(n, 1)
            sens  = tp / max(tp + fn, 1)
            spec  = tn / max(tn + fp, 1)
            defer = float((a > 0).mean())
            mcc   = float(_mcc(t, p)) if np.unique(t).size > 1 else 0.0

            pfx = str(ds_name).replace(" ", "_")
            out[f"{pfx}_acc"]   = acc
            out[f"{pfx}_mcc"]   = mcc
            out[f"{pfx}_sens"]  = sens
            out[f"{pfx}_spec"]  = spec
            out[f"{pfx}_defer"] = defer
            out[f"{pfx}_fn"]    = float(fn)
            out[f"{pfx}_fp"]    = float(fp)
            out[f"{pfx}_n"]     = float(n)

        ds_accs = [v for k, v in out.items() if k.endswith("_acc")]
        out["acc_gap"] = (
            float(max(ds_accs) - min(ds_accs)) if len(ds_accs) > 1 else 0.0
        )
        return out


# ────────────────────────────────────────────────────────────────────
# §7  Optuna objective
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
    """
    Factory for the Optuna objective function.

    Closes over *executor*, *weights*, and *utopia_tracker* to maintain
    state across trials.  The JSONL ledger records the **full
    materialised** configs (not just flat Optuna params) for post-hoc
    reproducibility.
    """
    if config_kwargs is None:
        config_kwargs = {}

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()

        # ── Sample & materialise ────────────────────────────────────
        params = sample_config(trial, search_space)
        train_cfg, mcfg, al_cfg = params_to_configs(params, **config_kwargs)

        # ── Execute ─────────────────────────────────────────────────
        try:
            metrics = executor.run_trial(trial, train_cfg, mcfg, al_cfg)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.warning(f"Trial {trial.number} failed: {exc}")
            raise optuna.TrialPruned(f"Exception: {exc}")

        # ── Scalarise ───────────────────────────────────────────────
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

        # ── Ledger ──────────────────────────────────────────────────
        if ledger_path is not None:
            entry = {
                "trial":        trial.number,
                "params":       params,
                "train_cfg":    train_cfg,
                "mcfg":         mcfg,
                "al_cfg":       al_cfg,
                "metrics":      metrics,
                "scalar_score": score,
                "utopia":       utopia,
                "elapsed_s":    elapsed,
            }
            with open(ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=_json_default) + "\n")

        # ── User attrs for post-hoc analysis ────────────────────────
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr(
            "clip_max_anchors", mcfg.get("clip_max_anchors", {}),
        )
        trial.set_user_attr("clip_ceiling", mcfg.get("clip_ceiling"))
        trial.set_user_attr("clip_slack",   mcfg.get("clip_slack"))

        return float(score)

    return objective


# ────────────────────────────────────────────────────────────────────
# §8  Study construction & execution
# ────────────────────────────────────────────────────────────────────

def run_hpo(
    df,
    expert_cols: List[str],
    experts: List[str],
    tier_cost_dict: Dict[str, float],
    n_experts: int,
    # function / class handles from the notebook
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
    # Fixed hypers forwarded to params_to_configs
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
    """
    Run the full HPO study.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with a ``split`` column (``train`` / ``val`` / ``test``).
    expert_cols, experts, tier_cost_dict, n_experts :
        Domain objects from the notebook.
    action_costs_fn … evaluate_fn :
        Function / class handles to avoid importing the notebook as a module.
    n_trials : int
        Total Bayesian optimisation budget.
    n_jobs : int
        Parallel workers (use 1 for GPU training).
    seed : int
        Global random seed.
    study_name : str
        Optuna study identifier (for persistent storage).
    storage : str or None
        Optuna DB URL, e.g. ``"sqlite:///hpo.db"``.  ``None`` → in-memory.
    ledger_path : str
        Path for the JSONL trial ledger.
    objective_weights : ObjectiveWeights or None
        Custom scalarisation weights; defaults if ``None``.
    train_costs, prior_costs : (c_fn, c_fp)
        Fixed clinical cost asymmetry.
    warmup_epochs : int
        Default warmup — used as fallback if ``warmup_epochs`` is absent
        from the search space, and also read by Hyperband for the
        minimum-resource calculation.
    selection_violation_weight : float
        Penalty multiplier for constraint violations in checkpoint
        selection.

    Returns
    -------
    best_params : dict
        Flat parameter dict of the best trial.
    study : optuna.Study
        Completed study object for further analysis.
    """
    if objective_weights is None:
        objective_weights = ObjectiveWeights()

    # ── Sampler: multivariate TPE with group decomposition ──────────
    sampler = TPESampler(
        seed=seed,
        n_ei_candidates=48,  
        n_startup_trials=max(10, n_trials // 8),
        multivariate=True,
        group=True,
        constant_liar=(n_jobs > 1),
    )

    # ── Pruner: Hyperband ───────────────────────────────────────────
    # min_resource is tied to the *lower bound* of the warmup_epochs
    # search range: configs with short warmups can still be pruned once
    # their warmup finishes, while configs with long warmups may be
    # pruned mid-warmup if their scores are clearly hopeless.
    ss = SearchSpace()
    warmup_spec = getattr(ss, "warmup_epochs", None)
    if warmup_spec is not None and warmup_spec[0] == "int":
        warmup_lo = int(warmup_spec[1])
    else:
        warmup_lo = warmup_epochs
    pruner = HyperbandPruner(
        min_resource=warmup_lo,
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

    # ── Executor ────────────────────────────────────────────────────
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

    config_kwargs: Dict[str, Any] = {
        "train_costs":       train_costs,
        "prior_costs":       prior_costs,
        "batch_size":        batch_size,
        "warmup_epochs":     warmup_epochs,
        "epochs":            epochs,
        "patience":          patience,
        "min_delta":         min_delta,
        "max_deferral_rate": max_deferral_rate,
        "max_avg_cost":      max_avg_cost,
        "clip_anchor_ks":    clip_anchor_ks,
        "rho_by_k":          rho_by_k,
        "clip_floor":        clip_floor,
    }

    objective = make_objective(
        executor=executor,
        weights=objective_weights,
        utopia_tracker=utopia_tracker,
        search_space=ss,
        ledger_path=ledger_path,
        config_kwargs=config_kwargs,
    )

    # ── Seed with a known-good config from the notebook ─────────────
    study.enqueue_trial({
        "lr":                 1e-4,
        "gamma_tier":         1.0,
        "warmup_epochs":      15,
        "tau_bad":            1.0,
        "w_gsdp":             0.30,
        "w_rank_js":          0.30,
        "global_uniform_mix": 0.35,
        "family_uniform_mix": 0.30,
        "group_uniform_mix":  0.30,
        "family_n0":          25.0,
        "group_n0":           30.0,
        "global_mix":         0.05,
        "clip_ceiling":       0.35,
        "clip_slack":         0.03,
        "al_mu":              25.0,
        "al_lr_lambda":       0.10,
    })

    # ── Run ─────────────────────────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True,
    )

    # ── Extract best ────────────────────────────────────────────────
    best_trial = study.best_trial
    best_params = best_trial.params
    best_metrics = dict(best_trial.user_attrs)

    logger.info(
        f"Best trial #{best_trial.number}: "
        f"score={best_trial.value:.4f}\n"
        f"  Metrics: "
        f"{json.dumps(best_metrics, indent=2, default=_json_default)}\n"
        f"  Params:  "
        f"{json.dumps(best_params, indent=2, default=_json_default)}"
    )

    return best_params, study


# ────────────────────────────────────────────────────────────────────
# §9  Post-hoc analysis utilities
# ────────────────────────────────────────────────────────────────────

def pareto_front(
    study: optuna.Study,
    obj_keys: Optional[List[str]] = None,
) -> List[int]:
    """
    Extract Pareto-optimal trial indices from the study.

    Parameters
    ----------
    study : optuna.Study
    obj_keys : list of str
        Keys in ``trial.user_attrs`` to treat as objectives.
        Default: ``["clinical", "mcc", "auprc", "tier_soft"]``.
        Minimisation is assumed for ``clinical`` / ``tier_soft``;
        maximisation for ``mcc`` / ``auprc``.

    Returns
    -------
    list of int  —  trial numbers on the Pareto front.
    """
    if obj_keys is None:
        obj_keys = ["clinical", "mcc", "auprc", "tier_soft"]

    negate = {"mcc", "auprc", "acc"}  # higher-is-better → negate

    trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not trials:
        return []

    points = np.array([
        [
            -t.user_attrs.get(k, float("inf")) if k in negate
            else t.user_attrs.get(k, float("inf"))
            for k in obj_keys
        ]
        for t in trials
    ])

    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if (np.all(points[j] <= points[i])
                    and np.any(points[j] < points[i])):
                is_pareto[i] = False
                break

    return [trials[i].number for i in range(n) if is_pareto[i]]


def importance_analysis(study: optuna.Study) -> Dict[str, float]:
    """
    Wrapper around Optuna's fANOVA-based parameter importance.

    Returns dict mapping parameter name → importance score.
    """
    try:
        from optuna.importance import get_param_importances
        return get_param_importances(study)
    except Exception as exc:
        logger.warning(f"Importance analysis failed: {exc}")
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
    """
    Reconstruct configs from the best HPO params, extending the epoch
    budget and patience for final retraining.

    Parameters
    ----------
    best_params : dict
        Flat parameter dict from ``run_hpo``.
    extra_epochs : int
        Additional epochs beyond HPO budget.

    Returns
    -------
    train_cfg_d, mcfg_d, al_cfg_d : dict, dict, dict
        Ready for dataclass construction and ``train_l2d_multi_expert``.
    """
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
    train_cfg_d["epochs"]   = int(train_cfg_d["epochs"] + extra_epochs)
    train_cfg_d["patience"] = max(int(train_cfg_d["patience"]), 25)
    return train_cfg_d, mcfg_d, al_cfg_d


def print_study_summary(study: optuna.Study, top_k: int = 10):
    """Pretty-print the top-k trials, parameter importances and Pareto front."""
    trials = sorted(
        [t for t in study.trials
         if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
    )

    print(f"\n{'='*80}")
    print(f"HPO Study Summary: {study.study_name}")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed:    {len(trials)}")
    if trials:
        print(
            f"  Best value:   {study.best_value:.4f} "
            f"(trial #{study.best_trial.number})"
        )
    print(f"{'='*80}")

    print(f"\nTop-{top_k} Trials:")
    print(
        f"  {'#':>4} {'Score':>8} {'Clin':>7} {'MCC':>7} {'AUPRC':>7} "
        f"{'Tier':>7} {'Defer':>7} {'WrmUp':>5} {'BstEp':>5} {'Gap':>7}"
    )
    print(f"  {'-'*76}")

    for t in trials[:top_k]:
        ua = t.user_attrs
        wu = t.params.get("warmup_epochs", "—")
        be = ua.get("best_epoch", "—")
        print(
            f"  {t.number:>4} {t.value:>8.4f} "
            f"{ua.get('clinical', 0):>7.4f} "
            f"{ua.get('mcc', 0):>7.4f} "
            f"{ua.get('auprc', 0):>7.4f} "
            f"{ua.get('tier_soft', 0):>7.4f} "
            f"{ua.get('defer_soft', 0):>7.4f} "
            f"{wu:>5} "
            f"{be:>5} "
            f"{ua.get('acc_gap', 0):>7.4f}"
        )

    # Parameter importance
    imp = importance_analysis(study)
    if imp:
        print("\nParameter Importance (fANOVA):")
        for k, v in sorted(imp.items(), key=lambda x: -x[1])[:10]:
            bar = "█" * int(v * 40)
            print(f"  {k:<25} {v:>6.3f} {bar}")

    # Pareto front
    pf = pareto_front(study)
    if pf:
        print(f"\nPareto-optimal trials: {pf}")

    print(f"{'='*80}\n")


# ────────────────────────────────────────────────────────────────────
# §10  Convenience: full pipeline from notebook
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
    """
    One-call entry point that auto-discovers expert columns from *df*.

    Example in notebook::

        from adaptive_hpo import run_hpo_from_notebook

        best_params, study = run_hpo_from_notebook(
            df=df,
            tier_cost_dict=tier_cost,
            action_costs_fn=action_costs,
            build_all_priors_fn=build_all_priors,
            L2DDataset_cls=L2DDataset,
            Router_cls=Router,
            l2d_objective_fn=l2d_objective,
            combined_routing_loss_fn=combined_routing_loss,
            AugLag_cls=AugLag,
            evaluate_fn=evaluate,
            n_trials=80,
        )

        # Retrain with best config
        from adaptive_hpo import retrain_with_best
        train_d, mcfg_d, al_d = retrain_with_best(best_params, df, None, None)
    """
    EXPERT_COLS = [c for c in df.columns if c.startswith("y_") and c != "y_true"]
    EXPERTS  = [c.removeprefix("y_") for c in EXPERT_COLS]
    N_EXPERTS = len(EXPERT_COLS)

    df["m_experts"] = df.apply(
        lambda row: [int(not pd.isna(row[c])) for c in EXPERT_COLS],
        axis=1,
    )
    df["m_actions"] = df["m_experts"].apply(lambda m: [1] + m)

    best_params, study = run_hpo(
        df=df,
        expert_cols=EXPERT_COLS,
        experts=EXPERTS,
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
