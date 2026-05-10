"""
Configuration dataclasses for the MPD2-Router pipeline.

* :class:`TrainConfig` — optimisation, schedule, and clinical cost asymmetry.
* :class:`PriorRegConfig` — hierarchical prior construction + GSDP / Rank-JS
  regularisation knobs.
* :class:`ALConfig` — augmented Lagrangian dual-update knobs for the
  deferral-rate and average-cost constraints.

The notebook also defined a shorter ``EXPERT_TIER`` / ``tier_cost`` table; those
live in :mod:`m2p.costs` because they are first-class clinical-cost data,
not configuration knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# Augmented Lagrangian
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ALConfig:
    """Augmented Lagrangian dual-update knobs."""
    mu:                float            = 20.0
    lr_lambda:         float            = 0.01
    max_deferral_rate: Optional[float]  = 0.75
    max_avg_cost:      Optional[float]  = None
    gamma_tier:        float            = 1.0   # used in eval scoring


# ─────────────────────────────────────────────────────────────────────
# Hierarchical prior + routing-regularisation knobs
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PriorRegConfig:
    """Single source of truth for prior construction and routing regularisation."""

    # ---- badness → softmax temperature ----
    tau_bad: float = 1.0

    # ---- uniform mixing at each hierarchy level ----
    global_uniform_mix: float = 0.50
    family_uniform_mix: float = 0.35
    group_uniform_mix:  float = 0.35

    # ---- hierarchical shrinkage pseudo-counts ----
    family_n0: float = 25.0
    group_n0:  float = 30.0

    # ---- global prior bleed-through at group level ----
    global_mix: float = 0.05

    # ---- clinical cost asymmetry used in badness aggregation ----
    c_fn: float = 1.8
    c_fp: float = 1.2

    # ---- anti-collapse cap by support size ----
    clip_max_anchors: Dict[int, float] = field(
        default_factory=lambda: {5: 0.36, 7: 0.35, 12: 0.34}
    )

    # ---- Beta-smoothing for FNR/FPR ----
    alpha: float = 1.0
    beta:  float = 1.0

    # ---- optional per-expert tier cost & capacity ----
    tier_cost: Optional[Sequence[float]] = None
    capacity:  Optional[Sequence[float]] = None

    # ---- column names ----
    split_col:   str = "split"
    train_value: str = "train"
    mask_col:    str = "m_actions"
    y_col:       str = "y_true"
    family_col:  str = "support_family"
    group_col:   str = "group_id"

    # ---- GSDP loss ----
    gsdp_divergence: str = "kl"          # "kl" or "js"

    # ---- Rank-JS loss ----
    rank_js_margin:    float = 0.0
    rank_js_mode:      str   = "any_excess"   # "any_excess" | "all_prefixes"
    rank_js_reduction: str   = "mean"

    # ---- combined-loss weights ----
    w_gsdp:    float = 1.0
    w_rank_js: float = 1.0


__all__ = ["TrainConfig", "ALConfig", "PriorRegConfig"]
