"""
MPD2-Router (M2P) — multi-expert learning-to-defer for glaucoma.

Public API
----------
* :mod:`m2p.feature_extraction` — frozen Swin-V2 logits / hidden states.
* :mod:`m2p.ood`                 — MSP / Energy / kNN / ViM / Mahalanobis.
* :mod:`m2p.grouping`            — three-stage hierarchical bucketing.
* :mod:`m2p.priors`              — global → family → group prior construction.
* :mod:`m2p.costs`               — clinical-cost asymmetry, tier costs, L2D objective.
* :mod:`m2p.models`              — Router + OVA expert head + structural head.
* :mod:`m2p.losses`              — GSDP and rank-majorisation JS regularisers.
* :mod:`m2p.augmented_lagrangian`— constraint enforcement via dual updates.
* :mod:`m2p.configs`             — :class:`TrainConfig`, :class:`PriorRegConfig`,
                                   :class:`ALConfig`.
* :mod:`m2p.data`                — :class:`L2DDataset`.
* :mod:`m2p.training`            — :func:`train_l2d_multi_expert` end-to-end loop.
* :mod:`m2p.evaluation`          — :func:`evaluate` and reporting helpers.
* :mod:`m2p.adaptive_hpo`        — current Bayesian HPO module (v3).
* :mod:`m2p.adaptive_hpo_geometric_clip` — earlier HPO variant kept for repro.
"""

from .configs import ALConfig, PriorRegConfig, TrainConfig
from .costs import (
    EXPERT_TIER, TIER_COST,
    action_costs, ai_expected_clinical_cost, expert_clinical_cost,
    l2d_objective, per_action_clinical_cost,
)
from .data import L2DDataset
from .losses import combined_routing_loss, gsdp_loss, rank_majorization_js_loss
from .models import OVAExpertHeadFixed, Router, StructuralRisk
from .priors import build_all_priors
from .augmented_lagrangian import AugLag

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ALConfig", "PriorRegConfig", "TrainConfig",
    "EXPERT_TIER", "TIER_COST",
    "action_costs", "ai_expected_clinical_cost", "expert_clinical_cost",
    "l2d_objective", "per_action_clinical_cost",
    "L2DDataset",
    "combined_routing_loss", "gsdp_loss", "rank_majorization_js_loss",
    "OVAExpertHeadFixed", "Router", "StructuralRisk",
    "build_all_priors",
    "AugLag",
]
