"""
MPD2-Router (M2P) — multi-expert learning-to-defer for glaucoma.

Public submodules
-----------------
* :mod:`m2p.feature_extraction` — frozen Swin-V2 logits / pooled embeddings.
* :mod:`m2p.ood`                — MSP / Energy / kNN / ViM / Mahalanobis.
* :mod:`m2p.grouping`           — three-stage hierarchical bucketing.
* :mod:`m2p.router_training`    — consolidated training pipeline (configs,
                                  data, model, priors, losses, augmented
                                  Lagrangian, training loop, evaluation).
* :mod:`m2p.adaptive_hpo`       — constraint-aware Bayesian HPO (consolidated
                                  ``v2`` + ``geometric_clip`` lineages).
"""

from .router_training import (
    # configs
    ALConfig, PriorRegConfig, TrainConfig,
    # costs / objective
    EXPERT_TIER, TIER_COST,
    action_costs, ai_expected_clinical_cost, expert_clinical_cost,
    l2d_objective, per_action_clinical_cost,
    # dataset
    L2DDataset,
    # model
    OVAExpertHeadFixed, Router, StructuralRisk,
    # priors
    build_all_priors,
    # routing-regularisation losses
    combined_routing_loss, gsdp_loss, rank_majorization_js_loss,
    # constraint enforcement
    AugLag,
    # training & evaluation
    train_l2d_multi_expert, evaluate,
    print_eval_report, print_eval_report_dataset,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ALConfig", "PriorRegConfig", "TrainConfig",
    "EXPERT_TIER", "TIER_COST",
    "action_costs", "ai_expected_clinical_cost", "expert_clinical_cost",
    "l2d_objective", "per_action_clinical_cost",
    "L2DDataset",
    "OVAExpertHeadFixed", "Router", "StructuralRisk",
    "build_all_priors",
    "combined_routing_loss", "gsdp_loss", "rank_majorization_js_loss",
    "AugLag",
    "train_l2d_multi_expert", "evaluate",
    "print_eval_report", "print_eval_report_dataset",
]
