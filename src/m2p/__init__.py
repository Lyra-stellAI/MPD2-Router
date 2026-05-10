"""
MPD2-Router (M2P) — multi-expert learning-to-defer for glaucoma.

Public submodules
-----------------
* ``m2p.grouping``                  — three-stage hierarchical bucketing.
* ``m2p.ood``                       — OOD detectors (MSP/Energy/kNN/ViM/Mahalanobis).
* ``m2p.adaptive_hpo``              — constraint-aware Bayesian HPO (current).
* ``m2p.adaptive_hpo_geometric_clip`` — earlier HPO variant retained for
  reproducibility of submitted experiments.
"""

__version__ = "0.1.0"
