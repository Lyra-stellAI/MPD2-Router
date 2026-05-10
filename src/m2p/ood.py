"""
OOD detection / risk estimation for the MPD2-Router pipeline.

This module bundles the post-hoc detectors used to compute per-sample OOD
scores from a frozen Swin-V2 glaucoma classifier
(``pamixsun/swinv2_tiny_for_glaucoma_classification``).  The detectors are
fit on the in-distribution train split (REFUGE) and scored on held-out
splits (REFUGE val/test) and on out-of-distribution datasets (ORIGA,
CHAKSU).

Detectors
---------
* **MSP / MaxLogit / Entropy / Energy**     — logit-only baselines.
* **Energy + ReAct**                         — clamps activations at the
  REACT_PCTL percentile of train pooled embeddings before re-running the
  classifier head.
* **kNN**                                    — distance to the k-th nearest
  train embedding (cosine-normalised L2).
* **ViM**                                    — Virtual-logit Matching:
  alpha * ||residual to PCA subspace|| - logsumexp(logits).
* **Mahalanobis**                            — multi-layer Mahalanobis with
  Ledoit-Wolf shrinkage on hidden states 2/3/4.

The main outputs are two standardised "risk" scores (``maha_risk``,
``vim_risk``) joined onto the dataset CSV and consumed by the router as
features.

References
----------
Lakshminarayanan et al. (2017), Hendrycks & Gimpel (2017),
Liang et al. (2018), Liu et al. (2020), Sun et al. (2021),
Sun et al. (2022) ViM, Lee et al. (2018) Mahalanobis.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _find_head(model):
    for attr in ("classifier", "head", "fc"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(
        "Cannot find classifier head (expected .classifier / .head / .fc)"
    )


# ─────────────────────────────────────────────────────────────────────
# Logit-based scores
# ─────────────────────────────────────────────────────────────────────

def score_msp(logits: torch.Tensor) -> np.ndarray:
    """Negated maximum softmax probability — higher is more OOD."""
    return (1.0 - logits.softmax(1).max(1).values).numpy()


def score_maxlogit(logits: torch.Tensor) -> np.ndarray:
    return (-logits.max(1).values).numpy()


def score_entropy(logits: torch.Tensor) -> np.ndarray:
    p = logits.softmax(1).clamp_min(1e-12)
    return (-(p * p.log()).sum(1)).numpy()


def score_energy(logits: torch.Tensor, T: float = 1.0) -> np.ndarray:
    """Negative free energy.  Sign-flipped so higher = more OOD."""
    return -torch.logsumexp(logits / T, dim=1).numpy()


def score_energy_react(model, feats: torch.Tensor, T: float = 1.0,
                       clip_c: float = 10.0,
                       device: Optional[torch.device] = None) -> np.ndarray:
    """Energy after ReAct activation clamping at percentile ``clip_c``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = _find_head(model)
    with torch.inference_mode():
        logits = head(feats.to(device).clamp(max=clip_c))
    return score_energy(logits.cpu(), T)


# ─────────────────────────────────────────────────────────────────────
# Feature-space detectors
# ─────────────────────────────────────────────────────────────────────

class KNN_OOD:
    """Distance to k-th cosine-normalised nearest train neighbour."""

    def __init__(self, k: int = 10, normalize: bool = True):
        self.k = k
        self.normalize = normalize

    def _prep(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.normalize:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X

    def fit(self, train_feats):
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(self._prep(train_feats))
        return self

    def score(self, feats):
        return self.nn.kneighbors(self._prep(feats), return_distance=True)[0][:, -1]


class ViM_OOD:
    """Virtual-logit Matching: alpha * ||residual|| - logsumexp(logits)."""

    def __init__(self, pca_dim: int = 256):
        self.pca_dim = pca_dim

    def fit(self, train_feats, train_logits):
        X = train_feats.astype(np.float32)
        self.mu = X.mean(0, keepdims=True)
        Xc = X - self.mu

        d = min(self.pca_dim, Xc.shape[0] - 1, Xc.shape[1])
        pca = PCA(n_components=d, svd_solver="full", random_state=0).fit(Xc)
        self.V = pca.components_.T  # [D, d]

        rnorm = np.linalg.norm(Xc - Xc @ self.V @ self.V.T, axis=1)
        self.alpha = float(train_logits.max(1).mean() / (rnorm.mean() + 1e-12))
        return self

    def score(self, feats, logits):
        X = feats.astype(np.float32) - self.mu
        rnorm = np.linalg.norm(X - X @ self.V @ self.V.T, axis=1)
        lse = torch.logsumexp(
            torch.as_tensor(logits, dtype=torch.float32), dim=1
        ).numpy()
        return self.alpha * rnorm - lse


class Mahalanobis_OOD:
    """Multi-layer Mahalanobis distance with Ledoit-Wolf shrinkage."""

    def __init__(self, layers: Iterable[int] = (2, 3, 4), n_classes: int = 2):
        self.layers = tuple(int(x) for x in layers)
        self.n_classes = n_classes
        self.means: Dict[int, np.ndarray] = {}
        self.icov: Dict[int, np.ndarray] = {}

    def fit(self, feats_by_layer, y):
        y = y.astype(int)
        for l in self.layers:
            X = feats_by_layer[l].astype(np.float32)
            self.means[l] = np.stack(
                [X[y == c].mean(0) for c in range(self.n_classes)]
            )
            cov = LedoitWolf().fit(X).covariance_.astype(np.float32)
            cov += 1e-6 * np.eye(X.shape[1], dtype=np.float32)
            self.icov[l] = np.linalg.inv(cov)
        return self

    def score(self, feats_by_layer):
        total = None
        for l in self.layers:
            X = feats_by_layer[l].astype(np.float32)
            diff = X[:, None, :] - self.means[l][None, :, :]
            dists = np.einsum("ncd,de,nce->nc", diff, self.icov[l], diff).min(axis=1)
            total = dists if total is None else total + dists
        return total


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────

def ood_metrics(id_scores, ood_scores, tau: float = 99.0) -> Dict[str, float]:
    """AUROC / AUPR-OOD / FPR at the τ-th percentile of the ID scores."""
    y = np.concatenate([
        np.zeros(len(id_scores), dtype=int),
        np.ones(len(ood_scores), dtype=int),
    ])
    s = np.concatenate([id_scores, ood_scores]).astype(np.float64)

    tau_val = np.percentile(id_scores, tau)
    return {
        "AUROC":     float(roc_auc_score(y, s)),
        "AUPR_OOD":  float(average_precision_score(y, s)),
        "FPR99":     float(np.mean(ood_scores <= tau_val)),
        f"tau@{int(tau)}ID": float(tau_val),
    }


__all__ = [
    "score_msp", "score_maxlogit", "score_entropy",
    "score_energy", "score_energy_react",
    "KNN_OOD", "ViM_OOD", "Mahalanobis_OOD",
    "ood_metrics",
]
