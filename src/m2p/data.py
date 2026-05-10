"""
PyTorch ``Dataset`` for the multi-expert L2D pipeline.

Every row of the underlying DataFrame must already carry:

* ``y_true`` and the per-expert label columns ``y_*`` (NaN where unavailable);
* the structural features ``vCDR``, ``aCDR``;
* the OOD risk scores ``vim_risk_z``, ``quality_risk``, ``uncertainty``;
* the AI logits ``logit_0``, ``logit_1`` (and optionally ``prob_1``);
* the grouping columns ``group_id`` and ``dataset``.

The grouping columns are produced by :mod:`m2p.grouping`; the risk columns by
:mod:`m2p.ood`; the embeddings by :mod:`m2p.feature_extraction`.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────
# Numerical helpers
# ─────────────────────────────────────────────────────────────────────

def prob1_from_logits_np(logit0: np.ndarray, logit1: np.ndarray) -> np.ndarray:
    """Numerically-stable ``softmax(logit)[1]`` on numpy."""
    m = np.maximum(logit0, logit1)
    e0 = np.exp(logit0 - m)
    e1 = np.exp(logit1 - m)
    p1 = e1 / (e0 + e1 + 1e-12)
    return np.clip(p1, 1e-6, 1.0 - 1e-6).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class L2DDataset(Dataset):
    """
    Returns a 14-tuple per sample::

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

        # Expert columns are auto-discovered.
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

        # Tabular features.
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


__all__ = ["L2DDataset", "prob1_from_logits_np"]
