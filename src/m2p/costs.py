"""
Clinical cost asymmetry and expert tier-cost tables.

This module is the single source of truth for the per-expert *scarcity* costs
(``tier_cost``) and the expert-tier mapping used throughout the paper.  The
default values reproduce the trained system reported in our submission.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────
# Clinical-cost defaults
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# Cost vectors and per-action clinical cost
# ─────────────────────────────────────────────────────────────────────

def action_costs(tier_cost: Dict[str, float],
                 experts: List[str]) -> np.ndarray:
    """
    ``costs[a]`` is the tier cost incurred by selecting action *a*.

    * ``a = 0``       → AI prediction, cost 0.
    * ``a = 1 + j``   → expert ``experts[j]`` with cost ``tier_cost[expert]``.
    """
    costs = np.zeros((1 + len(experts),), dtype=np.float32)
    for j, e in enumerate(experts):
        costs[1 + j] = float(tier_cost[e])
    return costs


def expert_clinical_cost(y_true: torch.Tensor, y_exp: torch.Tensor,
                         c_fn: float, c_fp: float) -> torch.Tensor:
    """
    Per-expert oracle clinical cost.

    Parameters
    ----------
    y_true : ``[B]``
    y_exp  : ``[B, M]`` — may contain NaN for unavailable experts; the caller
             is responsible for masking those out via the availability mask.
    """
    y_exp_filled = torch.nan_to_num(y_exp, nan=0.0)
    yt = y_true.unsqueeze(1)
    fn = (yt == 1) & (y_exp_filled == 0)
    fp = (yt == 0) & (y_exp_filled == 1)
    return c_fn * fn.float() + c_fp * fp.float()


def ai_expected_clinical_cost(y_true: torch.Tensor, p_ai: torch.Tensor,
                              c_fn: float, c_fp: float) -> torch.Tensor:
    """
    Differentiable expected FN/FP cost of the AI action.

    * ``y_true = 1`` → cost = ``c_fn · (1 − p_ai)``
    * ``y_true = 0`` → cost = ``c_fp · p_ai``
    """
    yt = (y_true > 0.5).float()
    return c_fn * yt * (1.0 - p_ai) + c_fp * (1.0 - yt) * p_ai


def per_action_clinical_cost(y_true: torch.Tensor, p_ai: torch.Tensor,
                             y_exp: torch.Tensor, c_fn: float,
                             c_fp: float) -> torch.Tensor:
    """``[B, 1+M]`` — column 0 is the AI cost, columns 1..M are expert costs."""
    L_ai  = ai_expected_clinical_cost(y_true, p_ai, c_fn, c_fp)
    L_exp = expert_clinical_cost(y_true, y_exp, c_fn, c_fp)
    return torch.cat([L_ai.unsqueeze(1), L_exp], dim=1)


# ─────────────────────────────────────────────────────────────────────
# L2D objective
# ─────────────────────────────────────────────────────────────────────

def l2d_objective(pi: torch.Tensor, action_mask: torch.Tensor,
                  y_true: torch.Tensor, p_ai: torch.Tensor,
                  y_exp: torch.Tensor, tier_costs: torch.Tensor,
                  c_fn: float, c_fp: float):
    """
    Core learning-to-defer objective.

    Returns ``(exp_clin, exp_tier, defer_rate, pi_masked)`` where

    * ``exp_clin``   — expected clinical cost over the policy ``pi``;
    * ``exp_tier``   — expected tier (consultation) cost over ``pi``;
    * ``defer_rate`` — soft total expert mass ``1 - π[:, 0]``;
    * ``pi_masked``  — re-normalised policy after masking inactive actions.
    """
    pi = pi * action_mask
    pi = pi / (pi.sum(dim=1, keepdim=True) + 1e-8)

    L_clin = per_action_clinical_cost(y_true, p_ai, y_exp, c_fn, c_fp)

    exp_clin   = (pi * L_clin).sum(dim=1).mean()
    exp_tier   = (pi * tier_costs.view(1, -1)).sum(dim=1).mean()
    defer_rate = 1.0 - pi[:, 0].mean()
    return exp_clin, exp_tier, defer_rate, pi


__all__ = [
    "C_FN_BASE", "C_FP_BASE", "TIER_COST", "EXPERT_TIER",
    "action_costs", "expert_clinical_cost",
    "ai_expected_clinical_cost", "per_action_clinical_cost",
    "l2d_objective",
]
