"""
Routing-regularisation losses.

* :func:`gsdp_loss` — group-conditional KL/JS between the empirical routing
  distribution ``q̄_g`` and the precomputed group prior ``p_g``.
* :func:`rank_majorization_js_loss` — per-sample JS gate against a truncated
  geometric rank prior, only active when the sample's sorted prefix sums
  exceed the geometric reference (i.e. the routing is "too peaky").
* :func:`combined_routing_loss` — weighted sum of the two for training.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .configs import PriorRegConfig


# ─────────────────────────────────────────────────────────────────────
# Probability utilities
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# GSDP — group-structural divergence prior
# ─────────────────────────────────────────────────────────────────────

def gsdp_loss(q: torch.Tensor, expert_mask: torch.Tensor,
              group_ids: torch.Tensor,
              group_prior_table: torch.Tensor,
              group_id_to_row: Dict[int, int],
              d: Optional[torch.Tensor] = None,
              divergence: str = "kl",
              eps: float = 1e-8) -> torch.Tensor:
    """
    GSDP on group-average conditional routing.

    .. code-block:: text

        q̄_g = Σ_i w_i · q_i / Σ_i w_i,   i ∈ group g
        loss = Σ_g w_g · D(q̄_g ‖ p_g) / Σ_g w_g
    """
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


# ─────────────────────────────────────────────────────────────────────
# Rank-majorisation JS
# ─────────────────────────────────────────────────────────────────────

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
    """
    Majorisation-gated JS against a truncated-geometric rank prior.

    1) Normalise ``q`` over active experts.
    2) Sort descending; compare cumulative prefix sums against the geometric
       reference :func:`truncated_geometric_prior`.
    3) Apply ``JS(sorted, geometric)`` only when the sample is *sharper* than
       the reference (``mode='any_excess'``) or when *every* prefix exceeds
       the reference (``mode='all_prefixes'``).
    """
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


# ─────────────────────────────────────────────────────────────────────
# Combined routing loss
# ─────────────────────────────────────────────────────────────────────

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


__all__ = [
    "safe_masked_prob_torch", "kl_div_prob", "js_div_prob",
    "gsdp_loss", "truncated_geometric_prior", "default_rho_by_k",
    "rank_majorization_js_loss", "combined_routing_loss",
]
