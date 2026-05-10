"""
MPD2-Router architecture.

The router has three feature branches that are concatenated before the
deferral head and the OVA expert head:

* ``risk_enc``    — 3 OOD / quality features.
* ``struct_enc``  — 2 calibrated structural-risk features
                    (``struct_margin`` and ``ai_struct_gap``).
* ``ai_enc``      — 2 raw AI logits.

The deferral head emits a sigmoid scalar; the expert head implements the
OVA gate-and-allocate scheme of :class:`OVAExpertHeadFixed`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
# Structural-risk module (calibrated linear head over CDR features)
# ─────────────────────────────────────────────────────────────────────

class StructuralRisk(nn.Module):
    """
    Calibrated linear glaucoma-risk head over ``(vCDR, aCDR)``.

    The default initialisation reproduces a logistic-regression model fitted
    on REFUGE-train (coefficients ``[7.51, 8.04]``, intercept ``-6.84``);
    ``trainable=True`` lets the L2D optimiser fine-tune it.
    """

    def __init__(self, trainable: bool = True):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        coef = [[7.50903265, 8.04041925]]
        bias = [-6.84406693]
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(coef, dtype=torch.float32))
            self.linear.bias.copy_(torch.tensor(bias, dtype=torch.float32))
        if not trainable:
            for p in self.linear.parameters():
                p.requires_grad = False

    def forward(self, vcdr, acdr):
        x = torch.stack([vcdr, acdr], dim=-1)
        z = self.linear(x).squeeze(-1)
        return z, torch.sigmoid(z)


# ─────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────

class MLPBlock(nn.Module):
    """``Linear → ReLU → LayerNorm → Dropout`` building block."""

    def __init__(self, in_dim: int, out_dim: int, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
# OVA expert head
# ─────────────────────────────────────────────────────────────────────

class OVAExpertHeadFixed(nn.Module):
    """
    Two-stage one-vs-all expert head with differentiable allocation.

    Stage 1 (gating).  An OVA classifier emits ``gate_probs[b, j]``; during
    training a Gumbel-sigmoid sample drives a straight-through hard
    selection.  Inactive (unavailable) experts are masked out.

    Stage 2 (allocation).  A separate softmax over the gated subset assigns
    routing mass ``q[b, j]`` proportional to per-sample affinity, then the
    output is masked again to forget any leakage onto unavailable experts.

    This factorisation keeps gradient flow through *both* the selection
    decision and the allocation weights, which prevented the original
    "gate-only" head from learning informative allocations.
    """

    def __init__(self, fused_dim: int, hidden: int, n_experts: int,
                 p_drop: float = 0.1, hard: bool = True):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(p_drop),
        )
        self.gate_logits  = nn.Linear(hidden, n_experts)
        self.alloc_logits = nn.Linear(hidden, n_experts)
        self.hard = hard

    def forward(self, h, mask, tau: float = 1.0):
        z = self.trunk(h)

        # ---- Stage 1: OVA gating ----
        g_logits = self.gate_logits(z) / tau
        g_logits = g_logits.masked_fill(mask <= 0, -1e8)
        gate_probs = torch.sigmoid(g_logits)

        if self.training:
            u = torch.rand_like(g_logits).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = torch.log(u) - torch.log(1 - u)
            selection_soft = torch.sigmoid((g_logits + gumbel_noise) / tau)
            if self.hard:
                # Straight-through estimator
                selection = (
                    (selection_soft > 0.5).float()
                    - selection_soft.detach()
                    + selection_soft
                )
            else:
                selection = selection_soft
        else:
            selection = (gate_probs > 0.5).float()

        selection = selection * mask

        # Fallback: if all gates closed, revert to the full availability mask.
        dead = (selection.sum(-1, keepdim=True) == 0).float()
        selection = selection + dead * mask

        # ---- Stage 2: allocation ----
        a_logits = self.alloc_logits(z) / tau
        a_logits_masked = a_logits.masked_fill(mask <= 0, -1e8)
        alloc_probs = torch.softmax(a_logits_masked, dim=-1)

        q = alloc_probs * selection
        q = q / q.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return q, gate_probs, selection, a_logits


# ─────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────

class Router(nn.Module):
    """MPD2-Router: deferral gate + OVA expert allocator on fused features."""

    def __init__(self, n_experts: int, hidden: int = 64,
                 hidden_expert: int = 128, p_drop: float = 0.1,
                 init_tau: float = 1.25):
        super().__init__()

        self.struct = StructuralRisk(trainable=True)

        # risk branch: (vim_risk, quality_risk, uncertainty)
        self.risk_enc   = MLPBlock(3, hidden, p_drop=p_drop)
        # struct branch: (struct_margin, ai_struct_gap)
        self.struct_enc = MLPBlock(2, hidden, p_drop=p_drop)
        # AI logit branch: (logit_0, logit_1)
        self.ai_enc     = MLPBlock(2, hidden, p_drop=p_drop)

        fused_dim = 3 * hidden

        self.defer_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(p_drop),
            nn.Linear(128, 1),
        )

        self.expert_head = OVAExpertHeadFixed(
            fused_dim=fused_dim,
            hidden=hidden_expert,
            n_experts=n_experts,
            p_drop=p_drop,
            hard=True,
        )

        self.log_tau = nn.Parameter(
            torch.log(torch.tensor(init_tau, dtype=torch.float32))
        )

    def forward(self, vim_risk, quality_risk, uncertainty, vcdr, acdr,
                p_ai, logit_0, logit_1, action_mask):
        # ---- Structural-risk side branch ----
        z_struct, p_struct = self.struct(vcdr, acdr)
        struct_margin = 2.0 * p_struct - 1.0
        ai_struct_gap = torch.abs(p_ai - p_struct)

        # ---- Branch encoders ----
        h_risk   = self.risk_enc(
            torch.stack([vim_risk, quality_risk, uncertainty], dim=-1)
        )
        h_struct = self.struct_enc(
            torch.stack([struct_margin, ai_struct_gap], dim=-1)
        )
        h_ai     = self.ai_enc(torch.stack([logit_0, logit_1], dim=-1))

        h = torch.cat([h_risk, h_struct, h_ai], dim=-1)

        # ---- Defer head ----
        d = torch.sigmoid(self.defer_head(h)).squeeze(-1)

        # ---- Expert head ----
        expert_mask = action_mask[:, 1:]
        tau = torch.exp(self.log_tau).clamp(min=0.5)
        q, gate_probs, selection, expert_logits = self.expert_head(
            h, expert_mask, tau,
        )

        # ---- Compose full policy ----
        pi = torch.cat([(1 - d).unsqueeze(-1), d.unsqueeze(-1) * q], dim=-1)

        aux = {
            "d":             d,
            "q":             q,
            "expert_logits": expert_logits,
            "expert_mask":   expert_mask,
            "gate_probs":    gate_probs,
            "selection":     selection,
            "z_struct":      z_struct,
            "p_struct":      p_struct,
            "struct_margin": struct_margin,
            "ai_struct_gap": ai_struct_gap,
            "tau":           tau,
            "risk_feats":    torch.stack(
                [vim_risk, quality_risk, uncertainty, z_struct, ai_struct_gap],
                dim=-1,
            ),
        }
        return pi, expert_logits, aux


__all__ = ["StructuralRisk", "MLPBlock", "OVAExpertHeadFixed", "Router"]
