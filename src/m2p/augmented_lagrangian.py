"""
Augmented-Lagrangian dual update for the MPD2-Router constraints.

Two inequality constraints can be enforced jointly:

* ``defer_rate ≤ ρ`` (``cfg.max_deferral_rate``);
* ``avg_cost   ≤ C`` (``cfg.max_avg_cost``).

For each violated constraint we add the standard augmented-Lagrangian term
``λ · g + ½ μ · max(0, g)²`` to the loss and update the dual variable
``λ ← max(0, λ + lr_λ · g)`` once per epoch.
"""

from __future__ import annotations

import torch

from .configs import ALConfig


class AugLag:
    """Stateful augmented-Lagrangian helper with non-negative dual variables."""

    def __init__(self, cfg: ALConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.lam_def  = torch.zeros((), device=device)
        self.lam_cost = torch.zeros((), device=device)

    def penalty(self, defer_rate: torch.Tensor,
                avg_cost: torch.Tensor) -> torch.Tensor:
        pen = torch.zeros((), device=self.device)

        if self.cfg.max_deferral_rate is not None:
            g = defer_rate - self.cfg.max_deferral_rate
            pen = pen + self.lam_def * g + 0.5 * self.cfg.mu * torch.clamp(g, min=0.0) ** 2

        if self.cfg.max_avg_cost is not None:
            g = avg_cost - self.cfg.max_avg_cost
            pen = pen + self.lam_cost * g + 0.5 * self.cfg.mu * torch.clamp(g, min=0.0) ** 2

        return pen

    @torch.no_grad()
    def update(self, defer_rate: torch.Tensor, avg_cost: torch.Tensor) -> None:
        if self.cfg.max_deferral_rate is not None:
            g = defer_rate - self.cfg.max_deferral_rate
            self.lam_def = torch.clamp(
                self.lam_def + self.cfg.lr_lambda * g, min=0.0,
            )

        if self.cfg.max_avg_cost is not None:
            g = avg_cost - self.cfg.max_avg_cost
            self.lam_cost = torch.clamp(
                self.lam_cost + self.cfg.lr_lambda * g, min=0.0,
            )


__all__ = ["AugLag"]
