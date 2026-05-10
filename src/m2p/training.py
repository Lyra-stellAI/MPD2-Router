"""
End-to-end training loop for the MPD2-Router.

Combines:

* :class:`~m2p.data.L2DDataset` and :class:`~m2p.models.Router`,
* :func:`~m2p.priors.build_all_priors`,
* :func:`~m2p.costs.l2d_objective` and
  :func:`~m2p.losses.combined_routing_loss`,
* :class:`~m2p.augmented_lagrangian.AugLag` constraint enforcement,
* per-epoch validation via :func:`~m2p.evaluation.evaluate` and
  patience-based early stopping with a warmup window.
"""

from __future__ import annotations

import copy
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .augmented_lagrangian import AugLag
from .configs import ALConfig, PriorRegConfig, TrainConfig
from .costs import TIER_COST, action_costs, l2d_objective
from .data import L2DDataset
from .evaluation import diagnostic_checks, evaluate
from .losses import combined_routing_loss
from .models import Router
from .priors import build_all_priors


def _make_optimizer(router: Router, lr: float) -> torch.optim.Optimizer:
    """AdamW with separate parameter groups; ``log_tau`` has zero weight decay."""
    return torch.optim.AdamW([
        {"params": list(router.risk_enc.parameters()),    "lr": lr, "weight_decay": 1e-4},
        {"params": list(router.struct_enc.parameters()),  "lr": lr, "weight_decay": 1e-4},
        {"params": list(router.ai_enc.parameters()),      "lr": lr, "weight_decay": 1e-4},
        {"params": list(router.defer_head.parameters()),  "lr": lr, "weight_decay": 1e-4},
        {"params": list(router.expert_head.parameters()), "lr": lr, "weight_decay": 1e-4},
        {"params": [router.log_tau],                      "lr": lr, "weight_decay": 0.0},
        {"params": list(router.struct.parameters()),      "lr": lr, "weight_decay": 1e-4},
    ])


def train_l2d_multi_expert(df: pd.DataFrame,
                           cfg: Optional[TrainConfig] = None,
                           mcfg: Optional[PriorRegConfig] = None,
                           al_cfg: Optional[ALConfig] = None,
                           shuffle: bool = False):
    """
    Train the MPD2-Router on the train split of *df* with constraint-aware
    early stopping on the val split.

    Returns ``(router, priors, all_q, all_pi)``.  The router is re-loaded with
    the best-scoring checkpoint before returning.
    """
    cfg = cfg or TrainConfig()
    mcfg = mcfg or PriorRegConfig()
    al_cfg = al_cfg or ALConfig()
    device = torch.device(cfg.device)

    EXPERT_COLS = [c for c in df.columns
                   if c.startswith("y_") and c != "y_true"]
    EXPERTS = [c.removeprefix("y_") for c in EXPERT_COLS]
    N_EXPERTS = len(EXPERT_COLS)

    df["m_experts"] = df.apply(
        lambda row: [int(not pd.isna(row[c])) for c in EXPERT_COLS], axis=1,
    )
    df["m_actions"] = df["m_experts"].apply(lambda m: [1] + m)

    costs_np = action_costs(TIER_COST, EXPERTS)
    tier_costs = torch.tensor(costs_np, dtype=torch.float32, device=device)

    priors = build_all_priors(df, EXPERT_COLS, cfg=mcfg)
    group_prior_table = priors["final_group_table"].to(device)
    group_id_to_row = priors["final_id_to_row"]

    df_tr = df[df["split"].astype(str) == "train"].reset_index(drop=True)
    df_va = (
        df[df["split"].astype(str) == "val"].reset_index(drop=True)
        if (df["split"].astype(str) == "val").any() else None
    )
    print(f"Train: {len(df_tr)}, Val: {len(df_va) if df_va is not None else 0}")

    ds_tr = L2DDataset(df_tr)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=shuffle)
    dl_va = None
    if df_va is not None and len(df_va) > 0:
        ds_va = L2DDataset(df_va)
        dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False)

    router = Router(N_EXPERTS, hidden=128).to(device)
    opt = _make_optimizer(router, lr=cfg.lr)
    al = AugLag(al_cfg, device=device)

    best_score = float("inf")
    best_epoch = 0
    best_state = None
    patience_ctr = 0

    all_pi: list = []
    all_q: list = []

    for ep in range(1, cfg.epochs + 1):
        router.train()
        total_loss = n_epoch = 0
        defer_sum = tier_sum = gsdp_sum = rank_js_sum = 0.0

        for batch_idx, (vim, q, u, p_ai, vcdr, acdr, logit_0, logit_1,
                        y_true, y_exp, m_actions, expert_mask, ds_ids,
                        group_ids) in enumerate(dl_tr):

            vim, q, u = vim.to(device), q.to(device), u.to(device)
            vcdr, acdr = vcdr.to(device), acdr.to(device)
            p_ai = p_ai.to(device)
            logit_0, logit_1 = logit_0.to(device), logit_1.to(device)
            y_true, y_exp = y_true.to(device), y_exp.to(device)
            m_actions = m_actions.to(device)
            expert_mask = expert_mask.to(device)
            ds_ids, group_ids = ds_ids.to(device), group_ids.to(device)
            B = y_true.size(0)

            pi, expert_logits, aux = router(
                vim, q, u, vcdr, acdr, p_ai, logit_0, logit_1, m_actions,
            )
            aux = dict(aux)
            expert_mask = m_actions[:, 1:].to(expert_logits.dtype)
            aux["expert_logits"] = expert_logits
            aux["expert_mask"]   = expert_mask

            d = aux["d"].view(-1)
            q_route = aux["q"].view(B, N_EXPERTS)
            expert_mask_route = m_actions[:, 1:].to(q_route.dtype)

            all_q.append(q_route.detach().cpu())
            all_pi.append(pi.detach().cpu())

            exp_clin, exp_tier, defer_rate, _ = l2d_objective(
                pi=pi, action_mask=m_actions,
                y_true=y_true, p_ai=p_ai, y_exp=y_exp,
                tier_costs=tier_costs,
                c_fn=cfg.c_fn, c_fp=cfg.c_fp,
            )
            reg = combined_routing_loss(
                q=q_route, expert_mask=expert_mask_route,
                group_ids=group_ids,
                group_prior_table=group_prior_table,
                group_id_to_row=group_id_to_row,
                cfg=mcfg, d=d,
            )
            pen = al.penalty(defer_rate, exp_tier)
            loss = (
                exp_clin
                + cfg.gamma_tier * exp_tier
                + pen
                + mcfg.w_gsdp * reg["gsdp"]
                + mcfg.w_rank_js * reg["rank_js"]
            )

            opt.zero_grad()
            loss.backward()
            if batch_idx % 50 == 0:
                diagnostic_checks(router, aux, q_route, expert_mask_route)
            opt.step()

            total_loss  += loss.item() * B
            defer_sum   += defer_rate.detach().item() * B
            tier_sum    += exp_tier.detach().item() * B
            gsdp_sum    += reg["gsdp"].item() * B
            rank_js_sum += reg["rank_js"].item() * B
            n_epoch     += B

        defer_mean = torch.tensor(defer_sum / max(n_epoch, 1), device=device)
        tier_mean  = torch.tensor(tier_sum  / max(n_epoch, 1), device=device)
        al.update(defer_mean, tier_mean)
        tau_now = torch.exp(router.log_tau).item()

        print(
            f"[Ep {ep:03d}] loss={total_loss / max(n_epoch, 1):.4f} "
            f"defer={defer_mean.item():.3f} tier={tier_mean.item():.4f} "
            f"gsdp={gsdp_sum / max(n_epoch, 1):.4f} "
            f"rank_js={rank_js_sum / max(n_epoch, 1):.4f} "
            f"tau={tau_now:.3f} lam_def={al.lam_def.item():.3f}"
        )

        if dl_va is not None:
            val_results = evaluate(
                router, dl_va, tier_costs, cfg, mcfg,
                group_prior_table, group_id_to_row, al_cfg, device, tag="VAL",
            )
            score = val_results["es_base"]
            if score < best_score - cfg.min_delta:
                if ep > cfg.warmup_epochs:
                    best_score = score
                    best_epoch = ep
                    patience_ctr = 0
                    best_state = copy.deepcopy(router.state_dict())
            elif ep > cfg.warmup_epochs:
                patience_ctr += 1
                if patience_ctr >= cfg.patience:
                    print(
                        f"Early stopping at epoch {ep}, best epoch {best_epoch}, "
                        f"score={best_score:.4f}"
                    )
                    router.load_state_dict(best_state)
                    break

    if dl_va is not None and best_state is not None:
        router.load_state_dict(best_state)
        print(f"Loaded best model from epoch {best_epoch}, "
              f"score={best_score:.4f}")

    return router, priors, all_q, all_pi


__all__ = ["train_l2d_multi_expert"]
