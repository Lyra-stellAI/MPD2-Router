"""
Evaluation pass and reporting helpers for the MPD2-Router.

* :func:`evaluate` — full pass over a held-out loader; returns a dict with
  per-action / per-dataset breakdowns and the early-stopping score.
* :func:`print_eval_report` — global confusion + per-action table.
* :func:`print_eval_report_dataset` — same broken down by source dataset.
* :func:`diagnostic_checks` — gradient/utilisation telemetry callable mid-loop.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef

from .configs import ALConfig, PriorRegConfig, TrainConfig
from .losses import combined_routing_loss


# ─────────────────────────────────────────────────────────────────────
# Evaluation pass
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(router, dl, tier_costs: torch.Tensor,
             cfg: TrainConfig, mcfg: PriorRegConfig,
             group_prior_table: torch.Tensor,
             group_id_to_row: Dict[int, int],
             al_cfg: ALConfig,
             device: torch.device,
             tag: str = "VAL") -> dict:
    router.eval()
    n_experts = len(dl.dataset.expert_cols)
    num_actions = 1 + n_experts

    N = 0
    sum_acc = sum_clin = 0.0
    sum_tier_hard = sum_tier_soft = 0.0
    sum_defer_hard = sum_defer_soft = 0.0
    sum_gsdp = sum_rank_js = 0.0

    action_count     = torch.zeros(num_actions)
    action_correct   = torch.zeros(num_actions)
    action_fn        = torch.zeros(num_actions)
    action_fp        = torch.zeros(num_actions)
    action_tp        = torch.zeros(num_actions)
    action_tn        = torch.zeros(num_actions)
    action_soft_mass = torch.zeros(num_actions)

    all_preds, all_trues, all_probs, all_actions = [], [], [], []
    all_ds, all_d = [], []

    for (vim, q, u, p_ai, vcdr, acdr, logit_0, logit_1, y_true, y_exp,
         m_actions, expert_mask, ds_ids, group_ids) in dl:

        vim, q, u = vim.to(device), q.to(device), u.to(device)
        vcdr, acdr = vcdr.to(device), acdr.to(device)
        p_ai = p_ai.to(device)
        logit_0, logit_1 = logit_0.to(device), logit_1.to(device)
        y_true, y_exp = y_true.to(device), y_exp.to(device)
        m_actions = m_actions.to(device)
        expert_mask = expert_mask.to(device)
        ds_ids, group_ids = ds_ids.to(device), group_ids.to(device)
        B = y_true.shape[0]

        pi, _, aux = router(
            vim, q, u, vcdr, acdr, p_ai, logit_0, logit_1, m_actions,
        )
        pi = pi * m_actions
        pi = pi / (pi.sum(dim=1, keepdim=True) + 1e-8)

        d = aux["d"].view(-1)
        q_route = aux["q"].view(B, n_experts)
        expert_mask_route = m_actions[:, 1:].to(q_route.dtype)

        all_d.append(d.cpu())
        all_ds.append(ds_ids.cpu())

        reg = combined_routing_loss(
            q=q_route, expert_mask=expert_mask_route,
            group_ids=group_ids,
            group_prior_table=group_prior_table,
            group_id_to_row=group_id_to_row,
            cfg=mcfg, d=d,
        )
        sum_gsdp    += reg["gsdp"].item() * B
        sum_rank_js += reg["rank_js"].item() * B

        defer_soft = 1.0 - pi[:, 0].mean()
        tier_soft  = (pi * tier_costs.view(1, -1)).sum(dim=1).mean()
        action_soft_mass += pi.sum(dim=0).cpu()

        a = torch.argmax(pi, dim=1)
        defer_hard = (a > 0).float().mean()
        tier_hard  = tier_costs[a].mean()

        y_true01 = (y_true > 0.5).long()
        pred = torch.empty_like(y_true01)
        pred_ai = (p_ai >= 0.5).long()
        y_exp_filled = torch.nan_to_num(y_exp, nan=0.0)

        ai_mask = (a == 0)
        ex_mask = (a > 0)
        pred[ai_mask] = pred_ai[ai_mask]
        if ex_mask.any():
            j = a[ex_mask] - 1
            pred[ex_mask] = y_exp_filled[ex_mask, j].long()

        # Soft probability for AUPRC: policy-weighted mixture
        p_pos = pi[:, 0] * p_ai + (pi[:, 1:] * y_exp_filled).sum(dim=1)
        all_probs.append(p_pos.cpu())

        acc = (pred == y_true01).float().mean()
        fn = (y_true01 == 1) & (pred == 0)
        fp = (y_true01 == 0) & (pred == 1)
        clin = (cfg.c_fn * fn.float() + cfg.c_fp * fp.float()).mean()

        for k in range(num_actions):
            mask_k = (a == k)
            count_k = mask_k.sum().item()
            action_count[k] += count_k
            if count_k > 0:
                action_correct[k] += (pred[mask_k] == y_true01[mask_k]).sum().item()
                action_fn[k] += ((y_true01[mask_k] == 1) & (pred[mask_k] == 0)).sum().item()
                action_fp[k] += ((y_true01[mask_k] == 0) & (pred[mask_k] == 1)).sum().item()
                action_tp[k] += ((y_true01[mask_k] == 1) & (pred[mask_k] == 1)).sum().item()
                action_tn[k] += ((y_true01[mask_k] == 0) & (pred[mask_k] == 0)).sum().item()

        all_preds.append(pred.cpu())
        all_trues.append(y_true01.cpu())
        all_actions.append(a.cpu())

        sum_acc        += acc.item() * B
        sum_clin       += clin.item() * B
        sum_tier_hard  += tier_hard.item() * B
        sum_tier_soft  += tier_soft.item() * B
        sum_defer_hard += defer_hard.item() * B
        sum_defer_soft += defer_soft.item() * B
        N += B

    all_d_cat   = torch.cat(all_d)
    all_preds_c = torch.cat(all_preds)
    all_trues_c = torch.cat(all_trues)
    all_probs_c = torch.cat(all_probs)

    q_mass = action_soft_mass[1:] / max(action_soft_mass[1:].sum().item(), 1e-8)
    top1_share = q_mass.max().item()
    top1_idx   = q_mass.argmax().item()

    y_np    = all_trues_c.numpy()
    p_np    = all_probs_c.numpy()
    pred_np = all_preds_c.numpy()
    auprc = (
        float(average_precision_score(y_np, p_np))
        if len(np.unique(y_np)) > 1 else float("nan")
    )
    mcc = float(matthews_corrcoef(y_np, pred_np))

    inv_expert_map = dl.dataset.expert_id_to_name
    action_names = ["AI"] + [
        inv_expert_map[i] for i in range(len(inv_expert_map))
    ]

    results = {
        "acc":              sum_acc / N,
        "clinical":         sum_clin / N,
        "tier_hard":        sum_tier_hard / N,
        "tier_soft":        sum_tier_soft / N,
        "defer_hard":       sum_defer_hard / N,
        "defer_soft":       sum_defer_soft / N,
        "gsdp":             sum_gsdp / N,
        "rank_js":          sum_rank_js / N,
        "auprc":            auprc,
        "mcc":              mcc,
        "N":                N,
        "tag":              tag,
        "action_count":     action_count,
        "action_correct":   action_correct,
        "action_fn":        action_fn,
        "action_fp":        action_fp,
        "action_tp":        action_tp,
        "action_tn":        action_tn,
        "action_soft_mass": action_soft_mass,
        "all_preds":        all_preds_c,
        "all_trues":        all_trues_c,
        "all_probs":        all_probs_c,
        "all_actions":      torch.cat(all_actions),
        "d_mean":           all_d_cat.mean().item(),
        "d_std":            all_d_cat.std().item(),
        "q_mass":           q_mass,
        "top1_expert":      top1_idx,
        "top1_share":       top1_share,
        "all_ds":           torch.cat(all_ds),
        "action_names":     action_names,
        "inv_expert_map":   inv_expert_map,
    }

    gamma_eval = getattr(cfg, "gamma_tier", 1.0)
    dmax = getattr(al_cfg, "max_deferral_rate", None)
    es_base = results["clinical"] + gamma_eval * results["tier_soft"]
    es_violation = (
        max(0.0, results["defer_soft"] - dmax) if dmax is not None else 0.0
    )
    results["es_base"]      = es_base
    results["es_violation"] = es_violation
    results["total_cost"]   = float(
        results["clinical"] + gamma_eval * results["tier_hard"]
    )

    print(
        f"[{tag}] acc={results['acc']:.4f} clin={results['clinical']:.4f} "
        f"auprc={auprc:.4f} mcc={mcc:.4f} "
        f"total_cost={results['total_cost']:.4f} "
        f"tier_h={results['tier_hard']:.4f} "
        f"def_h={results['defer_hard']:.4f} def_s={results['defer_soft']:.4f} "
        f"score={es_base:.4f} viol={es_violation:.4f} "
        f"d={results['d_mean']:.3f}±{results['d_std']:.3f} "
        f"top1=exp{top1_idx}({top1_share:.1%}) "
        f"gsdp={results['gsdp']:.4f} rank_js={results['rank_js']:.4f}"
    )
    return results


# ─────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────

def diagnostic_checks(router, aux, q_route, expert_mask_route) -> None:
    """Lightweight sanity probe; call after ``loss.backward()``."""
    with torch.no_grad():
        expert_grad_norm = sum(
            p.grad.norm().item()
            for p in router.expert_head.parameters() if p.grad is not None
        )
        defer_grad_norm = sum(
            p.grad.norm().item()
            for p in router.defer_head.parameters() if p.grad is not None
        )
        ratio = expert_grad_norm / max(defer_grad_norm, 1e-8)
        print(f"  [diag] grad norms — expert: {expert_grad_norm:.4f}, "
              f"defer: {defer_grad_norm:.4f}, ratio: {ratio:.2f}")

        gate_probs = aux["gate_probs"]
        avg_open = (gate_probs > 0.5).float().sum(dim=-1).mean().item()
        avg_available = expert_mask_route.sum(dim=-1).mean().item()
        select_ratio = avg_open / max(avg_available, 1e-8)
        print(f"  [diag] avg experts selected: {avg_open:.1f} / "
              f"{avg_available:.1f} ({select_ratio:.1%})")

        q_mass = (q_route * expert_mask_route).sum(dim=0)
        q_mass = q_mass / q_mass.sum().clamp(min=1e-8)
        sorted_mass, _ = q_mass.sort()
        n = len(sorted_mass)
        idx = torch.arange(1, n + 1, dtype=torch.float32, device=sorted_mass.device)
        gini = (
            2 * (idx * sorted_mass).sum() / (n * sorted_mass.sum())
            - (n + 1) / n
        ).item()
        print(f"  [diag] q_mass Gini: {gini:.3f}, "
              f"top-3: {sorted_mass[-3:].tolist()}")


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────

def print_eval_report(results: dict, dl=None) -> None:
    """Global metrics + per-action table."""
    tag = results["tag"]
    N = results["N"]

    if dl is not None:
        action_names = dl.dataset.action_names
    else:
        action_names = results.get(
            "action_names",
            ["AI"] + [
                f"Exp{k}"
                for k in range(1, int(results["all_actions"].max().item()) + 1)
            ],
        )
    num_actions = len(action_names)

    all_preds = results["all_preds"]
    all_trues = results["all_trues"]
    all_probs = results.get("all_probs", None)

    global_tp = ((all_trues == 1) & (all_preds == 1)).sum().item()
    global_fn = ((all_trues == 1) & (all_preds == 0)).sum().item()
    global_fp = ((all_trues == 0) & (all_preds == 1)).sum().item()
    global_tn = ((all_trues == 0) & (all_preds == 0)).sum().item()

    precision = global_tp / max(global_tp + global_fp, 1)
    recall    = global_tp / max(global_tp + global_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    specificity = global_tn / max(global_tn + global_fp, 1)

    trues_np = all_trues.cpu().numpy()
    preds_np = all_preds.cpu().numpy()
    mcc = matthews_corrcoef(trues_np, preds_np)
    auprc = (
        average_precision_score(trues_np, all_probs.cpu().numpy())
        if all_probs is not None and len(np.unique(trues_np)) > 1
        else float("nan")
    )

    score_val = results["es_base"] if results["es_base"] is not None else results["total_cost"]
    print(f"\n{'='*70}")
    print(f"[{tag}] FULL EVALUATION REPORT  (N={N})")
    print(f"{'='*70}")
    print(f"\n--- Global Metrics ---")
    print(f"  Accuracy:      {results['acc']:.4f}")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  F1:            {f1:.4f}")
    print(f"  Specificity:   {specificity:.4f}")
    print(f"  MCC:           {mcc:.4f}")
    print(f"  AUPRC:         {auprc:.4f}")
    print(f"  Clinical cost: {results['clinical']:.4f}  "
          f"(FN={global_fn}, FP={global_fp})")

    print(f"\n--- Deferral & Cost ---")
    print(f"  Defer (hard): {results['defer_hard']:.4f}   "
          f"Defer (soft): {results['defer_soft']:.4f}")
    print(f"  Tier  (hard): {results['tier_hard']:.4f}   "
          f"Tier  (soft): {results['tier_soft']:.4f}")
    print(f"  Total score:  {score_val:.4f}")

    print(f"\n--- Global Confusion Matrix ---")
    print(f"               Pred=0    Pred=1")
    print(f"  True=0 (TN)  {global_tn:>6}    (FP) {global_fp:>6}")
    print(f"  True=1 (FN)  {global_fn:>6}    (TP) {global_tp:>6}")

    action_count     = results["action_count"]
    action_correct   = results["action_correct"]
    action_tp_a      = results["action_tp"]
    action_fn_a      = results["action_fn"]
    action_fp_a      = results["action_fp"]
    action_soft_mass = results["action_soft_mass"]

    print(f"\n--- Per-Action Breakdown ---")
    print(f"  {'Action':<12} {'Count':>6} {'%Routed':>8} {'Acc':>7} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'FN':>5} {'FP':>5} {'SoftMass':>9}")
    print(f"  {'-'*80}")

    for k in range(num_actions):
        cnt = int(action_count[k].item())
        pct = 100.0 * cnt / N if N > 0 else 0.0

        if cnt > 0:
            acc_k  = action_correct[k].item() / cnt
            tp_k   = action_tp_a[k].item()
            fn_k   = action_fn_a[k].item()
            fp_k   = action_fp_a[k].item()
            prec_k = tp_k / (tp_k + fp_k + 1e-8)
            rec_k  = tp_k / (tp_k + fn_k + 1e-8)
            f1_k   = 2 * prec_k * rec_k / (prec_k + rec_k + 1e-8)
        else:
            acc_k = prec_k = rec_k = f1_k = 0.0
            fn_k = fp_k = 0

        soft_pct = 100.0 * action_soft_mass[k].item() / N if N > 0 else 0.0
        print(f"  {action_names[k]:<12} {cnt:>6} {pct:>7.1f}% "
              f"{acc_k:>7.4f} {prec_k:>7.4f} {rec_k:>7.4f} {f1_k:>7.4f} "
              f"{int(fn_k):>5} {int(fp_k):>5} {soft_pct:>8.1f}%")

    print(f"{'='*70}\n")


def print_eval_report_dataset(results: dict, dl=None) -> None:
    """Same metrics broken down per source dataset (REFUGE/ORIGA/CHAKSU)."""
    all_targets = results["all_trues"]
    all_preds   = results["all_preds"]
    all_actions = results["all_actions"]
    all_probs   = results["all_probs"]
    all_ds      = results["all_ds"]

    if "action_names" in results:
        action_names = results["action_names"]
    elif dl is not None:
        action_names = dl.dataset.action_names
    else:
        num_act = int(all_actions.max().item()) + 1
        action_names = ["AI"] + [f"Exp{k}" for k in range(1, num_act)]
    num_actions = len(action_names)

    if dl is not None:
        inv_ds_map = {v: k for k, v in dl.dataset.ds_map.items()}
    else:
        inv_ds_map = {i: f"ds_{i}" for i in sorted(all_ds.unique().tolist())}

    for ds_idx, ds_name in inv_ds_map.items():
        mask_ds = (all_ds == ds_idx)
        N_ds = int(mask_ds.sum().item())
        if N_ds == 0:
            print(f"\n{'='*70}\n  {ds_name}  (N=0, skipped)")
            continue

        t_ds = all_targets[mask_ds]
        p_ds = all_preds[mask_ds]
        a_ds = all_actions[mask_ds]
        probs_ds = all_probs[mask_ds]

        acc_ds   = (p_ds == t_ds).float().mean().item()
        fn_ds    = ((t_ds == 1) & (p_ds == 0)).sum().item()
        fp_ds    = ((t_ds == 0) & (p_ds == 1)).sum().item()
        tp_ds    = ((t_ds == 1) & (p_ds == 1)).sum().item()
        tn_ds    = ((t_ds == 0) & (p_ds == 0)).sum().item()
        defer_ds = (a_ds > 0).float().mean().item()
        mcc_ds   = matthews_corrcoef(t_ds.cpu().numpy(), p_ds.cpu().numpy())
        auprc_ds = (
            average_precision_score(t_ds.cpu().numpy(), probs_ds.cpu().numpy())
            if len(t_ds.unique()) > 1 else float("nan")
        )

        print(f"\n{'='*70}")
        print(f"  {ds_name}  (N={N_ds}, acc={acc_ds:.4f}, MCC={mcc_ds:.4f}, "
              f"AUPRC={auprc_ds:.4f}, defer={defer_ds:.3f}, "
              f"FN={int(fn_ds)}, FP={int(fp_ds)}, "
              f"TP={int(tp_ds)}, TN={int(tn_ds)})")
        print(f"{'='*70}")

        print(f"  {'Action':<25} {'Count':>6} {'Share':>7} "
              f"{'Acc':>7} {'FN':>5} {'FP':>5} "
              f"{'TP':>5} {'TN':>5} {'Sens':>7} {'Spec':>7}")
        print(f"  {'-'*85}")

        for k in range(num_actions):
            mask_k = (a_ds == k)
            count_k = int(mask_k.sum().item())
            if count_k == 0:
                print(f"  {action_names[k]:<25} {0:>6} {0:>7.1%}")
                continue
            tk = t_ds[mask_k]
            pk = p_ds[mask_k]
            acc_k  = (pk == tk).float().mean().item()
            fn_k   = ((tk == 1) & (pk == 0)).sum().item()
            fp_k   = ((tk == 0) & (pk == 1)).sum().item()
            tp_k   = ((tk == 1) & (pk == 1)).sum().item()
            tn_k   = ((tk == 0) & (pk == 0)).sum().item()
            sens_k = tp_k / max(tp_k + fn_k, 1)
            spec_k = tn_k / max(tn_k + fp_k, 1)
            share_k = count_k / N_ds
            print(f"  {action_names[k]:<25} {count_k:>6} {share_k:>7.1%} "
                  f"{acc_k:>7.3f} {fn_k:>5} {fp_k:>5} "
                  f"{tp_k:>5} {tn_k:>5} {sens_k:>7.3f} {spec_k:>7.3f}")


__all__ = [
    "evaluate", "diagnostic_checks",
    "print_eval_report", "print_eval_report_dataset",
]
