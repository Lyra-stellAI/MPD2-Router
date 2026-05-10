"""
Launch a constraint-aware Bayesian HPO study over the MPD2-Router knobs.

Loads the grouped dataset, wires in the package's training / evaluation
classes, and persists trial-level metadata in a JSONL ledger plus an Optuna
SQLite store.

Example
-------
.. code-block:: bash

    python scripts/run_hpo.py \
        --csv grouping_output/data.csv \
        --n_trials 80 \
        --storage sqlite:///hpo.db \
        --ledger hpo_ledger.jsonl
"""

from __future__ import annotations

import argparse

import pandas as pd

from m2p.adaptive_hpo import print_study_summary, run_hpo
from m2p.router_training import (
    AugLag, L2DDataset, Router, TIER_COST,
    action_costs, build_all_priors, combined_routing_loss,
    evaluate, l2d_objective,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",      required=True)
    ap.add_argument("--n_trials", type=int, default=80)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--storage",  default=None,
                    help="Optuna storage URL, e.g. sqlite:///hpo.db")
    ap.add_argument("--ledger",   default="hpo_ledger.jsonl")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Derive expert columns + availability mask from the loaded manifest.
    expert_cols = [c for c in df.columns if c.startswith("y_") and c != "y_true"]
    experts     = [c.removeprefix("y_") for c in expert_cols]
    df["m_experts"] = df.apply(
        lambda row: [int(not pd.isna(row[c])) for c in expert_cols], axis=1,
    )
    df["m_actions"] = df["m_experts"].apply(lambda m: [1] + m)

    best_params, study = run_hpo(
        df=df,
        expert_cols=expert_cols,
        experts=experts,
        tier_cost_dict=TIER_COST,
        n_experts=len(expert_cols),
        action_costs_fn=action_costs,
        build_all_priors_fn=build_all_priors,
        L2DDataset_cls=L2DDataset,
        Router_cls=Router,
        l2d_objective_fn=l2d_objective,
        combined_routing_loss_fn=combined_routing_loss,
        AugLag_cls=AugLag,
        evaluate_fn=evaluate,
        n_trials=args.n_trials,
        seed=args.seed,
        storage=args.storage,
        ledger_path=args.ledger,
    )

    print_study_summary(study)
    print(f"\nBest trial parameters:\n{best_params}")


if __name__ == "__main__":
    main()
