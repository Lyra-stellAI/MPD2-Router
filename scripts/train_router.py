"""
Train the MPD2-Router on a pre-grouped CSV using either a YAML config
(``configs/best_hpo.yaml``) or the package defaults.

Outputs the best checkpoint and a JSON summary of the final test metrics.

Example
-------
.. code-block:: bash

    python scripts/train_router.py \
        --csv grouping_output/final_dataset3_grouped.csv \
        --config configs/best_hpo.yaml \
        --out_dir runs/mpd2-router-001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from m2p.configs import ALConfig, PriorRegConfig, TrainConfig
from m2p.costs import TIER_COST, action_costs
from m2p.data import L2DDataset
from m2p.evaluation import (
    evaluate, print_eval_report, print_eval_report_dataset,
)
from m2p.training import train_l2d_multi_expert


def _load_yaml_config(path: str | None):
    if path is None:
        return TrainConfig(), PriorRegConfig(), ALConfig()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Install pyyaml to load YAML configs: pip install pyyaml"
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return (
        TrainConfig(**cfg.get("train", {})),
        PriorRegConfig(**cfg.get("prior", {})),
        ALConfig(**cfg.get("al", {})),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",     required=True)
    ap.add_argument("--config",  default=None)
    ap.add_argument("--out_dir", default="runs/mpd2-router")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    train_cfg, mcfg, al_cfg = _load_yaml_config(args.config)

    router, priors, _, _ = train_l2d_multi_expert(
        df=df, cfg=train_cfg, mcfg=mcfg, al_cfg=al_cfg,
    )

    # ---- Final test pass ----
    device = torch.device(train_cfg.device)
    df_test = df[df["split"].astype(str) == "test"].reset_index(drop=True)
    if len(df_test) == 0:
        print("No test split found — skipping final evaluation.")
        torch.save(router.state_dict(), out_dir / "router.pt")
        return

    dl_test = DataLoader(L2DDataset(df_test), batch_size=64, shuffle=False)
    experts = dl_test.dataset.expert_names
    tier_costs_t = torch.tensor(
        action_costs(TIER_COST, experts), dtype=torch.float32, device=device,
    )
    test_results = evaluate(
        router, dl_test, tier_costs_t, train_cfg, mcfg,
        priors["final_group_table"].to(device),
        priors["final_id_to_row"], al_cfg, device, tag="TEST",
    )

    print_eval_report(test_results, dl=dl_test)
    print_eval_report_dataset(test_results, dl=dl_test)

    torch.save(router.state_dict(), out_dir / "router.pt")
    summary = {k: v for k, v in test_results.items()
               if isinstance(v, (int, float, str)) and k != "tag"}
    summary["tag"] = test_results["tag"]
    (out_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir/'router.pt'} and {out_dir/'test_metrics.json'}")


if __name__ == "__main__":
    main()
