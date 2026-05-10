"""
Extract Swin-V2 logits, pooled embeddings, and intermediate hidden states for
each row of an image-manifest CSV.

Saves a single ``.pt`` payload per dataset and an aggregated ``.npy`` of pooled
last-layer embeddings keyed by ``global_id`` for downstream stages.

Example
-------
.. code-block:: bash

    python scripts/extract_features.py \
        --csv data/manifests/refuge.csv \
        --out_dir data/features/refuge \
        --path_col image_path --label_col y_true \
        --id_col case_id --global_id_col global_id
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from m2p.feature_extraction import extract_features, load_swinv2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",            required=True)
    ap.add_argument("--out_dir",        required=True)
    ap.add_argument("--path_col",       default="image_path")
    ap.add_argument("--label_col",      default="y_true")
    ap.add_argument("--id_col",         default="case_id")
    ap.add_argument("--global_id_col",  default="global_id")
    ap.add_argument("--batch_size",     type=int, default=32)
    ap.add_argument("--layers",         type=int, nargs="+", default=[2, 3, 4])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    model, processor, _ = load_swinv2()

    feat = extract_features(
        model, processor, df,
        batch_size=args.batch_size,
        layers=tuple(args.layers),
        path_col=args.path_col,
        label_col=args.label_col,
        id_col=args.id_col,
        global_id_col=args.global_id_col,
    )

    torch.save(
        {
            "global_id":   feat["global_id"].tolist(),
            "case_id":     feat["case_id"].tolist(),
            "labels":      feat["labels"],
            "logits":      feat["logits"],
            "pooled_last": feat["pooled_last"],
            "hidden":      feat["hidden"],
        },
        out_dir / "features.pt",
    )
    np.save(out_dir / "pooled_last.npy", feat["pooled_last"].numpy())
    np.save(out_dir / "global_ids.npy",  feat["global_id"])

    print(f"Wrote features for {len(feat['global_id'])} samples → {out_dir}")


if __name__ == "__main__":
    main()
