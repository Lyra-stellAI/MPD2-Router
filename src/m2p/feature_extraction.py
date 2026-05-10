"""
Frozen Swin-V2 feature extraction for OOD scoring and routing.

Wraps ``pamixsun/swinv2_tiny_for_glaucoma_classification`` (HuggingFace) and
returns logits, pooled last-layer embeddings, and intermediate hidden states
at user-specified depths.

Used by

* ``m2p.ood`` to fit OOD detectors on REFUGE-train embeddings;
* the router's ``risk_enc`` and ``ai_enc`` branches as side information.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


SWINV2_HF_NAME = "pamixsun/swinv2_tiny_for_glaucoma_classification"


def load_swinv2(device: torch.device | str | None = None):
    """Load the frozen Swin-V2 classifier and its image processor."""
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    processor = AutoImageProcessor.from_pretrained(SWINV2_HF_NAME)
    model = AutoModelForImageClassification.from_pretrained(SWINV2_HF_NAME)
    model.eval().to(device)
    return model, processor, device


@torch.inference_mode()
def extract_features(
    model,
    processor,
    df: pd.DataFrame,
    *,
    batch_size: int = 32,
    layers: Iterable[int] = (2, 3, 4),
    path_col: str = "image_path",
    label_col: str = "y_true",
    id_col: str = "case_id",
    global_id_col: str = "global_id",
) -> Dict[str, object]:
    """
    Extract logits + pooled hidden states for every image referenced by *df*.

    Parameters
    ----------
    model, processor : torch.nn.Module, AutoImageProcessor
        Output of :func:`load_swinv2`.
    df : DataFrame
        Must carry ``path_col``, ``label_col``, ``id_col``, ``global_id_col``.
        Rows are de-duplicated on ``path_col``.
    batch_size, layers : int, sequence of int
        Mini-batch size and which ``hidden_states`` indices to mean-pool.
    """
    model.eval()
    device = next(model.parameters()).device

    u = (
        df.drop_duplicates(subset=path_col)
        [[id_col, path_col, label_col, global_id_col]]
        .reset_index(drop=True)
    )
    paths = u[path_col].tolist()

    case_ids, labels, global_ids = [], [], []
    all_logits, all_last = [], []
    all_feats: Dict[int, list] = {int(l): [] for l in layers}

    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting"):
        batch_idx = list(range(i, min(i + batch_size, len(paths))))
        imgs, keep = [], []
        for j, p in enumerate([paths[k] for k in batch_idx]):
            try:
                imgs.append(Image.open(p).convert("RGB"))
                keep.append(j)
            except Exception:  # unreadable / missing → skip
                continue
        if not imgs:
            continue

        inputs = processor(images=imgs, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hs = outputs.hidden_states

        kept = [batch_idx[j] for j in keep]
        case_ids.extend(u.loc[kept, id_col].tolist())
        labels.extend(u.loc[kept, label_col].astype(int).tolist())
        global_ids.extend(u.loc[kept, global_id_col].tolist())
        all_logits.append(outputs.logits.cpu())
        all_last.append(hs[-1].mean(dim=1).cpu())
        for l in layers:
            all_feats[int(l)].append(hs[int(l)].mean(dim=1).cpu())

    if not all_logits:
        raise RuntimeError("No images were successfully processed.")

    return {
        "global_id":   np.array(global_ids),
        "case_id":     np.array(case_ids),
        "labels":      torch.tensor(labels, dtype=torch.long),
        "logits":      torch.cat(all_logits),
        "pooled_last": torch.cat(all_last),
        "hidden":      {int(l): torch.cat(all_feats[int(l)]) for l in layers},
    }


__all__ = ["SWINV2_HF_NAME", "load_swinv2", "extract_features"]
