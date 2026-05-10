"""
Hybrid similarity bucketing for MPD2-Router (leak-free v3).

Three-stage grouping pipeline:

  Stage 1  Build "support families" by clustering ALL unique expert-availability
           masks via Hamming distance.  Expert masks are fixed metadata
           determined by the annotation protocol, not learned representations,
           so using the full dataset introduces no leakage.

  Stage 2  Inside each family, run KMeans on Swin-V2 embeddings, **fit on the
           train split only**, then assign held-out samples to the nearest
           train centroid.  K is chosen by silhouette*log2(K) with hard
           per-cluster minimum-train constraints.

  Stage 3  Fold any group whose train count falls below ``min_group_train`` OR
           whose train fraction falls below ``min_train_frac`` into the largest
           healthy sibling within the same family.  The fraction check catches
           pathological cases like 38 train / 513 total (= 7.4%) that pass the
           absolute count threshold.

The CSV is expected to have a ``split`` column with values in
``{"train", "val", "test"}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────────────
# Mask utilities
# ─────────────────────────────────────────────────────────────────────

def parse_mask(x):
    if isinstance(x, np.ndarray):
        arr = x.astype(float).ravel()
    elif isinstance(x, (list, tuple)):
        arr = np.asarray(x, dtype=float).ravel()
    elif isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        s = s.replace(",", " ")
        arr = np.asarray([float(t) for t in s.split() if t], dtype=float)
    else:
        raise ValueError(f"Unsupported mask format: {type(x)}")
    return (arr > 0.5).astype(int)


def drop_ai_slot(mask_arr):
    if len(mask_arr) == 13:
        return mask_arr[1:]
    return mask_arr


def mask_to_str(mask_arr):
    return "".join(str(int(v)) for v in mask_arr)


def find_dataset_col(df):
    for c in ("dataset", "ds", "source"):
        if c in df.columns:
            return c
    return None


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Support families  (global mask clustering)
# ─────────────────────────────────────────────────────────────────────

def _agglomerative_cluster(unique_masks, distance_threshold):
    D = pairwise_distances(unique_masks, metric="hamming")
    try:
        agg = AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        agg = AgglomerativeClustering(
            n_clusters=None, affinity="precomputed", linkage="average",
            distance_threshold=distance_threshold,
        )
    return agg.fit_predict(D)


def build_support_families(df, mask_col="m_actions",
                           distance_threshold=0.40, min_family_size=10):
    expert_masks = df[mask_col].apply(lambda x: drop_ai_slot(parse_mask(x)))
    exact_mask_str = expert_masks.apply(mask_to_str)

    unique_mask_df = (
        pd.DataFrame({"exact_mask": exact_mask_str})
        .value_counts()
        .reset_index(name="n")
        .sort_values(["n", "exact_mask"], ascending=[False, True])
        .reset_index(drop=True)
    )
    unique_masks = np.stack(
        [np.array(list(s), dtype=int) for s in unique_mask_df["exact_mask"]]
    )

    print(f"  Total unique masks: {len(unique_mask_df)}")

    fam_labels = _agglomerative_cluster(unique_masks, distance_threshold)
    unique_mask_df["support_family"] = fam_labels

    fam_sizes = (
        unique_mask_df.groupby("support_family", as_index=False)["n"]
        .sum().sort_values("n", ascending=False).reset_index(drop=True)
    )
    remap = {old: new for new, old in enumerate(fam_sizes["support_family"])}
    unique_mask_df["support_family"] = unique_mask_df["support_family"].map(remap)

    if min_family_size > 0:
        fam_sample_sizes = unique_mask_df.groupby("support_family")["n"].sum()
        small_fams = set(fam_sample_sizes[fam_sample_sizes < min_family_size].index)

        if small_fams:
            rep_masks = {}
            for fam_id, grp in unique_mask_df.groupby("support_family"):
                top = grp.sort_values("n", ascending=False).iloc[0]
                rep_masks[fam_id] = np.array(list(top["exact_mask"]), dtype=int)

            large_fams = [f for f in rep_masks if f not in small_fams]
            large_reps = np.stack([rep_masks[f] for f in large_fams])

            n_absorbed = 0
            for sf in small_fams:
                sf_rep = rep_masks[sf].reshape(1, -1)
                dists = pairwise_distances(sf_rep, large_reps, metric="hamming")[0]
                nearest = large_fams[np.argmin(dists)]
                unique_mask_df.loc[
                    unique_mask_df["support_family"] == sf, "support_family"
                ] = nearest
                n_absorbed += 1

            old_ids = sorted(unique_mask_df["support_family"].unique())
            compact = {old: new for new, old in enumerate(old_ids)}
            unique_mask_df["support_family"] = (
                unique_mask_df["support_family"].map(compact)
            )

            if n_absorbed > 0:
                print(f"  Absorbed {n_absorbed} micro-families "
                      f"(< {min_family_size} samples)")

    family_reps = {}
    for fam_id, grp in unique_mask_df.groupby("support_family"):
        top = grp.sort_values("n", ascending=False).iloc[0]
        family_reps[fam_id] = np.array(list(top["exact_mask"]), dtype=int)

    mask_to_family = dict(
        zip(unique_mask_df["exact_mask"], unique_mask_df["support_family"])
    )

    df = df.copy()
    df["exact_mask"] = exact_mask_str.values
    df["support_family"] = df["exact_mask"].map(mask_to_family).astype(int)

    all_vecs = np.stack(
        [np.array(list(s), dtype=int) for s in df["exact_mask"].values]
    )
    assigned_reps = np.stack(
        [family_reps[f] for f in df["support_family"].values]
    )
    df["hamming_to_family_rep"] = (
        (all_vecs != assigned_reps).sum(axis=1) / all_vecs.shape[1]
    )

    return df, unique_mask_df, family_reps


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — KMeans sub-clustering  (fit on train, predict held-out)
# ─────────────────────────────────────────────────────────────────────

def choose_k_and_fit(X_norm_train, max_k=12, min_size_frac=0.05,
                     min_size_abs=10, min_train_per_cluster=20,
                     random_state=42):
    n = len(X_norm_train)
    min_cluster_size = max(min_size_abs, int(np.ceil(min_size_frac * n)))
    effective_min = max(min_cluster_size, min_train_per_cluster)

    if n < 2 * effective_min:
        return None, 1, np.nan, effective_min

    k_upper = min(max_k, max(2, int(np.sqrt(n / 2))))
    if k_upper < 2:
        return None, 1, np.nan, effective_min

    best_k, best_score = 1, -np.inf
    best_km, best_sil = None, np.nan

    for k in range(2, k_upper + 1):
        if k * effective_min > n:
            break

        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_norm_train)
        sizes = np.bincount(labels)

        if sizes.min() < effective_min:
            continue

        sil = silhouette_score(X_norm_train, labels, metric="cosine")
        composite = sil * np.log2(k)

        if composite > best_score:
            best_k, best_score = k, composite
            best_km, best_sil = km, sil

    return best_km, best_k, best_sil, effective_min


def _merge_starved_subclusters(km, labels_all, train_flags,
                               min_train_per_cluster):
    centroids = km.cluster_centers_
    unique_labels = np.unique(labels_all)

    train_counts = {}
    for c in unique_labels:
        train_counts[c] = int((train_flags & (labels_all == c)).sum())

    starved = {c for c, n in train_counts.items()
               if n < min_train_per_cluster}

    if not starved:
        return labels_all, {c: c for c in unique_labels}, 0

    healthy = sorted(set(unique_labels) - starved)
    if not healthy:
        return np.zeros_like(labels_all), {c: 0 for c in unique_labels}, len(starved)

    healthy_centroids = np.stack([centroids[c] for c in healthy])
    merge_map = {c: c for c in unique_labels}

    for sc in starved:
        sc_centroid = centroids[sc].reshape(1, -1)
        dists = pairwise_distances(sc_centroid, healthy_centroids,
                                   metric="cosine")[0]
        nearest = healthy[np.argmin(dists)]
        merge_map[sc] = nearest

    labels_merged = np.array([merge_map[l] for l in labels_all])

    remaining = sorted(set(labels_merged))
    compact = {old: new for new, old in enumerate(remaining)}
    labels_merged = np.array([compact[l] for l in labels_merged])
    final_map = {old: compact[merge_map[old]] for old in unique_labels}

    return labels_merged, final_map, len(starved)


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Small-group folding (count + train fraction)
# ─────────────────────────────────────────────────────────────────────

def fold_small_groups(df, train_mask, min_group_train=20,
                      min_train_frac=0.20):
    train_counts = (
        df[train_mask]
        .groupby(["support_family", "subcluster"]).size()
        .reset_index(name="n_train")
    )
    all_counts = (
        df.groupby(["support_family", "subcluster"]).size()
        .reset_index(name="n_all")
    )
    grp_info = all_counts.merge(
        train_counts, on=["support_family", "subcluster"], how="left"
    ).fillna({"n_train": 0})
    grp_info["n_train"] = grp_info["n_train"].astype(int)
    grp_info["train_frac"] = grp_info["n_train"] / grp_info["n_all"]

    fold_log: List[Dict] = []

    for fam_id in sorted(grp_info["support_family"].unique()):
        fam_rows = grp_info[grp_info["support_family"] == fam_id].copy()

        if len(fam_rows) <= 1:
            continue

        is_small = (
            (fam_rows["n_train"] < min_group_train) |
            (fam_rows["train_frac"] < min_train_frac)
        )
        small = fam_rows[is_small]
        large = fam_rows[~is_small]

        if small.empty:
            continue

        if large.empty:
            target_row = fam_rows.sort_values(
                ["train_frac", "n_train"], ascending=[False, False]
            ).iloc[0]
            target_sc = target_row["subcluster"]
            for _, row in small.iterrows():
                if row["subcluster"] == target_sc:
                    continue
                reason = []
                if row["n_train"] < min_group_train:
                    reason.append(f"n_train={int(row['n_train'])}<{min_group_train}")
                if row["train_frac"] < min_train_frac:
                    reason.append(f"train_frac={row['train_frac']:.3f}<{min_train_frac}")
                mask = ((df["support_family"] == fam_id) &
                        (df["subcluster"] == row["subcluster"]))
                fold_log.append({
                    "family": fam_id,
                    "from_subcluster": int(row["subcluster"]),
                    "to_subcluster": int(target_sc),
                    "n_train_folded": int(row["n_train"]),
                    "n_all_folded": int(row["n_all"]),
                    "train_frac": round(float(row["train_frac"]), 4),
                    "reason": "; ".join(reason) + " (all_small_merge)",
                })
                df.loc[mask, "subcluster"] = target_sc
        else:
            target_sc = large.sort_values(
                ["n_train"], ascending=False
            ).iloc[0]["subcluster"]
            for _, row in small.iterrows():
                reason = []
                if row["n_train"] < min_group_train:
                    reason.append(f"n_train={int(row['n_train'])}<{min_group_train}")
                if row["train_frac"] < min_train_frac:
                    reason.append(f"train_frac={row['train_frac']:.3f}<{min_train_frac}")
                mask = ((df["support_family"] == fam_id) &
                        (df["subcluster"] == row["subcluster"]))
                fold_log.append({
                    "family": fam_id,
                    "from_subcluster": int(row["subcluster"]),
                    "to_subcluster": int(target_sc),
                    "n_train_folded": int(row["n_train"]),
                    "n_all_folded": int(row["n_all"]),
                    "train_frac": round(float(row["train_frac"]), 4),
                    "reason": "; ".join(reason),
                })
                df.loc[mask, "subcluster"] = target_sc

    for fam_id in df["support_family"].unique():
        fam_mask = df["support_family"] == fam_id
        old_scs = sorted(df.loc[fam_mask, "subcluster"].unique())
        compact = {old: new for new, old in enumerate(old_scs)}
        df.loc[fam_mask, "subcluster"] = (
            df.loc[fam_mask, "subcluster"].map(compact)
        )

    group_keys = (
        df[["support_family", "subcluster"]]
        .drop_duplicates()
        .sort_values(["support_family", "subcluster"])
        .reset_index(drop=True)
    )
    group_keys["group_id"] = np.arange(len(group_keys))

    df.drop(columns=["group_id"], inplace=True, errors="ignore")
    df = df.merge(group_keys, on=["support_family", "subcluster"], how="left")

    return df, len(fold_log), fold_log


# ─────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────

def summarize_support_families(df, unique_mask_df):
    dataset_col = find_dataset_col(df)

    rep_mask = (
        df.groupby(["support_family", "exact_mask"]).size()
        .reset_index(name="n_mask")
        .sort_values(
            ["support_family", "n_mask", "exact_mask"],
            ascending=[True, False, True],
        )
        .drop_duplicates("support_family")
        .rename(columns={"exact_mask": "representative_mask"})
        [["support_family", "representative_mask"]]
    )

    n_exact = (
        df.groupby("support_family")["exact_mask"]
        .nunique().reset_index(name="n_exact_masks")
    )

    fam = (
        df.groupby("support_family", as_index=False)
        .agg(n=("y_true", "size"), glaucoma_rate=("y_true", "mean"))
        .merge(rep_mask, on="support_family", how="left")
        .merge(n_exact, on="support_family", how="left")
        .sort_values("support_family").reset_index(drop=True)
    )

    if dataset_col:
        mix = (
            df.groupby(["support_family", dataset_col]).size()
            .reset_index(name="cnt")
            .sort_values(
                ["support_family", "cnt", dataset_col],
                ascending=[True, False, True],
            )
        )
        mix_str = (
            mix.groupby("support_family")
            .apply(
                lambda g: ", ".join(
                    f"{r[dataset_col]}:{int(r['cnt'])}" for _, r in g.iterrows()
                )
            )
            .reset_index(name="dataset_mix")
        )
        fam = fam.merge(mix_str, on="support_family", how="left")

    return fam


def summarize_final_groups(df, split_col="split"):
    dataset_col = find_dataset_col(df)
    train_mask = df[split_col] == "train"

    grp = (
        df.groupby(["group_id", "support_family"], as_index=False)
        .agg(
            n=("y_true", "size"),
            glaucoma_rate=("y_true", "mean"),
            exact_mask_mode=(
                "exact_mask", lambda s: s.value_counts().index[0]
            ),
            n_exact_masks=("exact_mask", "nunique"),
            subcluster=("subcluster", "first"),
        )
        .sort_values("group_id").reset_index(drop=True)
    )

    train_counts = (
        df[train_mask].groupby("group_id").size()
        .reset_index(name="n_train")
    )
    grp = grp.merge(train_counts, on="group_id", how="left")
    grp["n_train"] = grp["n_train"].fillna(0).astype(int)
    grp["train_frac"] = (grp["n_train"] / grp["n"]).round(4)

    if dataset_col:
        mix = (
            df.groupby(["group_id", dataset_col]).size()
            .reset_index(name="cnt")
            .sort_values(
                ["group_id", "cnt", dataset_col], ascending=[True, False, True]
            )
        )
        mix_str = (
            mix.groupby("group_id")
            .apply(
                lambda g: ", ".join(
                    f"{r[dataset_col]}:{int(r['cnt'])}" for _, r in g.iterrows()
                )
            )
            .reset_index(name="dataset_mix")
        )
        grp = grp.merge(mix_str, on="group_id", how="left")

    return grp


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────

def group_dataset(
    csv_path,
    emb_path,
    out_dir="grouping_output",
    split_col="split",
    support_distance_threshold=0.40,
    min_family_size=10,
    max_k=12,
    min_size_frac=0.05,
    min_size_abs=10,
    min_train_per_cluster=20,
    min_group_train=20,
    min_train_frac=0.20,
    random_state=42,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    X = np.load(emb_path)
    assert len(df) == len(X), (
        f"CSV rows ({len(df)}) != embedding rows ({len(X)})"
    )

    assert split_col in df.columns, (
        f"Missing '{split_col}' column. Available: {list(df.columns)}"
    )
    splits = df[split_col].str.lower().str.strip()
    df[split_col] = splits
    train_mask = splits == "train"
    print(f"Split distribution:\n{splits.value_counts().to_string()}\n")

    print("=" * 60)
    print("Stage 1: Support Families (global mask clustering)")
    print("=" * 60)
    df, unique_mask_df, family_reps = build_support_families(
        df,
        mask_col="m_actions",
        distance_threshold=support_distance_threshold,
        min_family_size=min_family_size,
    )
    print(f"  Families: {len(family_reps)}")

    for fam_id in sorted(df["support_family"].unique()):
        fam_mask = df["support_family"] == fam_id
        n_fam = int(fam_mask.sum())
        n_fam_train = int((fam_mask & train_mask).sum())
        print(f"    Family {fam_id:>2d}: n={n_fam:>4d}, "
              f"n_train={n_fam_train:>4d} ({n_fam_train/n_fam:.1%})")

    print()
    print("=" * 60)
    print("Stage 2: KMeans Sub-clustering (fit on train)")
    print("=" * 60)
    X_norm = normalize(X, norm="l2")
    df["subcluster"] = -1
    family_cluster_rows: List[Dict] = []
    total_merges = 0

    for fam_id in sorted(df["support_family"].unique()):
        fam_train_idx = df[(df["support_family"] == fam_id) & train_mask].index.values
        fam_all_idx = df[df["support_family"] == fam_id].index.values
        X_fam_train = X_norm[fam_train_idx]
        X_fam_all = X_norm[fam_all_idx]

        fam_all_train_flags = np.isin(fam_all_idx, fam_train_idx)

        km, best_k, best_sil, eff_min = choose_k_and_fit(
            X_fam_train,
            max_k=max_k,
            min_size_frac=min_size_frac,
            min_size_abs=min_size_abs,
            min_train_per_cluster=min_train_per_cluster,
            random_state=random_state,
        )

        if km is not None:
            labels_all = km.predict(X_fam_all)
            labels_all, _, n_merges = _merge_starved_subclusters(
                km, labels_all, fam_all_train_flags,
                min_train_per_cluster=min_train_per_cluster,
            )
            total_merges += n_merges
            final_k = len(np.unique(labels_all))
        else:
            labels_all = np.zeros(len(fam_all_idx), dtype=int)
            n_merges = 0
            final_k = 1

        df.loc[fam_all_idx, "subcluster"] = labels_all

        sub_stats = {}
        y_train_fam = df.loc[fam_train_idx, "y_true"].values
        train_labels = labels_all[fam_all_train_flags]
        for c in sorted(np.unique(labels_all)):
            c_mask = train_labels == c
            sub_stats[int(c)] = {
                "n_train": int(c_mask.sum()),
                "glaucoma_rate_train": (
                    round(float(y_train_fam[c_mask].mean()), 4)
                    if c_mask.sum() > 0 else None
                ),
            }

        n_held_out = int((~fam_all_train_flags).sum())

        family_cluster_rows.append({
            "support_family": int(fam_id),
            "n_train": int(len(fam_train_idx)),
            "n_all": int(len(fam_all_idx)),
            "n_held_out": n_held_out,
            "kmeans_k": int(best_k),
            "final_k": final_k,
            "subclusters_merged": n_merges,
            "cosine_silhouette_train": (
                round(best_sil, 4) if not np.isnan(best_sil) else None
            ),
            "effective_min_cluster": int(eff_min),
            "subcluster_detail": sub_stats,
        })

        sil_str = f"{best_sil:.4f}" if not np.isnan(best_sil) else "N/A"
        merge_str = f", merged {n_merges}" if n_merges > 0 else ""
        print(f"  Family {fam_id:>2d}: n_train={len(fam_train_idx):>4d}, "
              f"n_all={len(fam_all_idx):>4d}, "
              f"k={best_k}->{final_k}{merge_str}, sil={sil_str}")

    if total_merges > 0:
        print(f"  Total subclusters merged (post-hoc): {total_merges}")

    group_keys = (
        df[["support_family", "subcluster"]]
        .drop_duplicates()
        .sort_values(["support_family", "subcluster"])
        .reset_index(drop=True)
    )
    group_keys["group_id"] = np.arange(len(group_keys))
    df = df.merge(group_keys, on=["support_family", "subcluster"], how="left")

    n_groups_before_fold = df["group_id"].nunique()

    print()
    print("=" * 60)
    print("Stage 3: Small-group folding")
    print("=" * 60)
    print(f"  Thresholds: min_group_train={min_group_train}, "
          f"min_train_frac={min_train_frac}")
    df, n_folded, fold_log = fold_small_groups(
        df, train_mask,
        min_group_train=min_group_train,
        min_train_frac=min_train_frac,
    )

    if n_folded > 0:
        print(f"  Folded {n_folded} group(s):")
        for entry in fold_log:
            print(f"    Family {entry['family']}: subcluster "
                  f"{entry['from_subcluster']}->{entry['to_subcluster']} "
                  f"(n_train={entry['n_train_folded']}, "
                  f"n_all={entry['n_all_folded']}, "
                  f"train_frac={entry['train_frac']:.3f}, "
                  f"{entry['reason']})")
    else:
        print(f"  No groups needed folding")

    print(f"  Groups: {n_groups_before_fold} -> {df['group_id'].nunique()}")

    support_family_summary = summarize_support_families(df, unique_mask_df)
    family_cluster_summary = (
        pd.DataFrame(family_cluster_rows)
        .sort_values("support_family").reset_index(drop=True)
    )
    final_group_summary = summarize_final_groups(df, split_col=split_col)

    split_group_dist = (
        df.groupby([split_col, "group_id"]).size()
        .reset_index(name="n")
        .pivot_table(index="group_id", columns=split_col, values="n", fill_value=0)
        .reset_index()
    )

    support_family_summary.to_csv(
        out_dir / "support_family_summary.csv", index=False
    )
    family_cluster_summary.to_csv(
        out_dir / "family_cluster_summary.csv", index=False
    )
    final_group_summary.to_csv(
        out_dir / "final_group_summary.csv", index=False
    )
    split_group_dist.to_csv(
        out_dir / "split_group_distribution.csv", index=False
    )
    if fold_log:
        pd.DataFrame(fold_log).to_csv(
            out_dir / "fold_log.csv", index=False
        )
    df.to_csv(out_dir / "final_dataset3_grouped.csv", index=False)

    print()
    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"rows               = {len(df)}")
    print(f"  train            = {int(train_mask.sum())}")
    print(f"  held-out         = {int((~train_mask).sum())}")
    print(f"exact masks        = {df['exact_mask'].nunique()}")
    print(f"support families   = {df['support_family'].nunique()}")
    print(f"final groups       = {df['group_id'].nunique()}")

    return df, support_family_summary, family_cluster_summary, final_group_summary


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Hybrid similarity bucketing for MPD2-Router (leak-free v3)"
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--emb", required=True)
    p.add_argument("--out_dir", default="grouping_output")
    p.add_argument("--split_col", default="split")
    p.add_argument("--support_distance_threshold", type=float, default=0.40)
    p.add_argument("--min_family_size", type=int, default=10)
    p.add_argument("--max_k", type=int, default=12)
    p.add_argument("--min_size_frac", type=float, default=0.05)
    p.add_argument("--min_size_abs", type=int, default=10)
    p.add_argument("--min_train_per_cluster", type=int, default=20)
    p.add_argument("--min_group_train", type=int, default=20)
    p.add_argument("--min_train_frac", type=float, default=0.20)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    group_dataset(
        csv_path=args.csv,
        emb_path=args.emb,
        out_dir=args.out_dir,
        split_col=args.split_col,
        support_distance_threshold=args.support_distance_threshold,
        min_family_size=args.min_family_size,
        max_k=args.max_k,
        min_size_frac=args.min_size_frac,
        min_size_abs=args.min_size_abs,
        min_train_per_cluster=args.min_train_per_cluster,
        min_group_train=args.min_group_train,
        min_train_frac=args.min_train_frac,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
