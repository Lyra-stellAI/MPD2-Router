# MPD²-Router (M2P)

**Hierarchical, constraint-aware multi-expert learning-to-defer for retinal-image glaucoma classification.**

MPD²-Router is a learning-to-defer (L2D) system that decides, for each fundus image, whether to (i) accept the AI prediction, (ii) defer to one of 12 human experts subject to availability and tier-cost constraints, or (iii) defer in a way that respects a hard deployment-level deferral-rate budget. The router is trained against a hierarchical badness-ranked routing prior (global → support-family → group), regularised with two divergences (GSDP and Rank-JS), and constrained via an augmented Lagrangian dual-update on the deferral rate and average tier cost.

This repository accompanies the submission and contains the full pipeline: feature extraction, OOD risk computation, dataset grouping, hyperparameter optimisation, and router training.

---

## Contents

1. [Pipeline overview](#pipeline-overview)
2. [Methodological highlights](#methodological-highlights)
3. [Repository layout](#repository-layout)
4. [Datasets](#datasets)
5. [Installation](#installation)
6. [Quick start](#quick-start)
7. [End-to-end walkthrough](#end-to-end-walkthrough)
8. [Module reference](#module-reference)
9. [Data schema](#data-schema)
10. [Configuration reference](#configuration-reference)
11. [Reproducibility](#reproducibility)
12. [Citation](#citation)

---

## Pipeline overview

```
       ┌──────────────────────────┐
       │  Fundus images (REFUGE,  │
       │  ORIGA, CHAKSU)          │
       └──────────────┬───────────┘
                      │  Swin-V2 (frozen)
                      ▼
       ┌──────────────────────────┐    notebooks/02_ood_risk.ipynb
       │  Pooled embeddings,      │   ┌─────────────────────────┐
       │  hidden states, logits   │──►│ OOD detectors           │
       │  (.npy + .csv)           │   │ MSP / Energy / kNN /    │
       └──────────────┬───────────┘   │ ViM / Mahalanobis       │
                      │               └────────────┬────────────┘
                      ▼                            │  vim_risk_z
       ┌──────────────────────────┐                │  maha_risk
       │ data/final_dataset3.csv  │◄───────────────┘  quality_risk
       └──────────────┬───────────┘
                      │  notebooks/01_grouping.ipynb  (src/m2p/grouping.py)
                      ▼
       ┌──────────────────────────┐
       │ Stage 1: support family  │  (Hamming-clustered expert masks)
       │ Stage 2: KMeans subgroup │  (fit on train only)
       │ Stage 3: small-group     │  (count + train-fraction folds)
       │         fold             │
       └──────────────┬───────────┘
                      │  group_id, support_family, subcluster
                      ▼
       ┌──────────────────────────┐  src/m2p/adaptive_hpo.py
       │ Adaptive Bayesian HPO    │  (multivariate TPE + Hyperband,
       │ over routing & AL knobs  │   constraint-aware checkpoint sel.)
       └──────────────┬───────────┘
                      │  best_params, study
                      ▼
       ┌──────────────────────────┐  notebooks/03_router_training.ipynb
       │ MPD²-Router              │
       │  • risk_enc + struct_enc │
       │  • OVA expert head       │
       │  • augmented Lagrangian  │
       │  • GSDP + Rank-JS reg.   │
       └──────────────────────────┘
```

---

## Methodological highlights

### 1. Hierarchical badness-ranked routing prior

The L2D routing prior is constructed at three levels and blended:

* **Global prior** — a uniform routing distribution over the action set with a small uniform-mix coefficient.
* **Family prior** — per support family (i.e. set of available experts), built from a Dirichlet-like pseudo-count (`family_n0`) interpolated with a uniform mix.
* **Group prior** — per (family, sub-cluster) bucket, a **truncated geometric** rank prior over experts ranked by *badness* (FN/FP cost relative to the group's prevalence), capped by an anti-collapse `clip_max(k)`.

The truncated geometric top mass

```
g_{k,1}(ρ) = (1 − ρ) / (1 − ρ^k)
```

is exposed as the natural reference point for `clip_max`, controlled by two intuitive knobs (`clip_ceiling`, `clip_slack`) — see `src/m2p/adaptive_hpo.py:99`. This replaces brittle per-anchor cap dictionaries.

### 2. Two complementary regularisers (`combined_routing_loss`)

* **GSDP** — group-conditional KL/JS divergence between the empirical routing distribution `q̄_g` and the learned group prior, enforcing group-wise structural conformance.
* **Rank-JS** — per-sample JS against the rank-ordered simplex with `rank_js_margin` slack and `any_excess` reduction, controlling routing peakiness.

### 3. Augmented Lagrangian deferral budget

`AugLag` enforces a deployment-level cap on the deferral rate (`max_deferral_rate ≤ 0.70`) and, optionally, on the average tier cost. The penalty term

```
loss += μ · max(0, defer̄ − ρ)² + λ · max(0, defer̄ − ρ)
```

is added to the clinical objective; the dual variable `λ` is updated each epoch with `lr_lambda`, and `μ` provides the quadratic backstop.

### 4. Asymmetric clinical cost

Both prior construction and the training loss respect FN ≫ FP via `c_fn = 2.0`, `c_fp = 1.5` (`train_costs` / `prior_costs`). The expert action head emits a one-vs-all (OVA) score; the AI action incurs an expected clinical cost from `(p_ai, y_true)`.

### 5. Constraint-aware Bayesian HPO

`src/m2p/adaptive_hpo.py` runs Optuna with:

* **Multivariate TPE** seeded with a known-good notebook config.
* **Hyperband pruner** keyed on the lower bound of the warmup search range.
* **Augmented Chebyshev scalarisation** over five objectives (clinical, MCC, AUPRC, tier cost, constraint violation) with adaptive utopia tracking.
* **Selection score** that balances ES base score against constraint violation, so trials that meet the deferral budget are preferred even when raw clinical cost ties.
* **JSONL ledger** of every materialised config + metrics for post-hoc analysis (`hpo_ledger.jsonl`).

---

## Repository layout

```
M2P/
├── README.md                              ← you are here
├── requirements.txt
├── data/
│   └── final_dataset3.csv                 ← OOD-scored dataset (3195 rows)
├── notebooks/
│   ├── 01_grouping.ipynb                  ← Stage 1-3 hierarchical bucketing
│   ├── 02_ood_risk.ipynb                  ← Swin-V2 features + OOD scoring
│   └── 03_router_training.ipynb           ← lbl + OVA router training & eval
├── src/m2p/
│   ├── __init__.py
│   ├── grouping.py                        ← scriptable version of notebook 01
│   ├── ood.py                             ← KNN/ViM/Mahalanobis + logit scores
│   ├── adaptive_hpo.py                    ← current HPO module (v3)
│   └── adaptive_hpo_geometric_clip.py     ← prior HPO variant kept for repro
├── configs/                               ← (intentionally empty — populate per study)
└── docs/                                  ← extended notes / figures
```

---

## Datasets

| Dataset | Total | Train | Val | Test | Glaucoma rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| **REFUGE**  | 1200 | 400 | 400 | 400 | 10.0% | In-distribution reference cohort |
| **CHAKSU**  | 1345 | 686 | 323 | 336 | 14.0% | Out-of-distribution |
| **ORIGA**   |  650 | 325 | 162 | 163 | 25.8% | Out-of-distribution |
| **Total**   | **3195** | 1411 | 885 | 899 | 14.9% | |

Twelve human experts annotate subsets of the data:

* **REFUGE experts (7):** `y_refuge_expert_1` … `y_refuge_expert_7`
* **CHAKSU experts (5):** `y_chaksu_expert_1` … `y_chaksu_expert_5`

The per-row `m_experts` column records availability (12-bit), and `m_actions = [1] + m_experts` (13-bit) prepends the always-available AI action.

---

## Installation

```bash
git clone <repo-url> M2P
cd M2P
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .         # optional: makes `m2p.*` importable from anywhere
```

A CUDA-enabled GPU is recommended for Swin-V2 feature extraction and router training, but the OOD detectors and grouping pipeline run comfortably on CPU.

---

## Quick start

```python
from m2p.grouping import group_dataset

# Three-stage hierarchical bucketing -> writes final_dataset3_grouped.csv
df, fam_summary, cluster_summary, group_summary = group_dataset(
    csv_path="data/final_dataset3.csv",
    emb_path="path/to/all_emb_swinv2.npy",   # (N, 768) Swin-V2 pooled embeddings
    out_dir="grouping_output",
)
```

```python
from m2p.adaptive_hpo import run_hpo_from_notebook
# Pass in the notebook-defined classes/functions directly:
best_params, study = run_hpo_from_notebook(
    df=df,                                  # grouped DataFrame
    tier_cost_dict=tier_cost,
    action_costs_fn=action_costs,
    build_all_priors_fn=build_all_priors,
    L2DDataset_cls=L2DDataset,
    Router_cls=Router,
    l2d_objective_fn=l2d_objective,
    combined_routing_loss_fn=combined_routing_loss,
    AugLag_cls=AugLag,
    evaluate_fn=evaluate,
    n_trials=80,
)
```

---

## End-to-end walkthrough

### Step 1 — feature extraction (`notebooks/02_ood_risk.ipynb`)

Loads `pamixsun/swinv2_tiny_for_glaucoma_classification` and runs `extract_features(...)` over each dataset, capturing:

* `logits` — `(N, 2)` — feeds MSP / MaxLogit / Entropy / Energy.
* `pooled_last` — `(N, 768)` — feeds kNN, ViM, Energy+ReAct.
* `hidden[2|3|4]` — `(N, *, *)` — multi-layer features for Mahalanobis.

The pooled embeddings are concatenated across REFUGE / ORIGA / CHAKSU into `all_emb_swinv2.npy` for the grouping step.

### Step 2 — OOD risk scoring (`notebooks/02_ood_risk.ipynb`)

The detectors in `src/m2p/ood.py` are fit on the REFUGE train split only and scored on every other row. The two scores actually consumed downstream are standardised against REFUGE-train statistics:

```python
df["maha_risk"] = (df["Mahalanobis"] - μ) / σ
df["vim_risk"]  = (df["ViM"]         - μ) / σ
```

They become input features to the router's `risk_enc`. Reported AUROC on the held-out OOD splits (notebook output):

| Detector  | CHAKSU AUROC | ORIGA AUROC |
|---|---:|---:|
| **ViM**          | **0.978** | **0.983** |
| Mahalanobis | 0.932 | 0.964 |
| kNN         | 0.784 | 0.953 |
| Energy+ReAct| 0.756 | 0.485 |
| MSP / Entropy / MaxLogit / Energy | ~0.72 | ~0.64 |

### Step 3 — grouping (`notebooks/01_grouping.ipynb` or `python -m m2p.grouping`)

Three-stage pipeline (see [Methodological highlights §1](#1-hierarchical-badness-ranked-routing-prior)):

```bash
python -m m2p.grouping \
    --csv data/final_dataset3.csv \
    --emb path/to/all_emb_swinv2.npy \
    --out_dir grouping_output \
    --support_distance_threshold 0.40 \
    --min_family_size 10 \
    --max_k 12 \
    --min_train_per_cluster 20 \
    --min_group_train 20 \
    --min_train_frac 0.20
```

**Why train-fraction folding matters.** Stage 3 folds groups whose train fraction `< min_train_frac` even when `n_train ≥ min_group_train`. This catches pathological cases (38 train / 513 total = 7.4 %) that pass the absolute-count check but starve the empirical group prior.

### Step 4 — hyperparameter optimisation (`src/m2p/adaptive_hpo.py`)

```python
from m2p.adaptive_hpo import run_hpo_from_notebook, print_study_summary

best_params, study = run_hpo_from_notebook(
    df=df_grouped,
    tier_cost_dict=tier_cost,
    action_costs_fn=action_costs,
    build_all_priors_fn=build_all_priors,
    L2DDataset_cls=L2DDataset,
    Router_cls=Router,
    l2d_objective_fn=l2d_objective,
    combined_routing_loss_fn=combined_routing_loss,
    AugLag_cls=AugLag,
    evaluate_fn=evaluate,
    n_trials=80,
    storage="sqlite:///hpo.db",          # persistent study
    ledger_path="hpo_ledger.jsonl",
)
print_study_summary(study)
```

The 16-dim search space (see `SearchSpace` in `src/m2p/adaptive_hpo.py:175`) covers learning rate, tier weight, warmup epochs, prior-mix coefficients, geometric clip knobs, and Lagrangian step sizes. A known-good seed configuration is enqueued first via `study.enqueue_trial` so the first trial always returns a usable baseline.

### Step 5 — final training (`notebooks/03_router_training.ipynb`)

The notebook builds the Router (`risk_enc` + `struct_enc` + `ai_enc` + OVA expert head + deferral head), trains for up to 150 epochs with 15-epoch warmup and patience-18 early stopping using the constraint-aware selection score from `_selection_score`, and reports per-dataset accuracy, MCC, sensitivity, specificity, deferral rate, and clinical cost.

To rebuild configs from a completed study:

```python
from m2p.adaptive_hpo import retrain_with_best
train_cfg, mcfg, al_cfg = retrain_with_best(
    best_params, df_grouped, expert_cols=None, train_fn=None,
    extra_epochs=30,
)
```

---

## Module reference

### `m2p.grouping`

| Function | Purpose |
|---|---|
| `parse_mask`, `drop_ai_slot`, `mask_to_str` | Utilities for the binary expert-availability mask. |
| `build_support_families(df, …)` | Stage 1 — Hamming Agglomerative clustering of unique masks, micro-family absorption, and per-row Hamming-to-rep distance. |
| `choose_k_and_fit(X_norm_train, …)` | Stage 2 — silhouette × log₂(K) selection with hard min-train-per-cluster constraint. |
| `_merge_starved_subclusters(...)` | Post-hoc merging of sub-clusters whose train count fell below threshold. |
| `fold_small_groups(df, train_mask, …)` | Stage 3 — count + train-fraction folding with full audit log. |
| `group_dataset(...)` | End-to-end driver writing `final_dataset3_grouped.csv` and four summary CSVs. |

### `m2p.ood`

| Symbol | Purpose |
|---|---|
| `score_msp` / `score_maxlogit` / `score_entropy` / `score_energy` / `score_energy_react` | Logit-based detectors (sign-flipped so higher = more OOD). |
| `KNN_OOD(k=10)` | Distance to k-th cosine-normalised train neighbour. |
| `ViM_OOD(pca_dim=256)` | Virtual-logit Matching with PCA-residual norm. |
| `Mahalanobis_OOD(layers=(2,3,4))` | Multi-layer Mahalanobis with Ledoit-Wolf shrinkage. |
| `ood_metrics(id, ood, tau=99.0)` | AUROC / AUPR_OOD / FPR@τ. |

### `m2p.adaptive_hpo`

| Symbol | Purpose |
|---|---|
| `default_rho_by_k`, `truncated_geometric_top_mass`, `clip_max_from_geom`, `build_geometric_clip_anchors` | Geometric anti-collapse cap utilities. |
| `SearchSpace` | 16-dim declarative search space. |
| `sample_config`, `params_to_configs` | Sampling + materialisation into `(train_cfg, mcfg, al_cfg)`. |
| `ObjectiveWeights`, `UtopiaTracker` | Augmented Chebyshev scalarisation with adaptive utopia. |
| `TrialExecutor` | Per-trial training loop with intermediate Hyperband reports and constraint-aware checkpoint selection. |
| `make_objective`, `run_hpo`, `run_hpo_from_notebook` | Top-level HPO drivers. |
| `pareto_front`, `importance_analysis`, `retrain_with_best`, `print_study_summary` | Post-hoc analysis. |

---

## Data schema

`data/final_dataset3.csv` — 3195 rows × 36 columns.

| Column | Type | Description |
|---|---|---|
| `global_id` | str | Stable identifier across datasets. |
| `dataset` | {`refuge`,`origa`,`chaksu`} | Source cohort. |
| `y_true` | {0,1} | Glaucoma label. |
| `y_*_expert_*` | float / NaN | 12 expert label columns; NaN means the expert did not annotate this image. |
| `m_experts` | list[bool] | 12-element availability mask. |
| `m_actions` | list[int] | 13-element action mask, with the AI slot prepended. |
| `split` | {`train`,`val`,`test`} | Split assignment (per-dataset). |
| `is_ood` | {0,1} | 1 for ORIGA / CHAKSU. |
| `logit_0`, `logit_1` | float | Frozen Swin-V2 logits. |
| `prob_0`, `prob_1`, `pred`, `confidence`, `uncertainty` | float | Softmax-derived. |
| `ViM`, `Mahalanobis`, `vim_risk`, `maha_risk`, `vim_risk_z` | float | OOD scores; `*_risk` are z-standardised against REFUGE-train. |
| `quality_score`, `quality_risk` | float | Image quality risk. |
| `vCDR`, `hCDR`, `aCDR` | float | Cup-disc-ratio structural features. |

After `m2p.grouping.group_dataset(...)` the grouped CSV additionally carries:

| Column | Description |
|---|---|
| `exact_mask` | String form of the 12-bit mask. |
| `support_family` | Hamming-clustered family id. |
| `hamming_to_family_rep` | Distance to family representative. |
| `subcluster` | Family-local KMeans cluster. |
| `group_id` | Flat (family, subcluster) id consumed by the prior table. |

---

## Configuration reference

Defaults from `m2p.adaptive_hpo.run_hpo`. Use these as the starting point unless ablating.

| Group | Knob | Default | Range / notes |
|---|---|---:|---|
| **Train**    | `lr`             | `1e-4` | log-uniform `[3e-5, 3e-4]` |
|              | `gamma_tier`     | `1.0`  | log-uniform `[0.05, 1.5]` |
|              | `warmup_epochs`  | `15`   | int `[8, 20]` |
|              | `epochs`         | `150`  | fixed |
|              | `patience`       | `18`   | fixed (min_delta = 1e-4) |
|              | `c_fn` / `c_fp`  | `2.0` / `1.5` | study-design constant |
| **Prior reg.**| `tau_bad`       | `1.0`  | `[0.30, 2.00]` |
|              | `w_gsdp`         | `0.30` | `[0.03, 1.00]` |
|              | `w_rank_js`      | `0.30` | `[0.03, 1.00]` |
|              | `global_uniform_mix` | `0.35` | `[0.05, 0.40]` |
|              | `family_uniform_mix` | `0.30` | `[0.10, 0.55]` |
|              | `group_uniform_mix`  | `0.30` | `[0.10, 0.55]` |
|              | `family_n0` / `group_n0` | `25` / `30` | Dirichlet pseudo-counts |
|              | `clip_ceiling`   | `0.35` | `[0.30, 0.38]` |
|              | `clip_slack`     | `0.03` | `[0.00, 0.06]` |
| **AugLag**   | `mu`             | `25.0` | `[8, 40]` |
|              | `lr_lambda`      | `0.10` | log-uniform `[0.02, 0.30]` |
|              | `max_deferral_rate` | `0.70` | hard deployment constraint |
| **Grouping** | `support_distance_threshold` | `0.40` | hamming agglomerative cutoff |
|              | `min_family_size` | `10` | absorbs micro-families |
|              | `min_train_per_cluster` | `20` | KMeans hard floor |
|              | `min_group_train` | `20` | Stage-3 abs. fold threshold |
|              | `min_train_frac`  | `0.20` | Stage-3 fraction fold threshold |

---

## Reproducibility

* **Determinism.** Each HPO trial seeds `torch.manual_seed`, `numpy.random.seed`, and `torch.cuda.manual_seed_all` with `seed + trial.number`, and DataLoader shuffling uses a per-trial generator.
* **Seed config.** `run_hpo` enqueues a known-good seed configuration so the very first trial reproduces the notebook's manual baseline.
* **Trial ledger.** Every trial appends a JSON line with the full `(params, train_cfg, mcfg, al_cfg, metrics, scalar_score, utopia, elapsed_s)` to `hpo_ledger.jsonl`. Together with `optuna.create_study(storage="sqlite:///hpo.db", load_if_exists=True)` this lets you resume / re-analyse studies.
* **Early NaN guards.** Both the per-batch loss and the final metric vector are NaN-checked; offending trials are pruned rather than silently corrupting the study.

---

## Citation

A formal citation entry will be added once the manuscript is published. In the meantime, please cite this repository directly.

---

## License

To be added.
