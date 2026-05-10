# MPD²-Router

**Hierarchical, constraint-aware multi-expert learning-to-defer for retinal-image glaucoma classification.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> Official implementation of the NeurIPS submission *“MPD²-Router: Hierarchical Constraint-Aware Multi-Expert Learning-to-Defer for Glaucoma Classification.”*

---

## Abstract

We address the problem of *learning-to-defer* (L2D) in clinical glaucoma screening, where a frozen image classifier (Swin-V2) must collaborate with a roster of twelve human experts under three real-world constraints: (i) experts are not uniformly available across cases; (ii) consultations carry asymmetric *tier* costs, and (iii) deployment caps the deferral budget at ρ ≤ 0.7. We introduce **MPD²-Router**, a routing policy regularised by a three-level *badness*-ranked prior (global → support family → local group) and trained with two complementary divergences (GSDP for group-conditional structure and Rank-JS for sample-level peakiness control). Hard deployment constraints are enforced via an augmented Lagrangian on the deferral rate and average tier cost. Hyperparameters are chosen by a constraint-aware Bayesian optimiser with multivariate TPE and a Hyperband pruner, scalarised over a five-objective augmented Chebyshev with adaptive utopia tracking. On REFUGE (in-distribution) and CHAKSU / ORIGA (out-of-distribution), MPD²-Router improves Matthews correlation and AUPRC over the AI-only baseline while keeping deferral within budget and concentrating queries on the experts with the strongest test-time evidence.

---

## Table of contents

1. [Method](#method)
2. [Pipeline overview](#pipeline-overview)
3. [Repository layout](#repository-layout)
4. [Installation](#installation)
5. [Reproducing the submission](#reproducing-the-submission)
6. [Datasets](#datasets)
7. [Module reference](#module-reference)
8. [Data schema](#data-schema)
9. [Configuration reference](#configuration-reference)
10. [Reproducibility](#reproducibility)
11. [Citation](#citation)
12. [License](#license)

---

## Method

### 1. Hierarchical badness-ranked routing prior

For each level of the hierarchy we compute per-expert *badness*

```
b_j = c_fn · FNR_j + c_fp · FPR_j + tier_cost_j
```

with Beta-smoothed FNR/FPR (`alpha = beta = 1`). Levels are then mixed via Dirichlet pseudo-counts:

* **Global**  — `p_global = (1 − u_g) · softmax(−τ · b) + u_g · U`
* **Family**  — `p_family = λ_f · p_raw_f + (1 − λ_f) · p_global`,    `λ_f = n_f / (n_f + n0_family)`
* **Group**   — `p_group  = λ_g · p_raw_g + λ_f · p_family + λ_glo · p_global`,    `λ_g = n_g / (n_g + n0_group)`

After mixing, group-level entries are clipped by `_adaptive_clip_max(k_active, anchors)` to prevent collapse onto a single expert. The truncated geometric top mass

```
g_{k,1}(ρ) = (1 − ρ) / (1 − ρ^k)
```

is exposed as the natural reference for the cap, controlled by two intuitive HPO knobs (`clip_ceiling`, `clip_slack`) — see `src/m2p/adaptive_hpo.py:99`.

### 2. Two complementary regularisers

Implemented in `src/m2p/losses.py`.

* **GSDP** — group-conditional KL or JS divergence between the empirical routing distribution `q̄_g = Σ_i d_i q_i / Σ_i d_i` and the precomputed group prior `p_g`.
* **Rank-JS** — per-sample JS against the truncated geometric rank prior, with a `mode='any_excess'` gate that activates only when the sample's sorted prefix sums exceed the geometric reference (i.e. the routing is *too peaky*).

### 3. Augmented Lagrangian deferral budget

`m2p.augmented_lagrangian.AugLag` enforces

```
  E[defer] ≤ ρ        (max_deferral_rate)
  E[cost]  ≤ C        (max_avg_cost, optional)
```

via the standard penalty `λ · g + ½ μ · max(0, g)²` on each constraint, with `λ ← max(0, λ + lr_λ · g)` updated once per epoch.

### 4. Constraint-aware Bayesian HPO

`src/m2p/adaptive_hpo.py` runs Optuna with multivariate TPE seeded by a known-good notebook config, a Hyperband pruner keyed on the lower bound of the warmup search range, and an augmented Chebyshev scalarisation over `(clinical, MCC, AUPRC, tier_soft, es_violation)` with an EMA-blended utopia point. A constraint-aware selection score then prefers checkpoints that meet the deferral budget even when raw clinical cost ties.

---

## Pipeline overview

```
       ┌──────────────────────────┐
       │  Fundus images (REFUGE,  │
       │  ORIGA, CHAKSU)          │
       └──────────────┬───────────┘
                      │  m2p.feature_extraction (Swin-V2 frozen)
                      ▼
       ┌──────────────────────────┐
       │  Pooled embeddings,      │   m2p.ood
       │  hidden states, logits   │──► MSP / Energy / kNN /
       └──────────────┬───────────┘     ViM / Mahalanobis  ───┐
                      │                                       │
                      │   vim_risk_z, maha_risk, quality_risk │
                      ▼ ◄─────────────────────────────────────┘
       ┌──────────────────────────┐
       │  data/final_dataset3.csv │
       └──────────────┬───────────┘
                      │  m2p.grouping  (Stage 1–3)
                      ▼
       ┌──────────────────────────┐
       │ support_family,          │
       │ subcluster, group_id     │
       └──────────────┬───────────┘
                      │  m2p.adaptive_hpo  (TPE + Hyperband)
                      ▼
       ┌──────────────────────────┐
       │ best_params              │
       └──────────────┬───────────┘
                      │  m2p.training.train_l2d_multi_expert
                      ▼
       ┌──────────────────────────┐
       │ MPD²-Router              │
       │  • risk_enc + struct_enc │
       │  • OVA expert head       │
       │  • augmented Lagrangian  │
       │  • GSDP + Rank-JS reg.   │
       └──────────────────────────┘
```

---

## Repository layout

```
M2P/
├── README.md
├── LICENSE                                ← MIT
├── CITATION.cff                           ← citation metadata
├── pyproject.toml                         ← installable package + console scripts
├── requirements.txt                       ← pinned transitive deps
├── configs/
│   └── best_hpo.yaml                      ← submission config (HPO trial 50)
├── data/
│   └── final_dataset3.csv                 ← 3195-row OOD-scored manifest
├── src/m2p/
│   ├── __init__.py
│   ├── configs.py                         ← TrainConfig / PriorRegConfig / ALConfig
│   ├── costs.py                           ← TIER_COST + clinical-cost asymmetry
│   ├── data.py                            ← L2DDataset
│   ├── feature_extraction.py              ← frozen Swin-V2 wrapper
│   ├── ood.py                             ← KNN / ViM / Mahalanobis + logit scores
│   ├── grouping.py                        ← three-stage hierarchical bucketing
│   ├── priors.py                          ← global → family → group prior tables
│   ├── models.py                          ← Router + OVA expert head + struct head
│   ├── losses.py                          ← GSDP + rank-majorisation JS
│   ├── augmented_lagrangian.py            ← AugLag dual updates
│   ├── training.py                        ← train_l2d_multi_expert
│   ├── evaluation.py                      ← evaluate + reporting
│   ├── adaptive_hpo.py                    ← current Bayesian HPO (v3)
│   └── adaptive_hpo_geometric_clip.py     ← prior HPO variant kept for repro
├── scripts/
│   ├── extract_features.py                ← Swin-V2 feature extraction
│   ├── run_grouping.py                    ← three-stage bucketing CLI
│   ├── run_hpo.py                         ← constraint-aware Bayesian study
│   └── train_router.py                    ← final training + test eval
└── docs/                                  ← (extended notes / figures)
```

---

## Installation

```bash
git clone https://github.com/Lyra-stellAI/M2P
cd M2P
python -m venv .venv && source .venv/bin/activate
pip install -e ".[yaml]"
```

A CUDA-enabled GPU is recommended for Swin-V2 feature extraction and router training; the OOD detectors and grouping pipeline run comfortably on CPU.

---

## Reproducing the submission

The submitted system can be reproduced from the shipped manifest in four steps; each step writes its outputs to disk so later steps can be resumed independently.

### 1. (Optional) Extract Swin-V2 features

Required only if you need to re-derive the OOD scores; the shipped `data/final_dataset3.csv` already carries `vim_risk_z`, `maha_risk`, etc.

```bash
python scripts/extract_features.py \
    --csv data/manifests/refuge.csv \
    --out_dir data/features/refuge
```

### 2. Hierarchical bucketing

```bash
python scripts/run_grouping.py \
    --csv data/final_dataset3.csv \
    --emb data/features/all_pooled_last.npy \
    --out_dir grouping_output
```

This writes `grouping_output/final_dataset3_grouped.csv` (used by the next two steps) and four summary CSVs (support family, family clustering, final groups, fold log).

### 3. (Optional) Re-run the HPO study

```bash
python scripts/run_hpo.py \
    --csv grouping_output/final_dataset3_grouped.csv \
    --n_trials 80 \
    --storage sqlite:///hpo.db \
    --ledger hpo_ledger.jsonl
```

Skip this step if you want to reproduce the submission exactly using `configs/best_hpo.yaml`.

### 4. Train the router and evaluate on the test split

```bash
python scripts/train_router.py \
    --csv grouping_output/final_dataset3_grouped.csv \
    --config configs/best_hpo.yaml \
    --out_dir runs/mpd2-router-001
```

Outputs:

* `runs/mpd2-router-001/router.pt` — best-checkpoint state dict.
* `runs/mpd2-router-001/test_metrics.json` — final test-split metrics.
* Console: per-action and per-dataset breakdown via `print_eval_report` and `print_eval_report_dataset`.

---

## Datasets

| Dataset | Total | Train | Val | Test | Glaucoma rate | Role |
|---|---:|---:|---:|---:|---:|---|
| **REFUGE**  | 1200 | 400 | 400 | 400 | 10.0 % | In-distribution reference cohort |
| **CHAKSU**  | 1345 | 686 | 323 | 336 | 14.0 % | Out-of-distribution |
| **ORIGA**   |  650 | 325 | 162 | 163 | 25.8 % | Out-of-distribution |
| **Total**   | **3195** | 1411 | 885 | 899 | 14.9 % | |

Twelve human experts annotate subsets of the data:

* **REFUGE experts (7):** `y_refuge_expert_1` … `y_refuge_expert_7`
* **CHAKSU experts (5):** `y_chaksu_expert_1` … `y_chaksu_expert_5`

The per-row `m_experts` column encodes availability (12-bit), and `m_actions = [1] + m_experts` (13-bit) prepends the always-available AI action.

---

## Module reference

### `m2p.feature_extraction`

| Symbol | Purpose |
|---|---|
| `SWINV2_HF_NAME` | HuggingFace model id (`pamixsun/swinv2_tiny_for_glaucoma_classification`). |
| `load_swinv2()` | Build the frozen classifier and image processor. |
| `extract_features(...)` | Returns `{logits, pooled_last, hidden, labels, …}` for an image manifest. |

### `m2p.ood`

| Symbol | Purpose |
|---|---|
| `score_msp` / `score_maxlogit` / `score_entropy` / `score_energy` / `score_energy_react` | Logit-based detectors (sign-flipped so higher = more OOD). |
| `KNN_OOD(k=10)` | Distance to the *k*-th cosine-normalised train neighbour. |
| `ViM_OOD(pca_dim=256)` | Virtual-logit Matching with PCA-residual norm. |
| `Mahalanobis_OOD(layers=(2,3,4))` | Multi-layer Mahalanobis with Ledoit-Wolf shrinkage. |
| `ood_metrics(id, ood, tau=99.0)` | AUROC / AUPR_OOD / FPR@τ. |

### `m2p.grouping`

| Function | Purpose |
|---|---|
| `build_support_families(...)` | Stage 1 — Hamming-clustered expert masks + micro-family absorption. |
| `choose_k_and_fit(...)` | Stage 2 — silhouette × log₂ K on train embeddings. |
| `_merge_starved_subclusters(...)` | Post-hoc merging of train-starved sub-clusters. |
| `fold_small_groups(...)` | Stage 3 — count + train-fraction folding with audit log. |
| `group_dataset(...)` | End-to-end driver writing `final_dataset3_grouped.csv`. |

### `m2p.priors`

| Function | Purpose |
|---|---|
| `_compute_expert_stats(...)` | Beta-smoothed FNR/FPR + badness for a (sub-)cohort. |
| `_scores_to_prior(...)` | softmax(−τ · badness) with uniform mixing. |
| `compute_global_prior(...)` / `compute_family_prior(...)` / `compute_group_prior(...)` | Hierarchy primitives. |
| `build_all_priors(...)` | One-shot driver returning `final_group_table` and metadata. |

### `m2p.models`

| Symbol | Purpose |
|---|---|
| `StructuralRisk` | Calibrated 2-D linear head over (vCDR, aCDR). |
| `MLPBlock` | `Linear → ReLU → LayerNorm → Dropout`. |
| `OVAExpertHeadFixed` | Two-stage gate-and-allocate expert head with differentiable selection. |
| `Router` | Three-branch fusion + deferral head + OVA expert head. |

### `m2p.losses`

| Function | Purpose |
|---|---|
| `gsdp_loss(...)` | Group-conditional KL/JS against precomputed group priors. |
| `rank_majorization_js_loss(...)` | Per-sample JS against truncated-geometric rank prior, gated by majorisation. |
| `combined_routing_loss(...)` | `cfg.w_gsdp · GSDP + cfg.w_rank_js · Rank-JS`. |

### `m2p.augmented_lagrangian`

| Symbol | Purpose |
|---|---|
| `AugLag(cfg, device)` | Stateful augmented-Lagrangian helper with non-negative dual variables. |

### `m2p.training` / `m2p.evaluation`

| Symbol | Purpose |
|---|---|
| `train_l2d_multi_expert(df, cfg, mcfg, al_cfg)` | End-to-end loop with constraint-aware ES on the val split. |
| `evaluate(...)` | Held-out eval pass with per-action / per-dataset metrics. |
| `print_eval_report(...)` / `print_eval_report_dataset(...)` | Reporting tables. |

### `m2p.adaptive_hpo`

| Symbol | Purpose |
|---|---|
| `default_rho_by_k`, `clip_max_from_geom`, `build_geometric_clip_anchors` | Geometric anti-collapse cap utilities. |
| `SearchSpace` | 16-dim declarative search space. |
| `sample_config`, `params_to_configs` | Sampling + materialisation into `(train_cfg, mcfg, al_cfg)`. |
| `ObjectiveWeights`, `UtopiaTracker` | Augmented Chebyshev scalarisation + adaptive utopia. |
| `TrialExecutor`, `make_objective`, `run_hpo`, `run_hpo_from_notebook` | HPO drivers. |
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
| `quality_score`, `quality_risk` | float | Image-quality risk. |
| `vCDR`, `hCDR`, `aCDR` | float | Cup-disc-ratio structural features. |

After `m2p.grouping.group_dataset(...)` the grouped CSV additionally carries `exact_mask`, `support_family`, `hamming_to_family_rep`, `subcluster`, `group_id`.

---

## Configuration reference

Defaults from `m2p.adaptive_hpo.run_hpo`. Use these as the starting point unless ablating.

| Group | Knob | Default | Range / notes |
|---|---|---:|---|
| **Train**     | `lr`             | `1e-4` | log-uniform `[3e-5, 3e-4]` |
|               | `gamma_tier`     | `1.0`  | log-uniform `[0.05, 1.5]` |
|               | `warmup_epochs`  | `15`   | int `[8, 20]` |
|               | `epochs`         | `150`  | fixed |
|               | `patience`       | `18`   | fixed (`min_delta = 1e-4`) |
|               | `c_fn` / `c_fp`  | `2.0` / `1.5` | study-design constant |
| **Prior reg.**| `tau_bad`        | `1.0`  | `[0.30, 2.00]` |
|               | `w_gsdp`         | `0.30` | `[0.03, 1.00]` |
|               | `w_rank_js`      | `0.30` | `[0.03, 1.00]` |
|               | `global_uniform_mix` | `0.35` | `[0.05, 0.40]` |
|               | `family_uniform_mix` | `0.30` | `[0.10, 0.55]` |
|               | `group_uniform_mix`  | `0.30` | `[0.10, 0.55]` |
|               | `family_n0` / `group_n0` | `25` / `30` | Dirichlet pseudo-counts |
|               | `clip_ceiling`   | `0.35` | `[0.30, 0.38]` |
|               | `clip_slack`     | `0.03` | `[0.00, 0.06]` |
| **AugLag**    | `mu`             | `25.0` | `[8, 40]` |
|               | `lr_lambda`      | `0.10` | log-uniform `[0.02, 0.30]` |
|               | `max_deferral_rate` | `0.70` | hard deployment constraint |
| **Grouping**  | `support_distance_threshold` | `0.40` | hamming agglomerative cutoff |
|               | `min_family_size` | `10` | absorbs micro-families |
|               | `min_train_per_cluster` | `20` | KMeans hard floor |
|               | `min_group_train` | `20` | Stage-3 abs. fold threshold |
|               | `min_train_frac`  | `0.20` | Stage-3 fraction fold threshold |

---

## Reproducibility

* **Determinism.** Each HPO trial seeds `torch.manual_seed`, `numpy.random.seed`, and `torch.cuda.manual_seed_all` with `seed + trial.number`, and DataLoader shuffling uses a per-trial generator.
* **Seed config.** `run_hpo` enqueues a known-good seed configuration so the very first trial reproduces the notebook's manual baseline.
* **Trial ledger.** Every trial appends a JSON line with the full `(params, train_cfg, mcfg, al_cfg, metrics, scalar_score, utopia, elapsed_s)` to `hpo_ledger.jsonl`. Together with `optuna.create_study(storage="sqlite:///hpo.db", load_if_exists=True)` this lets you resume / re-analyse studies.
* **Submission config.** `configs/best_hpo.yaml` materialises the *exact* parameters of HPO trial 50, including the geometric clip anchors `{5: 0.3623, 7: 0.3263, 12: 0.3006}`.
* **Early NaN guards.** Both the per-batch loss and the final metric vector are NaN-checked; offending trials are pruned rather than silently corrupting the study.

---

## Citation

If you use this code or build on the methodology, please cite:

```bibtex
@inproceedings{mpd2router2026,
  title     = {MPD\textsuperscript{2}-Router: Hierarchical Constraint-Aware
               Multi-Expert Learning-to-Defer for Glaucoma Classification},
  author    = {The MPD\textsuperscript{2}-Router Authors},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

A `CITATION.cff` is provided for GitHub's "Cite this repository" link.

---

## License

This project is released under the [MIT License](LICENSE).
