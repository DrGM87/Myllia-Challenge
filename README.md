# Echoes of Silenced Genes — Kaggle Competition Solution

<p align="center">
  <img src="assets/banner.png" alt="Competition Banner" width="800"/>
</p>

> **Predicting how human cancer cells respond to CRISPR-interference perturbations**

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)](https://kaggle.com/competitions/echoes-of-silenced-genes)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

---

## Table of Contents

- [Competition Overview](#competition-overview)
- [Our Approach — The Full Journey](#our-approach--the-full-journey)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Lessons Learned](#lessons-learned)
- [Citation](#citation)

---

## Competition Overview

**[Myllia | Echoes of Silenced Genes: A Cell Challenge](https://kaggle.com/competitions/echoes-of-silenced-genes)** is a Kaggle competition organized by [Myllia Biotechnology](https://myllia.com/) (Vienna, Austria), announced at the High-Content CRISPR Screening Conference in March 2026.

### The Task

Given single-cell RNA-seq data from CRISPR-interference (CRISPRi) experiments on a human cancer cell line:

- **Training data:** 80 known gene perturbations (mean expression + single-cell-level data)
- **Prediction target:** Delta expression (log2 fold-change) of **5,127 genes** for **120 unseen perturbations**
  - 60 perturbations scored on the public leaderboard
  - 60 perturbations scored for the **final ranking** (revealed one week before deadline)

### The Metric

The competition uses a novel two-component metric:

```
Score = W × max(0, Wcos)
```

- **W** — Sum of log₂(WMAE_baseline / WMAE_prediction) across perturbations, capped at 5 per row. Measures improvement over a mean-baseline prediction.
- **Wcos** — Weighted cosine similarity with smoothstep gating. Measures directional alignment, focusing on genes with large effects.

Predictions must beat the baseline (average of 80 training deltas) to score positively.

### The Challenge

With only **80 training perturbations** and **5,127 target genes**, this is a severe low-data regime. The 120 test perturbations target completely disjoint genes — zero overlap with training. Models must generalize to unseen gene knockouts using limited biological signal.

---

## Our Approach — The Full Journey

This project spanned **24 notebook versions** over 6 weeks, each systematically testing a hypothesis. Below is the complete chronological story of how we went from negative scores to our best result of **LB = 4.090**.

### Phase 1: Foundation (V2–V5) — *"What even works?"*

**V2–V4 (Baseline Ensembles):** Started with a basic approach — mean predictor, KNN, Ridge regression, and a small MLP. The ensemble scored ~+0.20 CV. Key discovery: only the MLP contributed positively.

**V5.0–V5.3 (Knockout Correction):** Discovered that the self-targeting gene (the gene being knocked out) needs special treatment. CRISPRi suppresses the target gene — setting this to the observed mean knockout delta (+0.07 CV improvement) was our first real insight. **V5.3 reached +0.32 CV.**

> **Lesson:** Knockout correction is essential and remained in every subsequent version.

### Phase 2: External Data Experiments (V6–V13) — *"Can Replogle data help?"*

**V6.0–V6.11 (Blind MLP + Regularization Tuning):** Stripped down to a pure blind MLP (gene embedding → 2-layer MLP → 5127 outputs). Systematic regularization tuning (weight decay, dropout, ensemble size) pushed from +0.42 to **+0.62 CV, LB = 2.00**. The "blind" architecture — predicting the same delta pattern regardless of which gene is knocked out — proved surprisingly effective.

**V7.3 (scGPT Embeddings) — FAILED ❌:** Attempted to use pretrained scGPT gene embeddings as input features. All runs scored **negative** (worse than baseline). The pretrained embeddings were trained for cell representation, not perturbation response — they actively hurt predictions.

**V9.1 (Hyperparameter Search):** Deeper 3-layer MLP with extensive HP search reached +0.67 CV but **LB = 1.87** — worse than V6 despite higher CV. First evidence of CV/LB divergence from HP overfitting.

**V10.0–V10.4 (Replogle K562 Data Integration):** Downloaded raw single-cell CRISPRi data from Replogle et al. (2022) via scPerturb:
- V10.0 (Harmonizome z-scores): Failed — too processed, lost perturbation signal
- V10.1 (Raw pseudo-bulk): Breakthrough! OOF = 4.95 with Replogle as a blended predictor
- V10.2–V10.4: Added genome-wide Replogle (9,867 perturbations), HGNC gene aliases

**V11 (Global Calibration):** Replaced per-gene OLS calibration (10,254 params on 80 samples!) with global 4-parameter calibration. Added Norman CRISPRa data. Replogle still produced negative individual scores — it only helped through ensemble blending.

**V12 (Feature Engineering):** Feature MLP, Ridge, CIPHER, SVD denoising. OOF = 5.13 but **LB = 1.99** — SVD and alpha scaling inflated OOF without improving generalization.

**V13 (HybridMLP + Cosine Loss) — WORST LB EVER ❌:** Highest CV yet (+1.04) but **LB = 1.73** — our worst submission. The cosine loss directly optimized the competition metric on training data, creating severe metric-specific overfitting. Replogle PCA features added via a gating mechanism made it worse.

> **Critical Lesson (V6–V13):** Every attempt to add complexity — external features, sophisticated losses, architectural changes — DEGRADED leaderboard performance. The blind MLP remained king. **CV is an unreliable guide when it inflates from training-specific patterns.**

### Phase 3: The PseudoBulk Revolution (V14–V16) — *"Data augmentation, not architecture"*

**V14 (PseudoBulk + Loss Grid):** Returned to the V6 blind MLP but added **PseudoBulk augmentation** from single-cell data. By bootstrapping cells per perturbation, we generated 880 augmented training samples from 80. Combined with cosine_light loss (λ=0.05):
- V14a (cosine_light + PB): **LB = 3.02** — massive +1.03 improvement!
- V14b (WMAE + PB): LB = 2.75

**V15 (3-Tier PseudoBulk):** Introduced three tiers of PseudoBulk diversity:
1. **Tier 1:** Full bootstrap (10 resamples per perturbation, all cells)
2. **Tier 2:** Per-channel means (using batch/channel information)
3. **Tier 3:** Half-cell bootstrap (5 resamples, 50% of cells)

Quality gate: only keep samples with correlation > 0.3 to ground truth. **LB = 3.14** with WMAE loss.

**V15.1 (3-Tier PB + cosine_light):** Combined V15's PseudoBulk with V14's cosine_light loss. **LB = 3.42** — gains were perfectly additive (+0.28 from loss + +0.39 from PB diversity).

**V16 (Bisection HP Tuning):** Systematic bisection search over 4 hyperparameters (176 evaluations):
- `COSINE_LAMBDA`: 0.05 → **0.08** (stronger cosine nudge)
- `COS_RIGHT`: 0.2 → **0.0405** (much tighter smoothstep gate — biggest single lever)
- `GT_UPWEIGHT`: 3.0 → **4.5** (ground truth samples weighted more than PB)
- `N_ENSEMBLE`: 40 → **30**

Result: **LB = 3.58** — new all-time best.

> **Lesson:** The path to improvement was data augmentation (PseudoBulk) and loss tuning — never architecture changes.

### Phase 4: LOO Augmentation Breakthrough (V17–V20) — *"Breaking the 4.0 barrier"*

**V17 (Refined Architecture):** Switched to `LightMLP` (Embedding → BatchNorm → 2-layer MLP, H=384). The frozen architecture that would carry us to our best score. **LB = 3.63.**

**V18 (Focal Cosine Loss):** Attempted focal-style loss weighting. **LB = 3.62** — marginal regression. Confirmed: loss engineering has diminishing returns.

**V19 (Context-Conditioned MLP) — CATASTROPHIC ❌:** ResidualContextMLP with 5,192-dim context vectors. **LB = 0.71** — worst ever. The model made 44% perturbation-specific predictions (vs V17's 8.7% shared). With only 80 training samples, any perturbation-specific signal is pure memorization.

> **The more perturbation-specific the model, the worse the LB score. Proven across 4 versions (V7, V13, V18, V19).**

**V20 (Leave-One-Out Synthetic Augmentation) — THE BREAKTHROUGH 🏆:**

Introduced **genome-wide LOO synthetic perturbations**: for each of the 5,127 genes, find control cells where that gene is naturally low-expressed (5th percentile), compute pseudo-bulk deltas — simulating what a knockout might look like. With 3 bootstrap resamples per gene, this generated **~15,000 synthetic perturbation samples**.

| Run | LOO Resamples | LOO Samples | CV | LB |
|-----|---------------|-------------|------|------|
| RUN=0 | 0 (baseline) | 0 | 0.866 | — |
| RUN=4 | 2 | ~10k | **0.981** | 4.033 |
| **RUN=3** | **3** | **~15k** | **0.967** | **4.090** 🏆 |
| RUN=2 | 4 | ~20k | 0.944 | — |
| RUN=8 | 3 (N=40) | ~15k | 0.969 | 4.068 |

**V20 RUN=3 scored LB = 4.090** — our all-time best, breaking the 4.0 barrier by a wide margin.

**Key insight: Inverse CV/LB correlation.** More LOO resamples = lower CV (dilutes GT signal) but higher LB (better generalization to unseen genes). The LB tests ~5,000 unseen genes — more synthetic data forces the model to learn robust representations.

### Phase 5: Diminishing Returns (V21–V23) — *"Trying to beat V20"*

**V21 (Targeted LOO + Hard Threshold):** Reduced LOO to top 2,000 genes + zeroed predictions below 0.02. **LB = 3.79** — regression. LOO coverage is non-negotiable; thresholding destroys small but correct predictions.

**V22 (Per-Channel LOO):** 4× LOO resamples with per-channel normalization. **LB = 4.05** — slight regression. Per-channel splits reduce cells per group, producing noisier bootstraps.

**V23 (LOO Quality Isolation — 7 Runs):** Systematic isolation of LOO quality strategies:

| Run | Strategy | CV | LB | Finding |
|-----|----------|------|------|---------|
| R5 | Baseline + Replogle | 0.943 | **4.064** | **4th best ever!** Replogle as regularizer works |
| R3 | Adaptive LOO | 0.959 | 3.985 | Mild per-gene weighting OK |
| R6 | Weighted + Replogle | 0.968 | 3.874 | Weighted LOO hurts even with Replogle |
| R1 | NN-Matched LOO | 0.977 | 3.855 | Quality filtering hurts LB |
| R4 | Weighted LOO | **0.977** | 3.772 | **Highest CV = worst LB!** |
| R2 | Residual LOO | 0.860 | 3.590 | Residual approach fails |

**V23 proved CV→LB is inverted (r ≈ -0.85).** LOO quality tricks inflate CV by overfitting to the train/val split but degrade generalization. The simple V20 baseline remains best.

**Paradigm shift:** Replogle data works as a **regularizer** (adds noise that prevents overfitting), NOT as a feature source. V23 R5 reversed the entire V10–V13 narrative.

### Phase 6: Final Submission (V_FINAL) — *"Predict all 120 perturbations"*

One week before the deadline, the organizers released `pert_ids_all.csv` — identities of all 120 perturbations. **All existing submissions had zeros for rows 61–120** (the final ranking rows). We had to generate real predictions.

**V_FINAL** used the proven V20 RUN=3 configuration (frozen) to predict all 120 perturbations:

- **Run A:** Pure V20 RUN=3 reproduction for all 120 perts (lowest risk)
- **Run B:** V20 RUN=3 + Replogle K562 augmentation (regularization upside)
- **Run C:** 50/50 ensemble of A + B (hedged bet)

Every cell includes resume/checkpoint logic for Google Colab crash recovery.

---

## Key Results

### Final Leaderboard Rankings (Public LB, pert_1–60)

| # | Version | Public LB | CV Score | Strategy |
|---|---------|-----------|----------|----------|
| 🥇 | **V20 RUN=3** | **4.090** | 0.967 | Blind MLP + 3× LOO + 3-tier PB |
| 2 | V20 RUN=8 | 4.068 | 0.969 | Same, N_ENSEMBLE=40 |
| 3 | V20 cv9642 | 4.066 | 0.964 | Same family |
| 4 | V23 R5 | 4.064 | 0.943 | Baseline + Replogle regularization |
| 5 | V22 | 4.051 | 0.948 | 4× LOO + per-channel |
| 6 | V20 RUN=4 | 4.033 | 0.981 | 2× LOO |

**Competition context:** 1st place scored 4.45. Our best = 4.090.

### Score Progression Over Time

```
LB Score
  4.1 |                                              ★ V20 RUN=3 (4.090)
  4.0 |                                            ●●● V20 family
  3.5 |                             ■ V16 (3.58)  ■ V17 (3.63)
  3.0 |                    ▲ V14a (3.02)  ▲ V15.1 (3.42)
  2.5 |
  2.0 |  ● V6.11 (2.00)  ● V10 (2.10)  ● V12 (1.99)
  1.5 |                                   ✗ V13 (1.73)
  1.0 |
  0.5 |                                                ✗ V19.1 (0.71)
  0.0 |----+----+----+----+----+----+----+----+----+----
       V5   V6   V7   V9  V10  V12  V13  V14  V16  V20
```

---

## Architecture

### Model: Blind LightMLP (V17)

The winning architecture is remarkably simple:

```
Gene Index (int) → Embedding(5128, 64) → BatchNorm → ReLU → Dropout(0.5)
                                        → Linear(64, 384) → BatchNorm → ReLU → Dropout(0.5)
                                        → Linear(384, 384) → BatchNorm → ReLU → Dropout(0.5)
                                        → Linear(384, 5127)
```

- **Blind:** Takes only a gene index as input — no perturbation-specific features
- **Shared prediction:** Outputs the same delta pattern for any gene (except KO correction)
- **~2.5M parameters**, 30-model ensemble

### Loss: cosine_light

```python
loss = WMAE(pred, target, weights) - λ × cosine_similarity(pred × smoothstep_gate, target × smoothstep_gate)
```

- `λ = 0.08`, `cos_right = 0.0405` (bisection-tuned in V16)
- `GT_UPWEIGHT = 4.5` (ground truth samples weighted 4.5× vs augmented data)

### Data Augmentation Pipeline

| Source | Samples | Weight | Description |
|--------|---------|--------|-------------|
| Ground Truth | 80 | 4.5× | 80 known perturbation deltas |
| 3-Tier PseudoBulk | ~1,500 | 1.0× | Bootstrap resamples from single-cell data |
| LOO Synthetic | ~15,000 | 0.05× | Genome-wide leave-one-out perturbation simulation |
| Replogle K562 (Run B) | ~2,000 | 0.03× | External CRISPRi data as regularizer |

### Post-Processing

**Knockout (KO) correction only.** For each perturbation, force the self-targeting gene's prediction to the mean observed KO delta (−0.572). No other post-processing — no SVD, no alpha scaling, no thresholding.

---

## Repository Structure

```
GitHub/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── .gitignore                   # Excludes large data, credentials, checkpoints
│
├── notebooks/
│   ├── Myllia_Challenge_V_FINAL.ipynb    # Final submission notebook (all 120 perts)
│   ├── Myllia_Challenge_V20_LOO x2.ipynb # Best model (V20 RUN=3, LB=4.090)
│   └── Myllia_Challenge_V23_LOO_Quality.ipynb  # V23 7-run quality isolation
│
├── submissions/
│   ├── submission_v20_loo_best_4.090.csv           # Best public LB (pert_1-60)
│   └── submission_vfinal_runA_all_120_perts.csv    # Final submission (all 120)
│
├── docs/
│   ├── COMPETITION.md           # Full competition description & metric details
│   ├── SCORES.md                # Complete score tracking across all 24 versions
│   ├── V_FINAL_Blueprint.md     # Final submission strategy document
│   └── V23_Blueprint.md         # V23 LOO quality isolation experiment design
│
├── analysis/
│   ├── ensemble_analysis.py     # Cross-version ensemble analysis + HTML report
│   └── v23_submission_analysis.py  # V23 run comparison + LB prediction model
│
└── assets/
    └── banner.png               # (placeholder for competition banner image)
```

### Key Files

- **`notebooks/Myllia_Challenge_V_FINAL.ipynb`** — The final production notebook. 14 cells with resume/checkpoint support. Predicts all 120 perturbations with KO correction. Generates Run A (baseline), Run B (+Replogle), Run C (ensemble).

- **`notebooks/Myllia_Challenge_V20_LOO x2.ipynb`** — The notebook that produced our best public LB score (4.090). Contains the full LOO augmentation pipeline and 9-run hyperparameter sweep.

- **`docs/SCORES.md`** — Complete chronological record of every version, every CV score, every LB score, and detailed post-mortem analysis.

---

## How to Run

### Requirements

- Google Colab (GPU runtime — A100 or T4)
- Kaggle account (for data download)
- Google Drive (for checkpointing)

### Quick Start

1. Upload `Myllia_Challenge_V_FINAL.ipynb` to Google Colab
2. Run cells sequentially — each cell has resume logic
3. Enter your Kaggle API token when prompted (Cell 2)
4. The notebook will:
   - Download competition data from Kaggle
   - Generate PseudoBulk augmentation from single-cell data
   - Generate LOO synthetic perturbations (~15,000 samples)
   - (Optional) Download and process Replogle K562 data for Run B
   - Train 30-model ensembles for Run A and Run B
   - Generate 120-row submission files with KO correction
   - Package everything into a downloadable ZIP

**Total runtime:** ~6–8 hours on A100, ~12–16 hours on T4.

### Resume After Crash

Every cell checkpoints to Google Drive (`/content/drive/MyDrive/myllia_vfinal/`). If Colab crashes, simply restart and re-run — completed stages are automatically skipped.

---

## Lessons Learned

### What Worked

1. **Simplicity wins.** A blind MLP that ignores perturbation identity outperformed every "smart" architecture we tried. With only 80 training samples, perturbation-specific predictions are pure memorization.

2. **Data augmentation > architecture.** PseudoBulk (+1.03 LB), LOO synthetic perturbations (+0.46 LB), and loss tuning (+0.17 LB) drove all gains. Architecture changes either helped marginally or actively hurt.

3. **LOO synthetic augmentation is transformative.** Simulating knockouts by selecting naturally low-expressing cells from control data generated ~15,000 synthetic perturbations covering all 5,127 genes — enabling generalization to completely unseen perturbations.

4. **Inverse CV/LB correlation is real.** More augmentation data lowers CV (dilutes ground truth signal) but raises LB (better generalization). Optimizing for CV alone would have led us astray.

5. **Knockout correction is a free lunch.** Simply forcing the self-targeting gene to the observed mean KO delta consistently improved scores with zero downside.

### What Failed

1. **External data as features (V10–V13, V19).** Replogle PCA features, scGPT embeddings, context vectors — all inflated CV while degrading LB. Coverage gaps (47% gene overlap) and domain shift made features unreliable.

2. **Perturbation-specific models (V7, V13, V19).** The more a model tried to learn per-perturbation patterns, the worse it performed. V19 (44% perturbation-specific signal) scored 0.71 vs V17 (8.7% specific) at 3.63.

3. **Post-processing tricks (V12, V21).** SVD denoising, alpha scaling, hard thresholding — all inflated OOF/CV without improving generalization. The competition metric rewards accurate predictions, not clever rescaling.

4. **LOO quality filtering (V23).** NN-matching, adaptive weighting, quality scoring — all hurt LB despite improving CV. The simple 5th-percentile baseline was optimal.

5. **Replogle as a predictor (V10–V13).** Failed repeatedly. But Replogle as a **regularizer** (V23 R5) works — it adds beneficial noise that prevents overfitting.

### The Counter-Intuitive Truth

> The winning strategy for this competition is to predict **the same thing for every perturbation** — a shared delta pattern learned from the covariance structure of the training data. The only per-perturbation element is a simple lookup correction for the knocked-out gene. Everything else is noise.

---

## Citation

If you use any part of this work, please cite the competition:

```bibtex
@misc{echoes-of-silenced-genes,
    author = {Myllia Biotechnology},
    title = {Myllia | Echoes of Silenced Genes: A Cell Challenge},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/echoes-of-silenced-genes}},
    note = {Kaggle}
}
```

### Key References

- **Replogle et al. (2022)** — Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell*, 185(14), 2559-2575. [scPerturb data](https://zenodo.org/records/13350497)
- **Mejia et al.** — Weighted MAE metric design inspiration
- **Myllia Biotechnology** — Competition organizers, metric design, and data generation

---

## Acknowledgments

- **Myllia Biotechnology** for organizing this competition and designing a thoughtful evaluation metric
- **Google Colab** for providing GPU compute (A100/T4)
- **scPerturb / Zenodo** for hosting the Replogle K562 CRISPRi datasets
- The **Virtual Cell Research Community** for discussions and feedback

---

*Built with persistence, 24 notebook iterations, and the humbling realization that in biology, simple models often beat complex ones.*
