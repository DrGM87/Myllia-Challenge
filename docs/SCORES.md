# Score Tracking — Echoes of Silenced Genes

**Best Public LB:** 4.08996 (V20 RUN=3) — Blind MLP + LOO (3 resamples) + 3-tier PB  
**Best CV:** 0.9806 (V20 RUN=4)  
**Competition Leader:** 4.45+  
**Metric:** W × max(0, Wcos) where W = sum of log₂(WMAE_base/WMAE_pred)

---

## Complete Score History

| Version | CV Score | LB Score | Date | Key Changes |
|---------|----------|----------|------|-------------|
| V4 Baseline | ~0.20 | — | 2026-01-30 | Basic ensemble (mean, knn, ridge, mlp) |
| V5.3 | 0.3220 | — | 2026-01-31 | Moderate knockout correction |
| V6.0 | -0.2408 ❌ | — | 2026-02-01 | Replogle K562 integration — val gene mapping bug |
| V6.11 | 0.6239 🏆 | 1.997 | 2026-02-02 | Blind MLP + regularization tuning |
| V7.3 | -0.5600 ❌ | — | 2026-02-03 | scGPT embeddings — FAILED |
| V9.1 | 0.6696 | 1.875 | 2026-02-08 | LightMLP, 3-layer, HP search overfitting |
| V10.1 | 0.8199 | 2.100 | 2026-02-08 | Raw scPerturb pseudo-bulk + MLP |
| V10.4 | 0.8643 | — | 2026-02-09 | + HGNC aliases, KNN v2 failed |
| V12.1 | 0.6973 | 1.988 | 2026-02-10 | Feature MLP + SVD — OOF inflation |
| V13.0 | 1.0379 | 1.730 ❌ | 2026-02-10 | HybridMLP + cosine loss — WORST LB EVER |
| V14a | 0.8302 | 3.021 | 2026-02-11 | V6 blind MLP + PseudoBulk + cosine_light |
| V15.0 | 0.7328 | 3.136 | 2026-02-12 | 3-tier PseudoBulk + WMAE |
| V15.1 | 0.8059 | 3.419 | 2026-02-16 | 3-tier PB + cosine_light — gains additive |
| V16 | 0.8656 🏆 | 3.584 🏆 | 2026-02-17 | Bisection HP tuning (λ=0.08, cos_right=0.0405) |
| V17 | 0.8108 | 3.634 🏆 | 2026-02-17 | LightMLP (H=384, D=2) — frozen architecture |
| V18 | 0.8458 | 3.623 | 2026-02-20 | Focal cosine loss — marginal regression |
| V19.1 | 0.2252 | 0.712 ❌❌ | 2026-02-21 | ResidualContextMLP — CATASTROPHIC |
| V20 RUN=4 | 0.9806 | 4.033 | 2026-02-22 | LOO (2 resamples) |
| **V20 RUN=3** | **0.9672** | **4.090** 🏆 | **2026-02-22** | **LOO (3 resamples) — ALL-TIME BEST** |
| V20 RUN=8 | 0.9692 | 4.068 | 2026-02-25 | LOO (3 resamples, N=40) |
| V21 | 0.9306 | 3.792 ❌ | 2026-03-01 | Targeted LOO (2000 genes) + threshold — regression |
| V22 | 0.9481 | 4.051 | 2026-03-03 | 4× LOO + per-channel — slight regression |
| V23 R1 | 0.9769 | 3.855 | 2026-03-05 | NN-Matched LOO — filtering hurt LB |
| V23 R2 | 0.8602 | 3.590 | 2026-03-05 | Residual LOO — worst V23 run |
| V23 R3 | 0.9585 | 3.985 | 2026-03-05 | Adaptive LOO |
| V23 R4 | 0.9770 | 3.772 | 2026-03-05 | Weighted LOO — high CV, bad LB |
| **V23 R5** | **0.9430** | **4.064** | **2026-03-05** | **Baseline + Replogle — 4th best LB!** |
| V23 R6 | 0.9679 | 3.874 | 2026-03-05 | Weighted + Replogle |

---

## All-Time LB Ranking

| # | Submission | LB | CV | Strategy |
|---|-----------|-----|-----|----------|
| 1 | V20 RUN=3 | **4.090** | 0.967 | Simple 3× LOO, no tricks |
| 2 | V20 RUN=8 | 4.068 | 0.969 | Same, N_ENSEMBLE=40 |
| 3 | V20 cv9642 | 4.066 | 0.964 | Same family |
| 4 | V23 R5 | 4.064 | 0.943 | Baseline + Replogle augmentation |
| 5 | V22 | 4.051 | 0.948 | 4× LOO + per-channel |
| 6 | V20 RUN=4 | 4.033 | 0.981 | 2× LOO |
| 7 | V23 R3 | 3.985 | 0.959 | Adaptive LOO |
| 8 | V23 R6 | 3.874 | 0.968 | Weighted + Replogle |
| 9 | V23 R1 | 3.855 | 0.977 | NN-Matched LOO |
| 10 | V21 | 3.792 | 0.931 | Targeted LOO + threshold |
| 11 | V23 R4 | 3.772 | 0.977 | Weighted LOO |
| 12 | V17 | 3.634 | 0.811 | LightMLP baseline |
| 13 | V18 | 3.623 | 0.846 | Focal cosine loss |
| 14 | V23 R2 | 3.590 | 0.860 | Residual LOO |
| 15 | V16 | 3.584 | 0.866 | Bisection HP tuning |
| 16 | V15.1 | 3.419 | 0.806 | cosine_light + 3-tier PB |
| 17 | V15 | 3.136 | 0.733 | 3-tier PB + WMAE |
| 18 | V14a | 3.021 | 0.830 | PseudoBulk + cosine_light |
| 19 | V14b | 2.750 | 0.750 | PseudoBulk + WMAE |
| 20 | V6.8 | 2.124 | 0.577 | Regularization tuning |
| 21 | V10 | 2.101 | 0.820 | scPerturb Replogle + MLP |
| 22 | V6.10 | 2.040 | 0.611 | Regularization tuning |
| 23 | V6.11 | 1.997 | 0.624 | Best V6 config |
| 24 | V12 | 1.988 | 0.697 | Feature MLP + SVD |
| 25 | V9.1 | 1.875 | 0.670 | HP search |
| 26 | V13 | 1.730 | 1.038 | HybridMLP + cosine loss |
| 27 | V19.1 | 0.712 | 0.225 | Context-conditioned MLP |

---

## Key Findings

### 1. CV→LB is Unreliable (and Sometimes Inverted)

Within V23, CV and LB were **negatively correlated** (r ≈ -0.85):

- R5 (lowest CV = 0.943) → best LB (4.064)
- R4 (highest CV = 0.977) → worst LB (3.772)

LOO quality filtering tricks inflate CV by overfitting to the train/val split structure but degrade generalization to unseen test perturbations.

### 2. Simplicity Wins

Every "clever" modification — external features (V10-V13), cosine loss optimization (V13), context conditioning (V19), LOO quality filtering (V23 R1/R4) — either didn't help or actively hurt LB. The best submission (V20 RUN=3) uses the simplest possible pipeline.

### 3. Data Augmentation is the Key Lever

| Innovation | LB Impact | Version |
|-----------|-----------|---------|
| PseudoBulk augmentation | +1.03 | V14 (2.00 → 3.02) |
| LOO synthetic perturbations | +0.46 | V20 (3.63 → 4.09) |
| Bisection HP tuning | +0.17 | V16 (3.42 → 3.58) |
| cosine_light loss | +0.28 | V15.1 (3.14 → 3.42) |
| 3-tier PB diversity | +0.12 | V15 (3.02 → 3.14) |

### 4. Replogle Data: Regularizer, Not Feature Source

Replogle K562 data **failed as features** (V10-V13: each usage made LB worse). But it **worked as a regularizer** (V23 R5: LB=4.064, 4th best ever). The difference: adding it as noisy training data prevents overfitting, while using it as input features creates domain shift.

### 5. Perturbation-Specific Models Always Fail

| Model | Perturbation-Specific Signal | LB |
|-------|------------------------------|-----|
| V17 (Blind MLP) | 8.7% | 3.63 |
| V18 (Blind + focal) | 8.7% | 3.62 |
| V13 (HybridMLP + Replogle PCA) | ~20% | 1.73 |
| V19.1 (ResidualContextMLP) | 44.1% | 0.71 |

The more perturbation-specific the model, the worse the LB. With only 80 training samples, any per-perturbation signal is memorization.

---

## V20 Detailed Results (Best Version)

### 9-Run Hyperparameter Sweep

| Run | Configuration | CV | LB |
|-----|---------------|------|------|
| RUN=0 | V17 baseline (no LOO) | 0.866 | — |
| RUN=1 | Mixup only (p=0.3) | 0.871 | — |
| RUN=2 | LOO (4 resamples) | 0.944 | — |
| **RUN=3** | **LOO (3 resamples)** | **0.967** | **4.090** 🏆 |
| RUN=4 | LOO (2 resamples) | 0.981 | 4.033 |
| RUN=5 | LOO (2×, weight=0.10) | 0.964 | — |
| RUN=6 | LOO (2×, weight=0.01) | 0.977 | — |
| RUN=7 | LOO (2×) + Mixup | 0.976 | — |
| RUN=8 | LOO (3×, N=40) | 0.969 | 4.068 |

### Key Insight: Inverse CV/LB on Resamples

- More LOO resamples → lower CV (dilutes GT signal) → **higher LB** (better generalization)
- The LB tests ~5,000 unseen genes. More synthetic data forces robust representations.
- LOO sample weight = 0.05 is the sweet spot.

---

## V23 Detailed Results (LOO Quality Isolation)

### 7 Runs — Systematic Isolation

| Run | Strategy | CV | LB | Finding |
|-----|----------|------|------|---------|
| R0 | Baseline (V20 repro) | 0.969 | — | Reference |
| R1 | NN-Matched LOO | 0.977 | 3.855 | Filtering hurt |
| R2 | Residual LOO | 0.860 | 3.590 | Wrong approach |
| R3 | Adaptive LOO | 0.959 | 3.985 | Mild weighting OK |
| R4 | Weighted LOO | 0.977 | 3.772 | Best CV = worst LB |
| R5 | Baseline + Replogle | 0.943 | 4.064 | Replogle as regularizer works! |
| R6 | Weighted + Replogle | 0.968 | 3.874 | Weighted still hurts |

### Critical Discovery

**CV→LB is inverted for V23.** Pearson r ≈ -0.85.

Replogle augmentation on the **simple baseline** (R5) produced our 4th best LB ever, despite having the **lowest** non-residual CV. This completely reversed the V10-V13 narrative that "Replogle always hurts."

---

## Final Submission (V_FINAL)

After organizers released `pert_ids_all.csv` with all 120 perturbation identities:

| Run | Strategy | Description |
|-----|----------|-------------|
| A | Pure V20 RUN=3 | Exact V20 reproduction for all 120 perts — lowest risk |
| B | V20 + Replogle | Same + Replogle K562 regularization — potential upside |
| C | 50/50 Ensemble | Blend of A + B — hedged bet |

All 120 rows contain **real predictions** (no zeros for rows 61-120).

---

*Last updated: March 2026*
