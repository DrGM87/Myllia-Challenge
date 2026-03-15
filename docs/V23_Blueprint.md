# V23 Blueprint — LOO Quality Maximization + Replogle Augmentation

**Date**: 2026-03-04  
**Current Best**: V20 RUN=3, LB=4.08996  
**Top Score**: 5.12  
**Gap to #1**: 1.03 (25%)  
**Compute Budget**: 6-7 hours per run (360-420 min)  
**Notebook**: `Myllia_Challenge_V23_LOO_Quality.ipynb`

---

## Executive Summary

V22 proved that per-channel LOO normalization **hurts** (LB=4.051 vs 4.090 baseline).
V21 proved that reducing LOO gene coverage **hurts** (LB=3.792).
V20 proved that more LOO resamples **help** LB but hurt CV (inverse correlation).

**The untapped lever is LOO QUALITY.** Current LOO uses the simplest possible cell selection
(bottom 5th percentile by raw expression). This produces noisy synthetic knockouts because:
1. "Naturally low" cells may be low for confounding reasons (cell cycle, batch)
2. A hard percentile cutoff is arbitrary and suboptimal per gene
3. No matching is done to isolate the gene-specific effect

V23 tests **4 smarter LOO selection methods** + **Replogle external data augmentation**,
then combines the best into a final submission.

---

## What is LOO (Leave-One-Out Synthetic Perturbation)?

LOO simulates gene knockouts using natural expression variation in control cells:

1. We have ~1026 unperturbed control cells
2. For target gene G, find cells where G is naturally very low ("pseudo-knockout")
3. Compute: `delta = mean(low_G_cells) - mean(other_cells)` across all 5127 genes
4. This approximates the downstream effect of knocking out gene G
5. Repeat with bootstrap resampling → multiple synthetic training samples per gene

**Why it's noisy**: The "naturally low" cells aren't true knockouts. Their low expression
of gene G may correlate with low expression of other genes (confounding), cell cycle state,
or batch effects. The 4 quality improvements below address these confounds.

**Current LOO stats** (from online analysis):
- LOO↔GT correlation: 0.051 (genome-wide, basically zero)
- LOO↔GT KO delta correlation: 0.724 (strong self-targeting signal only)
- LOO/GT magnitude ratio: 0.952 (well-calibrated at 5th percentile)

---

## Frozen Components (Do NOT Change)

| Component | Value | Source |
|-----------|-------|--------|
| Architecture | V17 LightMLP (H=384, D=2) | Proven across V14-V22 |
| Loss | cosine_light (λ=0.08, cos_right=0.0405) | V16 bisection optimum |
| GT Upweight | 4.5 | V16 bisection optimum |
| N_MLP_ENSEMBLE | 30 | V16/V20 optimum |
| MLP_EPOCHS | 300 | V17 standard |
| LOO_N_RESAMPLES | 3 | V20 RUN=3 proven best LB |
| LOO_SAMPLE_WEIGHT | 0.05 | V20 weight sweep optimum |
| LOO_PERCENTILE | 5.0 (baseline, varies per run) | V20 default |
| PER_CHANNEL_LOO | False | V22 proved harmful |
| 3-Tier PseudoBulk | Full bootstrap + per-channel + half-cell | V15/V20 standard |
| KO Correction | Applied post-prediction | All versions |
| Hard Threshold | None | V21 proved harmful |
| Mixup | False | V20 proved harmful with LOO |
| CV | 5-Fold | Standard |

---

## The 6 Runs

### Run 0: Baseline (Pooled LOO, 3x resamples) — REFERENCE

**Purpose**: Fair comparison baseline using V20 RUN=3 config on the V23 codebase.
All other runs are compared against this.

**LOO Method**: Current pooled percentile (bottom 5% of cells by raw expression of gene G).
```
For gene G:
  1. Rank all 1026 control cells by expression of G
  2. Select bottom 5% (~51 cells) as "low-G" group
  3. Bootstrap resample low-G cells → compute mean expression (5127 genes)
  4. Compute mean of remaining cells → baseline
  5. Delta = low_G_mean - baseline_mean
  6. Repeat 3x (resamples)
```

**Expected CV**: ~0.967 (matching V20 RUN=3)

---

### Run 1: Nearest-Neighbor Matched LOO ⭐

**Purpose**: Remove confounding from cell state by matching each "low-G" cell
to a similar "normal-G" cell, isolating the gene-specific effect.

**LOO Method**: Propensity-score-style matching from causal inference.
```
For gene G:
  1. Compute PCA on all 1026 control cells (50 components, EXCLUDING gene G)
  2. Split cells into low-G (bottom 5%) and normal-G (rest)
  3. For EACH low-G cell:
     a. Find its nearest neighbor in normal-G group (by Euclidean distance in PCA space)
     b. Compute pair_delta = low_cell_expression - matched_normal_expression
  4. Average all pair_deltas → matched LOO delta for gene G
  5. Repeat 3x with bootstrap resampling of the low-G cells
```

**Why better**: If a low-G cell is also low in genes Y, Z (because it's in G1 phase),
its matched normal cell will ALSO be in G1 phase — canceling the cell-state confound.
The remaining delta is specific to gene G's low expression.

**Implementation notes**:
- PCA is computed ONCE on all 1026 cells (fast)
- For each gene G, we only recompute the cell ranking and matching (not PCA)
- Use `sklearn.neighbors.NearestNeighbors` with k=1 for speed
- Precompute PCA without gene G by zeroing column G in the PCA-transformed space
  (approximation: subtract gene G's contribution via `X_pca - X[:,G:G+1] @ pca.components_[:,G:G+1].T`)
- Fallback: if < 10 low-G cells, use baseline pooled LOO

**Estimated overhead**: ~2-3 min for PCA + matching setup (negligible vs training time)

---

### Run 2: Residual-Based LOO ⭐

**Purpose**: Select cells where gene G is lower THAN EXPECTED given the cell's
overall state, not just cells where G happens to be low.

**LOO Method**: Regression residual selection.
```
For gene G:
  1. Fit a quick linear model on control cells: expr(G) ~ PC1 + PC2 + ... + PC20
     (PCA computed once, reused across all genes)
  2. Compute residuals for each cell: residual = actual_expr(G) - predicted_expr(G)
  3. Select cells with the largest NEGATIVE residuals (bottom 5% by residual)
     → these cells have gene G much lower than their overall state predicts
  4. Bootstrap resample selected cells → compute mean
  5. Compute baseline from cells with residuals near 0 (middle 50%)
  6. Delta = selected_mean - baseline_mean
  7. Repeat 3x
```

**Why better**: A cell may have low gene G simply because it's a quiet cell overall.
Residual selection finds cells that are specifically low in gene G while being normal
in everything else — a closer match to a true knockout.

**Implementation notes**:
- PCA: 20 components (lighter than Run 1's 50, sufficient for regression)
- Linear regression: `np.linalg.lstsq` per gene (fast, 1026×20 matrix)
- Baseline: middle 50% by residual (not all remaining cells) to avoid contamination
- Fallback: if < 10 cells pass residual threshold, use baseline pooled LOO

**Estimated overhead**: ~1-2 min total

---

### Run 3: Adaptive Percentile LOO

**Purpose**: Use gene-specific percentile thresholds instead of fixed 5% for all genes.
Bimodal genes (clear on/off states) should use their natural "off" population.
Unimodal genes should use a tighter threshold.

**LOO Method**: Gaussian mixture model per gene.
```
For gene G:
  1. Fit a 2-component Gaussian Mixture Model to control cell expressions of G
  2. Classify:
     a. BIMODAL: if the two components are well-separated
        (|mean1 - mean2| > 1.5 * max(std1, std2))
        → Select ALL cells in the lower component as "low-G"
        → This could be 5-40% of cells depending on the gene
     b. UNIMODAL: components overlap heavily
        → Use strict 2nd percentile (tighter than baseline 5th)
  3. Bootstrap resample selected cells → compute mean
  4. Delta = selected_mean - rest_mean
  5. Repeat 3x
```

**Why better**: Gene expression distributions vary wildly:
- Some genes are bimodal (clearly "on" in some cells, "off" in others)
  → The "off" population is a natural pseudo-knockout, regardless of percentile
- Some genes are unimodal with long tails
  → 5th percentile may include cells that still express G at moderate levels
  → 2nd percentile gives sharper knockouts

**Implementation notes**:
- `sklearn.mixture.GaussianMixture(n_components=2)` per gene
- Bimodality test: separation ratio = |μ1-μ2| / max(σ1,σ2) > 1.5
- Cache GMM results (computed once, used for all 3 resamples)
- Fallback: if GMM fails to converge, use baseline 5th percentile
- Expected: ~30-40% of genes bimodal, ~60-70% unimodal (using 2nd percentile)

**Estimated overhead**: ~3-5 min for 5127 GMM fits

---

### Run 4: Expression-Weighted LOO

**Purpose**: Replace the hard percentile cutoff with continuous weighting.
Cells with lower expression get more weight in the "knockout" mean.

**LOO Method**: Soft weighting by expression level.
```
For gene G:
  1. For each control cell, compute:
     weight(cell) = max(0, median_expr(G) - cell_expr(G)) / (median_expr(G) + 1e-6)
     → Cells below median get positive weight proportional to how low they are
     → Cells above median get weight = 0
  2. Normalize weights: w = w / sum(w)
  3. Compute weighted_low_mean = sum(w * cell_expressions) across all cells
  4. Compute baseline_mean = mean(cells where weight=0) — i.e., cells above median
  5. Delta = weighted_low_mean - baseline_mean
  6. For resampling: add noise by resampling weights (Dirichlet perturbation)
     → w_resampled = w * Dirichlet(alpha=10) → renormalize
  7. Repeat 3x
```

**Why better**: No arbitrary cutoff. Cells with expression=0 for gene G contribute
most to the "knockout" signal. Cells with expression slightly below median contribute
proportionally less. This is smoother and less sensitive to the percentile choice.

**Implementation notes**:
- Vectorized: compute all weights for gene G in one operation
- Dirichlet resampling: `np.random.dirichlet(alpha * np.ones(n_weighted))` for bootstrap diversity
- Alpha=10 gives moderate perturbation (enough diversity across resamples without destroying signal)
- Fallback: if fewer than 5 cells have positive weight (gene expressed in nearly all cells), 
  use baseline percentile

**Estimated overhead**: Negligible (faster than baseline — no sorting/selection step)

---

### Run 5: Replogle Blind Augmentation ⭐⭐⭐

**Purpose**: Add ~2000+ REAL knockout delta profiles from Replogle K562 CRISPRi
as additional training samples for the blind MLP.

**Key insight**: V10-V13 used Replogle as MODEL INPUT (features, PCA) → failed due to
47% coverage gap on test genes. But the blind MLP doesn't use perturbation-specific
features — it maps `gene_index → delta_vector`. Feeding Replogle deltas as additional
training samples avoids the coverage problem entirely.

**Data Source**: 
- Replogle Weissman 2022, K562 CRISPRi (same cell line as competition!)
- Essential: ~2057 perturbations
- Genome-Wide (GWPS): ~7810 additional perturbations
- Download from Zenodo (scPerturb): ~1.5 GB + ~8.8 GB

**Processing Pipeline**:
```
1. Download ReplogleK562essential.h5ad + ReplogleK562gwps.h5ad from Zenodo
2. For each dataset:
   a. Normalize: log2(1 + count/total * 10000) — MUST match competition normalization
   b. Identify control cells (non-targeting)
   c. Compute per-perturbation pseudo-bulk delta = mean(pert_cells) - mean(ctrl_cells)
3. Map Replogle gene space → competition 5127 genes via:
   a. Direct gene name matching
   b. HGNC alias matching (adds ~114 genes)
4. For each Replogle perturbation that maps to one of our 5127 gene indices:
   a. Extract the 5127-dim delta vector
   b. Assign gene_index = position of the perturbed gene in gene_names
   c. Add as training sample with weight = REPLOGLE_SAMPLE_WEIGHT
5. Combine with GT + PB + baseline LOO data
```

**Config for Replogle augmentation**:
```python
REPLOGLE_SAMPLE_WEIGHT = 0.03  # Lower than LOO (0.05) due to domain shift (K562 vs competition cells)
REPLOGLE_USE_ESSENTIAL = True
REPLOGLE_USE_GWPS = True       # Include genome-wide (larger coverage, noisier)
REPLOGLE_MIN_CELLS = 10        # Minimum cells per perturbation
REPLOGLE_GWPS_DISCOUNT = 0.75  # GWPS deltas weighted 75% of essential (more noise)
```

**Expected training data composition** (per fold):
```
64 GT samples        (weight=1.0 × gt_upweight=4.5)
1216 PB samples      (weight=1.0)
~15,000 LOO samples  (weight=0.05)
~2,000+ Replogle     (weight=0.03, essential=0.03, gwps=0.03×0.75)
```

**Why this is the highest-ceiling idea**:
- Replogle K562 CRISPRi = REAL knockouts in a RELATED cell line
- ~2000+ real knockout profiles vs ~5000 synthetic LOO approximations
- The blind MLP sees them as just more (gene_idx, delta) pairs
- No coverage problem: the MLP is blind to perturbation identity
- V10 showed Replogle correlation with competition GT: r=0.959 for matched genes
- Even with domain shift, real knockouts > synthetic approximations

**Critical normalization note**: V13 discovered that Replogle data uses ln(1+x) normalization
while competition uses log2(1+x). Must convert: divide sparse data by ln(2).

**Download strategy**: 
- Essential only: ~1.5 GB, ~5 min download, ~2057 perturbations
- GWPS: ~8.8 GB, ~20 min download, ~7810 additional perturbations
- If GWPS download fails/times out, proceed with Essential only
- Cache processed deltas to .npz file for resume capability

---

### Run 6: Best LOO + Replogle Combined 🏆

**Purpose**: Combine the winning LOO method (from Runs 1-4 comparison) with
Replogle augmentation (Run 5) for the final submission.

**Config**: 
```
LOO method = winner of Runs 1-4 (by CV score)
+ Replogle augmentation from Run 5
+ All frozen components from above
```

If Run 5 (Replogle alone) beats all LOO variants, Run 6 still combines the best
LOO with Replogle, because they provide complementary signal:
- LOO: synthetic, covers all 5127 genes, noisy but complete
- Replogle: real knockouts, covers ~2000 genes, high quality but partial

---

## Checkpoint & Resume Strategy

Every run saves state at key points to enable resume after crashes:

### Checkpoints saved:
```
/content/drive/MyDrive/myllia_v23/
├── run{N}_config.json              # Full config for reproducibility
├── run{N}_loo_samples.npz          # LOO synthetic data (gene_idx, deltas, weights)
├── run{N}_replogle_deltas.npz      # Replogle processed data (Run 5/6 only)
├── run{N}_fold{K}_models.pt        # Trained model states per fold
├── run{N}_fold{K}_preds.npy        # Validation predictions per fold
├── run{N}_cv_results.json          # CV scores per fold
├── run{N}_submission.csv           # Final submission file
├── run{N}_diagnostics.json         # Charts data for offline review
└── comparison_dashboard.html       # Final comparison across all runs
```

### Resume logic:
```python
def check_resume(run_id, fold_id):
    """Check if a fold was already completed."""
    path = f'/content/drive/MyDrive/myllia_v23/run{run_id}_fold{fold_id}_preds.npy'
    if os.path.exists(path):
        print(f'  ✓ Run {run_id} Fold {fold_id} already complete, loading...')
        return np.load(path)
    return None
```

Each fold checks for existing predictions before training. If found, it skips
training and loads the saved predictions. This means:
- A crash after fold 3 → resume from fold 4
- A Colab timeout → restart and pick up where you left off
- Re-running the cell → no redundant training

---

## Comparison Methodology

### Per-Run Metrics (saved to run{N}_cv_results.json):
- 5-fold CV mean ± std
- OOF score (W, Wcos, combined)
- Per-fold breakdown (score, W, Wcos, time)
- LOO diagnostic stats (mean |delta|, KO delta corr, gene coverage)

### Cross-Run Comparison Dashboard (comparison_dashboard.html):
- Bar chart: CV scores across all 6 runs with error bars
- Table: detailed metrics per run
- Scatter: CV vs LOO quality metrics
- Heatmap: pairwise prediction correlation between runs
- Decision: which run(s) to submit to LB

### Decision Logic:
```
1. If any Run 1-4 beats Run 0 (baseline) CV by > 0.01:
   → LOO quality improvement is real, use it
2. If Run 5 (Replogle) beats Run 0 by any margin:
   → Replogle augmentation works, include it
3. Run 6 combines the best LOO + Replogle
4. Submit Run 6 to LB (primary)
5. Submit the best individual run to LB (backup)
6. If no run beats Run 0: submit Run 0 (confirmed V20 RUN=3 config)
```

### CV Interpretation Warning:
Remember the **inverse CV/LB correlation** for LOO-heavy runs:
- Lower CV on GT perts may indicate BETTER generalization to unseen genes
- Don't dismiss a run just because its CV is slightly lower than baseline
- The key metric is relative CV ranking among V23 runs (same codebase)

---

## Notebook Cell Layout

| Cell | Type | Content |
|------|------|---------|
| 0 | Markdown | V23 overview, strategy, this blueprint summary |
| 1 | Code | Setup & Drive Mount + checkpoint directory creation |
| 2 | Code | Kaggle download (reused from V21/V22) |
| 3 | Code | Dependencies (+ sklearn GMM, NearestNeighbors) |
| 4 | Code | Config dataclass with run-specific parameters |
| 5 | Code | Load competition data (means, GT, submission template) |
| 6 | Code | Competition metric + KO correction (reused) |
| 7 | Code | 3-Tier PseudoBulk generation + control cell caching (reused) |
| 8 | Code | **LOO Quality Methods** — all 4 methods in one cell with selector |
| 9 | Code | **Replogle Download & Processing** — scPerturb → mapped deltas |
| 10 | Code | Model + Loss + Training function (frozen V17, reused) |
| 11 | Code | **Multi-Run CV Engine** — loops over 6 configs with checkpointing |
| 12 | Code | **Comparison Dashboard** — charts, tables, decision gate |
| 13 | Code | **Final Training + Submission** — best config, all data |
| 14 | Code | **Post-Mortem Template** |

### Cell 8 Design: LOO Quality Methods

```python
class LOOMethod(Enum):
    BASELINE = "baseline"           # Run 0: bottom 5% by raw expression
    NN_MATCHED = "nn_matched"       # Run 1: nearest-neighbor matched pairs
    RESIDUAL = "residual"           # Run 2: regression residual selection
    ADAPTIVE = "adaptive"           # Run 3: GMM bimodal detection
    WEIGHTED = "weighted"           # Run 4: continuous expression weighting

def generate_loo_samples(method: LOOMethod, ctrl_dense, gene_names, cfg, rng):
    """Generate LOO samples using the specified method.
    
    Returns: (gene_indices, deltas, sample_weights) tensors
    """
    if method == LOOMethod.BASELINE:
        return _loo_baseline(ctrl_dense, gene_names, cfg, rng)
    elif method == LOOMethod.NN_MATCHED:
        return _loo_nn_matched(ctrl_dense, gene_names, cfg, rng)
    elif method == LOOMethod.RESIDUAL:
        return _loo_residual(ctrl_dense, gene_names, cfg, rng)
    elif method == LOOMethod.ADAPTIVE:
        return _loo_adaptive(ctrl_dense, gene_names, cfg, rng)
    elif method == LOOMethod.WEIGHTED:
        return _loo_weighted(ctrl_dense, gene_names, cfg, rng)
```

Each method returns the same format: `(gene_indices, delta_targets, sample_weights)`
so the training loop is identical regardless of LOO method.

### Cell 9 Design: Replogle Processing

```python
def load_replogle_augmentation(gene_names, cfg):
    """Download, process, and cache Replogle K562 CRISPRi deltas.
    
    Returns: (gene_indices, deltas, sample_weights) matching LOO format
    """
    cache_path = '/content/drive/MyDrive/myllia_v23/replogle_deltas.npz'
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['gene_indices'], data['deltas'], data['weights']
    
    # Download + process + map to competition space
    # ... (reused from V10/V13 with normalization fix)
    
    np.savez(cache_path, gene_indices=gi, deltas=deltas, weights=weights)
    return gi, deltas, weights
```

### Cell 11 Design: Multi-Run CV Engine

```python
RUN_CONFIGS = [
    {"id": 0, "name": "Baseline",     "loo_method": "baseline",   "replogle": False},
    {"id": 1, "name": "NN-Matched",   "loo_method": "nn_matched", "replogle": False},
    {"id": 2, "name": "Residual",     "loo_method": "residual",   "replogle": False},
    {"id": 3, "name": "Adaptive",     "loo_method": "adaptive",   "replogle": False},
    {"id": 4, "name": "Weighted",     "loo_method": "weighted",   "replogle": False},
    {"id": 5, "name": "Replogle",     "loo_method": "baseline",   "replogle": True},
    # Run 6 is added dynamically after Runs 0-5 complete
]

for run_cfg in RUN_CONFIGS:
    # Check resume
    existing = load_run_results(run_cfg["id"])
    if existing is not None:
        print(f'Run {run_cfg["id"]} already complete: CV={existing["cv_mean"]:.4f}')
        continue
    
    # Generate LOO with specified method
    loo_data = generate_loo_samples(run_cfg["loo_method"], ...)
    
    # Optionally add Replogle
    if run_cfg["replogle"]:
        rep_data = load_replogle_augmentation(...)
        loo_data = concatenate(loo_data, rep_data)
    
    # 5-fold CV with checkpointing per fold
    for fold in range(5):
        fold_preds = check_resume(run_cfg["id"], fold)
        if fold_preds is not None: continue
        # ... train ensemble, save fold predictions
    
    # Save submission file for this run
    save_submission(run_cfg["id"], oof_predictions)
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Replogle download timeout | Essential-only fallback (1.5 GB vs 10 GB total) |
| Colab crash mid-run | Checkpoint/resume at fold level |
| LOO improvements don't help | Baseline run provides direct comparison |
| Replogle domain shift too large | Low weight (0.03) limits damage |
| GMM convergence issues | Fallback to baseline percentile per gene |
| NN matching too slow | PCA reduces to 50-dim, NearestNeighbors is O(n log n) |
| Total runtime exceeds 7 hours | 6 runs × ~130 min = ~13 hours. Need to run across 2 sessions |

### Runtime Estimate

| Run | LOO Gen | Training (5-fold) | Total |
|-----|---------|-------------------|-------|
| 0: Baseline | ~5 min | ~130 min | ~135 min |
| 1: NN-Matched | ~8 min | ~130 min | ~138 min |
| 2: Residual | ~5 min | ~130 min | ~135 min |
| 3: Adaptive | ~8 min | ~130 min | ~138 min |
| 4: Weighted | ~3 min | ~130 min | ~133 min |
| 5: Replogle | ~25 min (download) | ~135 min | ~160 min |
| 6: Combined | ~8 min (cached) | ~135 min | ~143 min |
| **Total** | | | **~15 hours** |

**Execution plan**: Split across 2-3 Colab sessions:
- Session 1: Runs 0-2 (~7 hours)
- Session 2: Runs 3-5 (~7 hours) 
- Session 3: Run 6 + comparison + final submission (~3 hours)

With checkpoint/resume, each session picks up exactly where the last one left off.

---

## Expected Outcomes

| Run | Expected CV | Confidence | Rationale |
|-----|-------------|------------|-----------|
| 0: Baseline | 0.967 | HIGH | Reproduces V20 RUN=3 |
| 1: NN-Matched | 0.955-0.975 | MEDIUM | Cleaner signal, may lower or raise CV |
| 2: Residual | 0.960-0.975 | MEDIUM | Similar to NN but different mechanism |
| 3: Adaptive | 0.960-0.970 | MEDIUM | Gene-specific thresholds, modest impact |
| 4: Weighted | 0.960-0.975 | MEDIUM | Smoother version of baseline |
| 5: Replogle | 0.950-0.980 | HIGH | 2000+ real knockouts should help |
| 6: Combined | 0.945-0.970 | HIGH | Best LOO + Replogle should maximize LB |

**Remember**: Lower CV may predict higher LB (inverse correlation).
The true test is the leaderboard submission.

---

## Post-Comparison Decision Tree

```
After all 6 runs complete:

1. Rank runs by CV score
2. Identify best LOO method (Runs 1-4 vs Run 0)
3. Check if Replogle helped (Run 5 vs Run 0)

IF best_loo > baseline AND replogle_helps:
  → Run 6 = best_loo + replogle → SUBMIT
  
IF best_loo > baseline AND NOT replogle_helps:
  → Submit best_loo run alone
  
IF NOT best_loo > baseline AND replogle_helps:
  → Submit replogle (baseline LOO + replogle)
  
IF nothing beats baseline:
  → Submit baseline (confirmed V20 RUN=3 config)
  → Re-evaluate: maybe the gap to 5.12 requires a paradigm shift
```

---

## What This Blueprint Does NOT Cover

These are deferred to V24+ if V23 shows promise:

1. **Curriculum Training**: Phase 1 (GT+PB only) → Phase 2 (add LOO). Deferred because it requires training schedule changes.
2. **Cross-Run Ensembling**: Average multiple V23 submissions. Simple but requires multiple LB submissions.
3. **Gene Covariance Regularization**: Use control cell correlation matrix as output regularizer. Requires loss function changes (frozen).
4. **Test-Time LOO Blending**: Blend model predictions with raw LOO deltas at inference. Requires post-processing changes.
5. **5x+ Resamples**: If 4x pooled LOO helps, push further. Requires more compute.
