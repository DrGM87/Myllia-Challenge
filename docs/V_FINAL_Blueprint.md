# V_FINAL Blueprint — Last Submission

## SITUATION ASSESSMENT

### Competition Phase Change

The organizers released `pert_ids_all.csv` containing the identities of ALL 120 perturbations:

- **pert_1–60 (public LB):** 60 "val" perturbations — identities were known since day 1. All our public LB scores are on these.
- **pert_61–120 (FINAL ranking):** 60 "test" perturbations — **identities just revealed.** Only these matter for the final leaderboard.

### CRITICAL DISCOVERY: All Existing Submissions Have ZEROS for Rows 61–120

Every submission we ever made — including V20 RUN=3 (LB=4.090, our best) — has **all-zero predictions for pert_61–120**. This was the correct behavior when test identities were unknown (no KO correction possible, and public LB only scored pert_1–60). But now:

- **If we submit any existing CSV as-is for the final, pert_61–120 predictions are all zeros → catastrophic final score.**
- **The ENTIRE final ranking depends on generating real predictions for these 60 new perturbations.**
- Our public LB of 4.090 was earned purely on pert_1–60. We have zero signal on how we perform on pert_61–120.

### What We Know About the 60 Final Test Perturbations

**Gene identities (pert_61–120):**

```
KMT2C, SP1, TADA1, FOSL1, BECN1, HDAC2, UVRAG, DLG5, VCL, JUN,
STRAP, CSK, DHX36, MED12, FEN1, ITGAV, NDUFA7, FOXA1, MED13L, ERBB3,
VPS45, KMT2E, SIRT6, ZNF581, BIRC2, HDAC3, HIRA, HSP90AA1, KDM1A, CS,
C1QBP, KLF10, HMGN1, TAF13, KDM2B, PHF14, CEBPB, RAB5A, HSPA8, MBTPS1,
SETD7, KDM3B, YEATS2, CUL2, SUPV3L1, SRCAP, PBX1, EIF4B, KDM6B, ACTR3,
BAX, STAT6, VIM, PHF10, GRB2, KRT18, SOX2, TCF3, KMT2D, OXCT1
```

**Overlap analysis:**

- Train ∩ Test: **0 genes** (no overlap)
- Train ∩ Val: **0 genes** (no overlap)
- Val ∩ Test: **0 genes** (no overlap)
- Total unique genes across all sets: **200** (80 train + 60 val + 60 test, all disjoint)

**Functional categories of test genes:**

| Category | Count | Genes |
|----------|-------|-------|
| Chromatin/epigenetic | 19 | KMT2C, TADA1, HDAC2, KMT2E, SIRT6, HDAC3, HIRA, KDM1A, HMGN1, TAF13, KDM2B, PHF14, SETD7, KDM3B, YEATS2, SRCAP, KDM6B, PHF10, KMT2D |
| Signaling/TF | 15 | SP1, FOSL1, JUN, CSK, FOXA1, ERBB3, BIRC2, KLF10, CEBPB, PBX1, BAX, STAT6, GRB2, SOX2, TCF3 |
| Metabolism | 6 | NDUFA7, CS, C1QBP, MBTPS1, SUPV3L1, OXCT1 |
| Other (structural/trafficking/misc) | 20 | BECN1, UVRAG, DLG5, VCL, STRAP, DHX36, MED12, FEN1, ITGAV, MED13L, VPS45, ZNF581, HSP90AA1, RAB5A, HSPA8, CUL2, EIF4B, ACTR3, VIM, KRT18 |

**Gene family overlaps with training set (same prefix/family, different gene):**

| Test Gene | Related Training Genes |
|-----------|----------------------|
| HDAC2, HDAC3 | HDAC4, HDAC8 |
| SIRT6 | SIRT1, SIRT2 |
| KDM1A, KDM2B, KDM3B, KDM6B | KDM4A, KDM5C |
| NDUFA7 | NDUFA10, NDUFA2, NDUFB6 |
| HSP90AA1, HSPA8 | HSPA4, HSPD1 |
| SETD7 | SETD1A |
| EIF4B | EIF3H |
| STAT6 | STAT5B |

→ The test set is **heavily enriched for chromatin modifiers** (32% vs ~15% in training). This means the test perturbations may have stronger downstream effects (chromatin modifiers tend to have broad transcriptomic impact).

---

## WHAT WE KNOW WORKS (from 12+ LB submissions)

### Complete LB Ranking (updated with V23 R3)

| # | Submission | LB | CV | Strategy |
|---|-----------|-----|-----|---------|
| 1 | V20 RUN=3 | **4.090** | 0.967 | Simple 3x LOO, no tricks |
| 2 | V20 RUN=8 | 4.068 | 0.969 | Same, N_ENSEMBLE=40 |
| 3 | V20 cv9642 | 4.066 | 0.964 | Same family |
| 4 | V23 R5 | 4.064 | 0.943 | Baseline + Replogle augmentation |
| 5 | V22 | 4.051 | 0.948 | 4x LOO + per-channel |
| 6 | V20 RUN=4 | 4.033 | 0.981 | 2x LOO |
| 7 | **V23 R3** | **3.985** | **0.959** | **Adaptive LOO** |
| 8 | V23 R6 | 3.874 | 0.968 | Weighted + Replogle |
| 9 | V23 R1 | 3.855 | 0.977 | NN-Matched LOO |
| 10 | V21 | 3.792 | 0.931 | Targeted LOO + threshold |
| 11 | V23 R4 | 3.772 | 0.977 | Weighted LOO |
| 12 | V23 R2 | 3.590 | 0.860 | Residual LOO |

### Proven Principles

1. **Simplicity wins.** V20 RUN=3 (simple 3x pooled LOO) is our best ever. No tricks.
2. **Replogle augmentation helps as regularizer.** V23 R5 (baseline + Replogle) is #4 all-time despite lowest non-residual CV. Replogle data adds noise that PREVENTS overfitting.
3. **LOO quality filtering HURTS LB.** NN-matching (R1), weighting (R4), adaptive (R3) — all inflated CV but degraded LB. The exception: R3 (Adaptive) at 3.985 was better than expected, suggesting mild per-gene weighting is less harmful than aggressive filtering.
4. **CV is anti-correlated with LB for V23 runs (r≈-0.85).** Do NOT trust CV as a guide.
5. **Correlation with V20_RUN3 is the strongest LB predictor (r=0.91).** Submissions similar to V20 RUN=3 score best.
6. **cosine_light loss with λ=0.08 is optimal.** Proven across V16-V23.
7. **KO correction matters.** The self-targeting gene must be set to large negative.
8. **3-tier PseudoBulk is a proven augmenter.** Consistently used in all top submissions.

### Correlation Anchor

**Reference file:** `submission_v20_loo_20260222_180450_0_9672x3.csv` (V20 RUN=3, LB=4.090)

- This file has real predictions for pert_1–60 and zeros for pert_61–120.
- The pert_1–60 predictions in this file are our **best-performing public predictions.**
- The final submission should maintain maximum correlation with this file for pert_1–60.

---

## STRATEGY FOR V_FINAL

### Core Principle: Don't Overthink It

Every "clever" modification has hurt LB. The winning formula is:
**Blind MLP + cosine_light + 3-tier PB + LOO 3x pooled + KO correction + (optional) Replogle regularization**

### Architecture

**Identical to V20 RUN=3.** No changes.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Blind MLP (H=384, D=2) | Proven across V17-V23 |
| Loss | cosine_light (λ=0.08) | Bisection-tuned in V16, stable since |
| LOO resamples | 3 | Sweet spot (2→4.033, 3→4.090, 4→4.051) |
| LOO weight | 0.05 | V20 proven value |
| N_ENSEMBLE | 30 | V20 default (40 gave 4.068, marginal) |
| CV folds | 5 | Standard |
| Epochs | 300, ES=50 | Standard |
| PseudoBulk | 3-tier | Standard |
| KO correction | Yes | Standard |

### The ONLY Change: Predict All 120 Perturbations

The notebook must be modified to:

1. **Load `pert_ids_all.csv`** instead of `pert_ids_val.csv`
2. **Generate predictions for ALL 120 rows** (not just first 60)
3. **Apply KO correction for pert_61–120** using the now-known gene identities
4. **Fill rows 61–120 with real model predictions** instead of zeros

### Two-Run Strategy

Generate TWO final submission variants:

#### Run A: Pure V20 RUN=3 Reproduction (for all 120 perts)

- **Exact V20 RUN=3 config** (LOO 3x, weight=0.05, no Replogle)
- Train on ALL 80 training perturbations (full training, no CV holdout — this is the final model)
- Predict for all 120 perturbations with correct KO correction
- **Expected pert_1–60 quality:** Should be identical or near-identical to V20 RUN=3 public LB (4.090)
- **Rationale:** This is our most proven and reliable model. Rows 1–60 will correlate ~1.0 with the V20 RUN=3 reference.

#### Run B: V20 RUN=3 + Replogle Augmentation (for all 120 perts)

- **V20 RUN=3 config + Replogle K562 pseudo-bulk deltas as additional training data**
- Same architecture, same hyperparameters, just adding Replogle augmentation
- Train on ALL 80 training perturbations + Replogle pseudo-bulk deltas (full training)
- Predict for all 120 perturbations with correct KO correction
- **Expected quality:** V23 R5 showed Replogle helps LB (+0.29 over R4). The regularization effect may be especially valuable for pert_61–120 (unseen genes from different functional categories)
- **Rationale:** Replogle coverage of test gene families (HDACs, KDMs, SIRTs) provides indirect biological signal that pure LOO lacks.

### Correlation Guidance

After generating both runs:

1. **Compare pert_1–60 of Run A with V20 RUN=3 reference.** Correlation should be >0.999. If not, something is wrong with the notebook reproduction.
2. **Compare pert_1–60 of Run B with V20 RUN=3 reference.** Correlation should be >0.99 (Replogle adds slight variation).
3. **Compare pert_61–120 of Run A vs Run B.** The difference here quantifies the Replogle effect on unseen perturbations.

### Ensemble Option (Run C)

If time permits, create a simple 50/50 blend:

```
Run C = 0.5 * Run A + 0.5 * Run B
```

This combines the stability of pure V20 with the regularization benefit of Replogle. If Run A and Run B disagree on some pert_61–120 rows, the blend hedges our bets.

### Final Submission Selection

| Scenario | Submit |
|----------|--------|
| If only 1 submission allowed | Run A (most proven, lowest risk) |
| If 2 submissions allowed | Run A + Run B |
| If 3 submissions allowed | Run A + Run B + Run C (ensemble) |

---

## IMPLEMENTATION CHECKLIST

### What to Modify in the Notebook

1. **Download new `pert_ids_all.csv`** from Kaggle data section
2. **Replace `pert_ids_val.csv` loading** with `pert_ids_all.csv` loading
3. **Build gene→pert_id mapping for ALL 120 perturbations** (currently only 60)
4. **In final prediction loop:** iterate over all 120 pert_ids, not just 60
5. **KO correction:** For each pert_id, look up the gene name from `pert_ids_all.csv`, find the gene index in the 5127-gene space, and force that column to the KO delta value
6. **Output CSV:** All 120 rows must have real predictions (no zeros)
7. **Correlation check:** After generating submission, compute correlation of rows 1–60 with V20 RUN=3 reference file. Must be >0.999 for Run A.

### What NOT to Do

- ❌ Do NOT add any new LOO quality filtering (NN-matching, weighting, adaptive)
- ❌ Do NOT change the loss function or hyperparameters
- ❌ Do NOT add any post-processing (SVD, alpha scaling, etc.)
- ❌ Do NOT use perturbation identity as model input (keep it blind)
- ❌ Do NOT trust CV scores — they are misleading for this task
- ❌ Do NOT try to "optimize" specifically for the test gene categories

### Safety Rails

- **Correlation gate:** If pert_1–60 correlation with V20 RUN=3 reference < 0.995, abort and debug
- **Sanity check:** pert_61–120 should have similar statistics (mean, std, pct_near_zero) to pert_1–60
- **KO check:** For each pert_61–120, verify the KO gene column has a large negative delta

---

## WHY THIS WILL WORK

1. **The model is blind.** It predicts the same delta regardless of which gene is knocked out. The only perturbation-specific element is the KO correction. Since we now know the gene identities for pert_61–120, we can apply correct KO correction — giving us the same prediction quality as pert_1–60.

2. **LOO captures the generalization pattern.** LOO synthetic perturbations cover all 5127 genes, including the 60 test genes. The model has already "seen" synthetic knockouts of HDAC2, KMT2C, SP1, etc. through LOO augmentation.

3. **Replogle adds biological signal.** Replogle K562 data contains actual perturbation experiments for gene families present in the test set (HDACs, KDMs, SIRTs). While from a different cell line, V23 R5 proved this acts as beneficial regularization.

4. **No new risk.** We are NOT introducing any new techniques. We are simply applying our proven best model to the newly revealed test perturbations. The only "new" thing is generating predictions for rows 61–120 instead of zeros.

---

## EXPECTED OUTCOMES

| Submission | pert_1–60 (public) | pert_61–120 (final) | Overall |
|------------|-------------------|--------------------|---------| 
| Run A (V20 repro) | ~4.09 (matches best) | Unknown but comparable | Our floor |
| Run B (+ Replogle) | ~4.06 (matches R5) | Potentially better (regularization) | Upside |
| Run C (blend) | ~4.07 | Hedged | Safe middle |

The final ranking is determined ONLY by pert_61–120. Since our blind MLP treats all perturbations equally (except KO correction), and since LOO already provides synthetic data for these genes, our pert_61–120 predictions should be of comparable quality to pert_1–60.

**The biggest risk is NOT making a mistake — it's submitting zeros.** Any real prediction >>> zeros.

---

## TIMELINE

1. Modify notebook to load `pert_ids_all.csv` and predict all 120 rows (~30 min)
2. Run A: V20 RUN=3 exact reproduction for 120 perts (~3-4 hrs on A100)
3. Run B: Same + Replogle augmentation (~4-5 hrs on A100)
4. Correlation check against V20 reference, sanity checks (~15 min)
5. Generate Run C ensemble if time permits (~5 min)
6. Submit

**Total: ~8 hrs compute + 1 hr prep/validation**
