# Competition: Echoes of Silenced Genes

> **Myllia | Echoes of Silenced Genes: A Cell Challenge**
> Organized by [Myllia Biotechnology](https://myllia.com/) (Vienna, Austria)
> [Kaggle Competition Page](https://kaggle.com/competitions/echoes-of-silenced-genes)

---

## Overview

Modelling cell behavior in response to stimuli and perturbations has been an increasingly popular topic. The so-called virtual cell models promise to navigate, or even replace lab experiments and thus significantly contribute to advances in medical research.

Your objective in this challenge is to predict how a human cancer cell line responds to perturbations induced by CRISPR-interference.

The challenge was organized to:

- See how capable current models of cell response are
- Evaluate a new metric of prediction quality
- Have fun

Winners were announced at the **High-Content CRISPR Screening Conference** in March 2026 in Vienna, Austria.

---

## Background

Single-cell RNA-seq has been used to catalog mRNA expression levels of almost all accessible cell types. The advent of techniques like CROP-seq and perturb-seq has enabled the generation of very large datasets containing transcriptomes of cells after a CRISPR perturbation. Key public datasets include:

- Replogle et al., 2022
- Nourreddine et al., 2024
- Huang et al., 2025
- Jiang et al., 2025
- Nadig et al., 2025
- Song et al., 2025
- Zhu et al., 2026

The increasing abundance of such datasets triggers ideas like: *Having seen what the perturbation does in one cell type, can we predict what it will do in another?*

---

## The Task

Predict **delta average expression** (log2 fold change) of **5,127 genes** for each of **120 perturbations**.

- **Training data:** 80 known perturbations with ground truth deltas
- **Validation set (public LB):** 60 perturbations (pert_1 to pert_60)
- **Test set (final ranking):** 60 perturbations (pert_61 to pert_120) — identities revealed one week before deadline
- All 200 genes (80 train + 60 val + 60 test) are **completely disjoint**

---

## Evaluation Metric

The metric consists of two parts: **Weighted Mean Absolute Error Ratio** and **Weighted Cosine Similarity**.

### Final Score

```
Score = W × max(0, Wcos)
```

### Weighted MAE Ratio (W)

For each perturbation:

```
WMAE(true, pred) = (1/n) × Σ wᵢ × |trueᵢ - predᵢ|
```

Where weights `w` are derived from moderated t-statistics (limma package), with the target gene's weight set to 0.

The aggregated W score over L perturbations:

```
W = Σ min(5, log₂(WMAE_baseline / WMAE_pred))
```

The baseline prediction is the arithmetic mean of all 80 training delta vectors.

### Weighted Cosine Similarity (Wcos)

Uses a smoothstep gating function to focus on genes with large effects:

```
gate(x) = smoothstep(clip((max(|a|, |b|) - left) / (right - left), 0, 1))
Wcos = Σ(gate² × a × b) / (√(Σ gate² × a²) × √(Σ gate² × b²))
```

With `left = 0`, `right = 0.3` (official) or `right = 0.2` (evaluation code).

---

## Data Files

### Main Files

| File | Description |
|------|-------------|
| `training_data_means.csv` | Average expression values for 80 perturbations + non-targeting control. 5,127 gene columns. |
| `pert_ids_val.csv` | Maps pert_id → gene symbol for 60 public LB perturbations |
| `pert_ids_all.csv` | Maps pert_id → gene symbol for all 120 perturbations (released near deadline) |
| `sample_submission.csv` | Template: 120 rows × 5,128 columns (pert_id + 5,127 genes) |

### Supplementary Files

| File | Description |
|------|-------------|
| `training_cells.h5ad` | Single-cell resolution raw UMI counts (19,226 genes). Used for PseudoBulk augmentation. |
| `training_data_ground_truth_table.csv` | Delta values, gene weights, and baseline WMAE for the 80 training perturbations. |

### Normalization

Raw UMI counts are processed as:

1. Divide by total UMI count per cell
2. Multiply by 10,000
3. Log-transform: `log2(x + 1)`
4. Subset to 5,127 competition genes
5. Average per perturbation

---

## About the Organizer

**Myllia Biotechnology** is a biotech company based in Vienna, Austria. Its technology platform combines CRISPR perturbation screens with single-cell RNA sequencing in primary human cells to deliver deeply physiologic, mechanistic insights.

---

## Citation

```bibtex
@misc{echoes-of-silenced-genes,
    author = {Myllia Biotechnology},
    title = {Myllia | Echoes of Silenced Genes: A Cell Challenge},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/echoes-of-silenced-genes}},
    note = {Kaggle}
}
```
