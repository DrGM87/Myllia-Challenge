"""
Ensemble Analysis of Historical Submission Files
=================================================
Analyzes correlations between submissions, tries various ensemble strategies,
and generates an interactive HTML report.

IMPORTANT: Rows 61-120 (pert_61 to pert_120) are intentionally zeroed in all
submission files. Only the first 60 rows (pert_1 to pert_60) contain real
scored predictions. All analysis uses ONLY these 60 scored rows.

Submissions:
  V22: submission_v22_loo_max_20260303_094459_0_9481.csv  LB=4.05101
  V21: submission_v21_sparse_20260226_081551.csv           LB=3.79172
  V20a: submission_v20_loo_20260225_235344_0_9642.csv      LB=4.06615
  V20b: submission_v20_loo_20260225_090826.csv             LB=4.06765 (RUN=8, N=40)
  V20c: submission_v20_loo_20260223_114020_0_9806.csv      LB=4.03252
  V20d: submission_v20_loo_20260222_180450_0_9672x3.csv    LB=4.08996 (BEST)
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
import json

# ============================================================
# 1. LOAD SUBMISSIONS
# ============================================================

BASE = Path(r"c:\Users\MSI\Desktop\projects\Myllia_Challenge")

SUBMISSIONS = {
    "V20_RUN3 (BEST)": {
        "file": "submission_v20_loo_20260222_180450_0_9672x3.csv",  # user's original is _0_9672.csv but local file has x3 suffix
        "lb": 4.08996,
        "cv": 0.9672,
        "desc": "3x pooled LOO, N=30",
    },
    "V20_RUN8": {
        "file": "submission_v20_loo_20260225_090826.csv",
        "lb": 4.06765,
        "cv": 0.9692,
        "desc": "3x LOO, N_ENSEMBLE=40",
    },
    "V20_cv9642": {
        "file": "submission_v20_loo_20260225_235344_0_9642.csv",
        "lb": 4.06615,
        "cv": 0.9642,
        "desc": "3x LOO variant",
    },
    "V20_RUN4": {
        "file": "submission_v20_loo_20260223_114020_0_9806.csv",
        "lb": 4.03252,
        "cv": 0.9806,
        "desc": "2x LOO, N=30",
    },
    "V22": {
        "file": "submission_v22_loo_max_20260303_094459_0_9481.csv",
        "lb": 4.05101,
        "cv": 0.9481,
        "desc": "4x LOO + per-channel",
    },
    "V21": {
        "file": "submission_v21_sparse_20260226_081551.csv",
        "lb": 3.79172,
        "cv": 0.9306,
        "desc": "Targeted 2000 genes + threshold",
    },
}

print("Loading submissions...")
dfs = {}
for name, info in SUBMISSIONS.items():
    path = BASE / info["file"]
    if not path.exists():
        print(f"  MISSING: {path.name}")
        continue
    df = pd.read_csv(path)
    dfs[name] = df
    print(f"  {name}: {df.shape} | LB={info['lb']}")

# Extract numeric matrices (drop pert_id column)
pert_ids = list(dfs.values())[0].iloc[:, 0].values
gene_names = list(dfs.values())[0].columns[1:]
n_perts = len(pert_ids)
n_genes = len(gene_names)

# CRITICAL: Only use first 60 rows (scored perturbations)
# Rows 61-120 are intentionally zeroed and would inflate correlations
SCORED_ROWS = 60

matrices_full = {}  # All 120 rows (for submission files)
matrices = {}       # Only 60 scored rows (for analysis)
for name, df in dfs.items():
    full = df.iloc[:, 1:].values.astype(np.float64)
    matrices_full[name] = full
    matrices[name] = full[:SCORED_ROWS]  # Only scored rows

n_perts = SCORED_ROWS  # Override to 60
print(f"\nFull matrix shape: {len(pert_ids)} perts × {n_genes} genes")
print(f"Analysis uses ONLY first {SCORED_ROWS} scored rows (rows 61-120 are zeroed)")
print(f"Analysis matrix shape: {n_perts} perts × {n_genes} genes")

# ============================================================
# 2. PAIRWISE CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("PAIRWISE CORRELATION ANALYSIS")
print("=" * 60)

names = list(matrices.keys())
num_subs = len(names)
corr_matrix = np.zeros((num_subs, num_subs))
flat_corrs = {}

for i in range(num_subs):
    for j in range(num_subs):
        a = matrices[names[i]].ravel()
        b = matrices[names[j]].ravel()
        corr_matrix[i, j] = np.corrcoef(a, b)[0, 1]
        if i < j:
            flat_corrs[f"{names[i]} vs {names[j]}"] = corr_matrix[i, j]

print("\nCorrelation matrix:")
for i in range(num_subs):
    row = " ".join(f"{corr_matrix[i, j]:.4f}" for j in range(num_subs))
    print(f"  {names[i]:20s}: {row}")

# Per-perturbation correlations
print("\nPer-perturbation correlation (row-wise):")
per_pert_corrs = {}
for i in range(num_subs):
    for j in range(i + 1, num_subs):
        row_corrs = []
        for r in range(n_perts):
            a_row = matrices[names[i]][r]
            b_row = matrices[names[j]][r]
            # Skip constant rows (std=0 causes NaN in corrcoef)
            if np.std(a_row) < 1e-12 or np.std(b_row) < 1e-12:
                continue
            rc = np.corrcoef(a_row, b_row)[0, 1]
            if not np.isnan(rc):
                row_corrs.append(rc)
        per_pert_corrs[f"{names[i]} vs {names[j]}"] = row_corrs
        if row_corrs:
            mean_rc = np.mean(row_corrs)
            min_rc = np.min(row_corrs)
            print(f"  {names[i]:20s} vs {names[j]:20s}: mean={mean_rc:.4f}, min={min_rc:.4f} ({len(row_corrs)}/{n_perts} valid)")
        else:
            print(f"  {names[i]:20s} vs {names[j]:20s}: NO valid correlations")

# ============================================================
# 3. PREDICTION STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION STATISTICS")
print("=" * 60)

stats = {}
for name in names:
    m = matrices[name]
    s = {
        "mean": float(np.mean(m)),
        "std": float(np.std(m)),
        "min": float(np.min(m)),
        "max": float(np.max(m)),
        "abs_mean": float(np.mean(np.abs(m))),
        "pct_near_zero": float(np.mean(np.abs(m) < 0.02) * 100),
        "pct_zero": float(np.mean(m == 0) * 100),
        "lb": SUBMISSIONS[name]["lb"],
    }
    stats[name] = s
    print(f"\n  {name} (LB={s['lb']}):")
    print(f"    mean={s['mean']:.6f}, std={s['std']:.6f}")
    print(f"    range=[{s['min']:.4f}, {s['max']:.4f}]")
    print(f"    |pred| mean={s['abs_mean']:.6f}")
    print(f"    near-zero (<0.02): {s['pct_near_zero']:.1f}%")
    print(f"    exact zero: {s['pct_zero']:.1f}%")

# ============================================================
# 4. ENSEMBLE STRATEGIES
# ============================================================

print("\n" + "=" * 60)
print("ENSEMBLE STRATEGIES")
print("=" * 60)

ensemble_results = []

# Strategy 1: Simple average of all
avg_all = np.mean([matrices[nm] for nm in names], axis=0)
ensemble_results.append({
    "name": "Simple Average (all 6)",
    "desc": "Equal weight to all submissions",
    "matrix": avg_all,
    "weights": {nm: 1.0 / len(names) for nm in names},
})

# Strategy 2: Exclude V21 (worst performer)
good_names = [nm for nm in names if nm != "V21"]
avg_no_v21 = np.mean([matrices[nm] for nm in good_names], axis=0)
ensemble_results.append({
    "name": "Average (excl V21)",
    "desc": "Exclude worst performer (V21, LB=3.79)",
    "matrix": avg_no_v21,
    "weights": {nm: (1.0 / len(good_names) if nm != "V21" else 0) for nm in names},
})

# Strategy 3: LB-weighted average
lbs = np.array([SUBMISSIONS[nm]["lb"] for nm in names])
lb_weights = lbs / lbs.sum()
avg_lb_weighted = np.zeros_like(avg_all)
for i, nm in enumerate(names):
    avg_lb_weighted += lb_weights[i] * matrices[nm]
ensemble_results.append({
    "name": "LB-Weighted Average",
    "desc": "Weight proportional to LB score",
    "matrix": avg_lb_weighted,
    "weights": {nm: float(lb_weights[i]) for i, nm in enumerate(names)},
})

# Strategy 4: LB-weighted, exclude V21
good_lbs = np.array([SUBMISSIONS[nm]["lb"] for nm in good_names])
good_lb_w = good_lbs / good_lbs.sum()
avg_lb_no_v21 = np.zeros_like(avg_all)
for i, nm in enumerate(good_names):
    avg_lb_no_v21 += good_lb_w[i] * matrices[nm]
ensemble_results.append({
    "name": "LB-Weighted (excl V21)",
    "desc": "LB-proportional weights, no V21",
    "matrix": avg_lb_no_v21,
    "weights": {nm: (float(good_lb_w[good_names.index(nm)]) if nm in good_names else 0) for nm in names},
})

# Strategy 5: Top-3 by LB
sorted_by_lb = sorted(names, key=lambda n: SUBMISSIONS[n]["lb"], reverse=True)
top3 = sorted_by_lb[:3]
avg_top3 = np.mean([matrices[nm] for nm in top3], axis=0)
ensemble_results.append({
    "name": f"Top-3 Average ({', '.join(top3)})",
    "desc": "Average of 3 best LB submissions",
    "matrix": avg_top3,
    "weights": {nm: (1.0 / 3 if nm in top3 else 0) for nm in names},
})

# Strategy 6: Inverse-CV weighted (lower CV → more LOO → better LB)
cvs = np.array([SUBMISSIONS[nm]["cv"] for nm in names])
inv_cv = 1.0 / cvs
inv_cv_w = inv_cv / inv_cv.sum()
avg_inv_cv = np.zeros_like(avg_all)
for i, nm in enumerate(names):
    avg_inv_cv += inv_cv_w[i] * matrices[nm]
ensemble_results.append({
    "name": "Inverse-CV Weighted",
    "desc": "Lower CV gets higher weight (inverse CV/LB trend)",
    "matrix": avg_inv_cv,
    "weights": {nm: float(inv_cv_w[i]) for i, nm in enumerate(names)},
})

# Strategy 7: Diversity-optimal — maximize pairwise diversity
# Pick the pair with lowest correlation and average them
min_corr = 1.0
min_pair = None
for i in range(num_subs):
    for j in range(i + 1, num_subs):
        if corr_matrix[i, j] < min_corr:
            min_corr = corr_matrix[i, j]
            min_pair = (names[i], names[j])

if min_pair:
    avg_diverse = 0.5 * (matrices[min_pair[0]] + matrices[min_pair[1]])
    ensemble_results.append({
        "name": f"Most Diverse Pair ({min_pair[0]} + {min_pair[1]})",
        "desc": f"Lowest pairwise corr = {min_corr:.4f}",
        "matrix": avg_diverse,
        "weights": {nm: (0.5 if nm in min_pair else 0) for nm in names},
    })

# Strategy 8: Best + Most Diverse partner
best_name = sorted_by_lb[0]
best_idx = names.index(best_name)
partner_corrs = [(names[j], corr_matrix[best_idx, j]) for j in range(num_subs) if j != best_idx]
partner_corrs.sort(key=lambda x: x[1])
most_diverse_partner = partner_corrs[0][0]
div_corr = partner_corrs[0][1]
# Weight best higher (0.6/0.4)
avg_best_diverse = 0.6 * matrices[best_name] + 0.4 * matrices[most_diverse_partner]
ensemble_results.append({
    "name": f"Best+Diverse (0.6×{best_name} + 0.4×{most_diverse_partner})",
    "desc": f"Best LB + most diverse partner (corr={div_corr:.4f})",
    "matrix": avg_best_diverse,
    "weights": {nm: (0.6 if nm == best_name else (0.4 if nm == most_diverse_partner else 0)) for nm in names},
})

# Strategy 9: Optimized weights via grid search on diversity metric
# Maximize: sum of weights[i]*LB[i] - lambda * sum of weights[i]*weights[j]*corr[i][j]
# This balances quality (LB) with diversity (low pairwise correlation)
best_score = -1
best_combo_weights = None
best_combo_name = None

# Try all subsets of size 2-5 with LB-proportional weights
for size in range(2, min(6, num_subs + 1)):
    for combo in combinations(range(num_subs), size):
        combo_names = [names[i] for i in combo]
        combo_lbs = np.array([SUBMISSIONS[names[i]]["lb"] for i in combo])
        w = combo_lbs / combo_lbs.sum()

        # Quality score: weighted LB
        quality = np.sum(w * combo_lbs)

        # Diversity score: negative weighted avg pairwise corr
        diversity = 0
        for ii in range(len(combo)):
            for jj in range(ii + 1, len(combo)):
                diversity -= w[ii] * w[jj] * corr_matrix[combo[ii], combo[jj]]

        # Combined score (lambda controls diversity weight)
        combined = quality + 0.5 * diversity

        if combined > best_score:
            best_score = combined
            best_combo_weights = {names[combo[i]]: float(w[i]) for i in range(len(combo))}
            best_combo_name = " + ".join(combo_names)

if best_combo_weights:
    avg_optimized = np.zeros_like(avg_all)
    for n_key, w in best_combo_weights.items():
        avg_optimized += w * matrices[n_key]
    ensemble_results.append({
        "name": f"Quality+Diversity Optimized",
        "desc": f"Subset: {best_combo_name}",
        "matrix": avg_optimized,
        "weights": {nm: best_combo_weights.get(nm, 0) for nm in names},
    })

# Strategy 10: Rank-based ensemble (average ranks, convert back)
print("\nComputing rank-based ensemble...")
rank_matrices = {}
for name in names:
    m = matrices[name]
    # Rank each perturbation's genes
    ranked = np.zeros_like(m)
    for r in range(n_perts):
        ranked[r] = np.argsort(np.argsort(m[r])).astype(np.float64) / n_genes
    rank_matrices[name] = ranked

# Average ranks (exclude V21)
avg_ranks = np.mean([rank_matrices[nm] for nm in good_names], axis=0)

# Convert back to prediction scale using best submission's statistics
best_m = matrices[best_name]
rank_ensemble = np.zeros_like(avg_all)
for r in range(n_perts):
    # Map average ranks back to values using best submission's distribution
    sort_order = np.argsort(avg_ranks[r])
    best_sorted = np.sort(best_m[r])
    rank_ensemble[r, sort_order] = best_sorted

# Also build full 120-row rank ensemble for submission saving
rank_matrices_full = {}
for name in names:
    m = matrices_full[name]
    ranked = np.zeros_like(m)
    for r in range(m.shape[0]):
        ranked[r] = np.argsort(np.argsort(m[r])).astype(np.float64) / n_genes
    rank_matrices_full[name] = ranked

avg_ranks_full = np.mean([rank_matrices_full[nm] for nm in good_names], axis=0)
best_m_full = matrices_full[best_name]
rank_ensemble_full = np.zeros_like(best_m_full)
for r in range(best_m_full.shape[0]):
    sort_order = np.argsort(avg_ranks_full[r])
    best_sorted = np.sort(best_m_full[r])
    rank_ensemble_full[r, sort_order] = best_sorted

ensemble_results.append({
    "name": "Rank-Based Ensemble (excl V21)",
    "desc": "Average gene ranks, map to best submission's value distribution",
    "matrix": rank_ensemble,
    "weights": {nm: (1.0 / len(good_names) if nm in good_names else 0) for nm in names},
    "full_matrix": rank_ensemble_full,  # Pre-built full matrix for saving
})

# ============================================================
# 5. ENSEMBLE COMPARISON METRICS
# ============================================================

print("\n" + "=" * 60)
print("ENSEMBLE COMPARISON")
print("=" * 60)

# Compare each ensemble to the best individual submission
for ens in ensemble_results:
    m = ens["matrix"]
    corr_to_best = np.corrcoef(m.ravel(), matrices[best_name].ravel())[0, 1]
    abs_mean = np.mean(np.abs(m))
    std = np.std(m)
    pct_near_zero = np.mean(np.abs(m) < 0.02) * 100

    ens["corr_to_best"] = float(corr_to_best)
    ens["abs_mean"] = float(abs_mean)
    ens["std"] = float(std)
    ens["pct_near_zero"] = float(pct_near_zero)

    print(f"\n  {ens['name']}:")
    print(f"    corr to best: {corr_to_best:.6f}")
    print(f"    |pred| mean: {abs_mean:.6f}")
    print(f"    std: {std:.6f}")
    print(f"    near-zero: {pct_near_zero:.1f}%")

# ============================================================
# 6. SAVE ENSEMBLE SUBMISSION FILES
# ============================================================

print("\n" + "=" * 60)
print("SAVING ENSEMBLE SUBMISSIONS")
print("=" * 60)

template_df = list(dfs.values())[0].copy()
saved_files = []

for i, ens in enumerate(ensemble_results):
    safe_name = ens["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("+", "plus")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
    fname = f"ensemble_{i:02d}_{safe_name[:50]}.csv"
    out_path = BASE / fname

    # Build full 120-row submission
    if "full_matrix" in ens:
        # Pre-built (e.g., rank-based ensemble)
        full_matrix = ens["full_matrix"]
    else:
        # Reconstruct from weights applied to full matrices
        full_matrix = np.zeros_like(matrices_full[names[0]])
        for nm, w in ens["weights"].items():
            if w > 0:
                full_matrix += w * matrices_full[nm]

    out_df = template_df.copy()
    out_df.iloc[:, 1:] = full_matrix
    out_df.to_csv(out_path, index=False)
    saved_files.append(fname)
    ens["saved_file"] = fname
    print(f"  Saved: {fname}")

# ============================================================
# 7. GENERATE HTML REPORT
# ============================================================

print("\n" + "=" * 60)
print("GENERATING HTML REPORT")
print("=" * 60)

# Prepare data for charts
corr_data_json = json.dumps({
    "labels": names,
    "matrix": corr_matrix.tolist(),
})

stats_json = json.dumps(stats)

ensemble_table_rows = ""
for i, ens in enumerate(ensemble_results):
    weights_str = ", ".join(f"{k}: {v:.3f}" for k, v in ens["weights"].items() if v > 0)
    ensemble_table_rows += f"""
    <tr>
        <td>{i + 1}</td>
        <td><strong>{ens['name']}</strong></td>
        <td>{ens['desc']}</td>
        <td>{ens['corr_to_best']:.6f}</td>
        <td>{ens['abs_mean']:.6f}</td>
        <td>{ens['pct_near_zero']:.1f}%</td>
        <td style="font-size:11px">{weights_str}</td>
        <td>{ens.get('saved_file', '')}</td>
    </tr>"""

individual_table_rows = ""
for name in sorted(names, key=lambda n: SUBMISSIONS[n]["lb"], reverse=True):
    s = stats[name]
    info = SUBMISSIONS[name]
    individual_table_rows += f"""
    <tr>
        <td><strong>{name}</strong></td>
        <td>{info['lb']:.5f}</td>
        <td>{info['cv']:.4f}</td>
        <td>{s['abs_mean']:.6f}</td>
        <td>{s['std']:.6f}</td>
        <td>{s['pct_near_zero']:.1f}%</td>
        <td>{s['pct_zero']:.1f}%</td>
        <td>{info['desc']}</td>
    </tr>"""

# Correlation heatmap data
heatmap_cells = ""
for i in range(num_subs):
    for j in range(num_subs):
        val = corr_matrix[i, j]
        # Color: green for low corr (diverse), red for high corr
        if i == j:
            color = "#e0e0e0"
        elif val > 0.999:
            color = "#ff4444"
        elif val > 0.995:
            color = "#ff8888"
        elif val > 0.99:
            color = "#ffaaaa"
        elif val > 0.98:
            color = "#ffcccc"
        else:
            color = "#aaffaa"
        heatmap_cells += f'<td style="background:{color};text-align:center;font-size:12px;padding:4px">{val:.5f}</td>'
    heatmap_cells += "</tr><tr>"

heatmap_headers = "".join(f"<th style='font-size:11px;writing-mode:vertical-lr;padding:4px'>{nm}</th>" for nm in names)

# LB vs stats scatter data
scatter_data = []
for name in names:
    scatter_data.append({
        "name": name,
        "lb": SUBMISSIONS[name]["lb"],
        "cv": SUBMISSIONS[name]["cv"],
        "abs_mean": stats[name]["abs_mean"],
        "near_zero": stats[name]["pct_near_zero"],
        "std": stats[name]["std"],
    })
scatter_json = json.dumps(scatter_data)

# Best ensemble recommendation
best_ens = max(ensemble_results, key=lambda e: -e["corr_to_best"] if e["corr_to_best"] < 0.9999 else 0)
# Actually, find the one most different from best (lowest corr) that still uses good submissions
diverse_ensembles = sorted(ensemble_results, key=lambda e: e["corr_to_best"])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Myllia Challenge — Ensemble Analysis Report</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #283593; margin-top: 40px; }}
    h3 {{ color: #3949ab; }}
    .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    th {{ background: #1a237e; color: white; padding: 8px 12px; text-align: left; font-size: 13px; }}
    td {{ padding: 6px 12px; border-bottom: 1px solid #e0e0e0; font-size: 13px; }}
    tr:hover {{ background: #f0f0ff; }}
    .highlight {{ background: #e8f5e9 !important; font-weight: bold; }}
    .warn {{ background: #fff3e0 !important; }}
    .bad {{ background: #ffebee !important; }}
    .metric {{ font-size: 28px; font-weight: bold; color: #1a237e; }}
    .metric-label {{ font-size: 12px; color: #666; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
    .insight {{ background: #e3f2fd; border-left: 4px solid #1565c0; padding: 12px 16px; margin: 10px 0; border-radius: 0 8px 8px 0; }}
    .warning {{ background: #fff3e0; border-left: 4px solid #e65100; padding: 12px 16px; margin: 10px 0; border-radius: 0 8px 8px 0; }}
    .recommendation {{ background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 12px 16px; margin: 10px 0; border-radius: 0 8px 8px 0; }}
    canvas {{ max-width: 100%; }}
    .chart-container {{ position: relative; height: 350px; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head>
<body>
<div class="container">

<h1>Myllia Challenge — Ensemble Analysis Report</h1>
<p>Generated: 2026-03-04 | Submissions analyzed: {len(names)} | Best individual LB: {SUBMISSIONS[best_name]['lb']}<br>
<strong>NOTE:</strong> All analysis uses only the <strong>first 60 scored perturbations</strong>. Rows 61-120 are intentionally zeroed and excluded from correlation/statistics calculations.</p>

<!-- Key Metrics -->
<div class="grid">
    <div class="card" style="text-align:center">
        <div class="metric">{SUBMISSIONS[best_name]['lb']:.5f}</div>
        <div class="metric-label">Best Individual LB ({best_name})</div>
    </div>
    <div class="card" style="text-align:center">
        <div class="metric">{len(names)}</div>
        <div class="metric-label">Submissions Analyzed</div>
    </div>
    <div class="card" style="text-align:center">
        <div class="metric">{len(ensemble_results)}</div>
        <div class="metric-label">Ensemble Strategies Tested</div>
    </div>
    <div class="card" style="text-align:center">
        <div class="metric">{min(corr_matrix[corr_matrix < 0.9999]):.5f}</div>
        <div class="metric-label">Min Pairwise Correlation</div>
    </div>
</div>

<!-- Section 1: Individual Submissions -->
<h2>1. Individual Submission Analysis</h2>
<div class="card">
    <table>
        <tr>
            <th>Submission</th><th>LB Score</th><th>CV</th>
            <th>|Pred| Mean</th><th>Std</th><th>Near-Zero</th><th>Exact Zero</th><th>Description</th>
        </tr>
        {individual_table_rows}
    </table>
</div>

<!-- LB vs CV Chart -->
<div class="card">
    <h3>LB Score vs CV Score</h3>
    <div class="chart-container">
        <canvas id="lbCvChart"></canvas>
    </div>
</div>

<!-- Section 2: Correlation Heatmap -->
<h2>2. Pairwise Correlation Heatmap</h2>
<div class="card">
    <p>Higher correlation = more similar predictions. For ensembling, we want <strong>diverse</strong> (lower correlation) submissions.</p>
    <table style="width:auto">
        <tr><th></th>{heatmap_headers}</tr>
        <tr>{heatmap_cells}</tr>
    </table>
</div>

<div class="insight">
    <strong>Key Finding:</strong> Most V20 runs are extremely correlated (>0.995), meaning ensembling them
    provides minimal diversity benefit. V21 is the most diverse (lowest correlation) but also the worst performer.
    V22 offers a middle ground — decent performance with slightly different predictions.
</div>

<!-- Section 3: LB Score Correlation Analysis -->
<h2>3. What Predicts LB Score?</h2>
<div class="card">
    <h3>Correlation of Prediction Properties with LB Score</h3>
    <div class="chart-container">
        <canvas id="lbCorrelationChart"></canvas>
    </div>
</div>

<!-- Section 4: Ensemble Strategies -->
<h2>4. Ensemble Strategies Comparison</h2>
<div class="card">
    <table>
        <tr>
            <th>#</th><th>Strategy</th><th>Description</th>
            <th>Corr to Best</th><th>|Pred| Mean</th><th>Near-Zero</th><th>Weights</th><th>File</th>
        </tr>
        {ensemble_table_rows}
    </table>
</div>

<div class="card">
    <h3>Ensemble Distance from Best Individual</h3>
    <div class="chart-container">
        <canvas id="ensembleChart"></canvas>
    </div>
</div>

<!-- Section 5: Analysis & Recommendations -->
<h2>5. Analysis & Recommendations</h2>

<div class="warning">
    <strong>Critical Caveat:</strong> We cannot compute the actual LB score of ensembles locally.
    The correlation to the best individual submission is a proxy. Lower correlation means the ensemble
    is <em>different</em> from the best — which could be better OR worse. The only true test is submitting to the LB.
</div>

<div class="recommendation">
    <strong>Recommended Ensembles to Submit (in priority order):</strong>
    <ol>
        <li><strong>LB-Weighted (excl V21)</strong> — Safest bet. Weights best submissions proportionally, excludes the clear underperformer. This is the "conservative improvement" strategy.</li>
        <li><strong>Top-3 Average</strong> — Simple average of the 3 best LB submissions. Minimal computation, good diversity within the best performers.</li>
        <li><strong>Quality+Diversity Optimized</strong> — Algorithmically optimized subset balancing LB quality and prediction diversity. Most likely to capture complementary signal.</li>
        <li><strong>Rank-Based Ensemble</strong> — Most radical: averages gene RANKINGS instead of values, then maps back to best submission's value distribution. This is robust to magnitude differences between submissions.</li>
    </ol>
</div>

<div class="insight">
    <strong>Why simple averaging might NOT work:</strong> All V20 submissions correlate at >0.995.
    Averaging nearly identical predictions barely changes anything. The key diversity comes from V21 and V22,
    but V21's lower quality (LB=3.79) may drag the ensemble down. The optimal strategy is to include V22
    (decent quality, slight diversity) while excluding V21 (low quality, high diversity but harmful).
</div>

<div class="card">
    <h3>Diversity vs Quality Tradeoff</h3>
    <p>For each submission pair, plot their LB score average vs their prediction correlation:</p>
    <div class="chart-container">
        <canvas id="diversityChart"></canvas>
    </div>
</div>

</div><!-- /container -->

<script>
// Chart 1: LB vs CV
const scatterData = {scatter_json};
new Chart(document.getElementById('lbCvChart'), {{
    type: 'scatter',
    data: {{
        datasets: [{{
            label: 'Submissions',
            data: scatterData.map(d => ({{ x: d.cv, y: d.lb }})),
            backgroundColor: scatterData.map(d => d.lb > 4.05 ? '#2e7d32' : (d.lb > 3.9 ? '#ff9800' : '#f44336')),
            pointRadius: 10,
            pointHoverRadius: 14,
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            tooltip: {{
                callbacks: {{
                    label: function(ctx) {{
                        const d = scatterData[ctx.dataIndex];
                        return d.name + ' (LB=' + d.lb.toFixed(3) + ', CV=' + d.cv.toFixed(4) + ')';
                    }}
                }}
            }},
            title: {{ display: true, text: 'Inverse CV/LB Trend — Lower CV often predicts higher LB' }}
        }},
        scales: {{
            x: {{ title: {{ display: true, text: 'CV Score' }}, reverse: false }},
            y: {{ title: {{ display: true, text: 'LB Score' }} }}
        }}
    }}
}});

// Chart 2: LB Correlation with properties
const propNames = ['|Pred| Mean', 'Std', 'Near-Zero %', 'CV'];
const propValues = scatterData.map(d => [d.abs_mean, d.std, d.near_zero, d.cv]);
const lbValues = scatterData.map(d => d.lb);

function pearson(x, y) {{
    const n = x.length;
    const mx = x.reduce((a, b) => a + b) / n;
    const my = y.reduce((a, b) => a + b) / n;
    let num = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) {{
        num += (x[i] - mx) * (y[i] - my);
        dx += (x[i] - mx) ** 2;
        dy += (y[i] - my) ** 2;
    }}
    return num / Math.sqrt(dx * dy);
}}

const correlations = propNames.map((_, pi) => {{
    const vals = propValues.map(pv => pv[pi]);
    return pearson(vals, lbValues);
}});

new Chart(document.getElementById('lbCorrelationChart'), {{
    type: 'bar',
    data: {{
        labels: propNames,
        datasets: [{{
            label: 'Correlation with LB Score',
            data: correlations,
            backgroundColor: correlations.map(c => c > 0 ? '#4caf50' : '#f44336'),
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        scales: {{ x: {{ min: -1, max: 1, title: {{ display: true, text: 'Pearson Correlation' }} }} }},
        plugins: {{ title: {{ display: true, text: 'What Prediction Properties Correlate with Higher LB?' }} }}
    }}
}});

// Chart 3: Ensemble corr to best
const ensNames = {json.dumps([e['name'] for e in ensemble_results])};
const ensCorrs = {json.dumps([e['corr_to_best'] for e in ensemble_results])};
new Chart(document.getElementById('ensembleChart'), {{
    type: 'bar',
    data: {{
        labels: ensNames,
        datasets: [{{
            label: 'Correlation to Best Individual',
            data: ensCorrs,
            backgroundColor: ensCorrs.map(c => c > 0.999 ? '#ffcdd2' : (c > 0.998 ? '#fff9c4' : '#c8e6c9')),
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        scales: {{ x: {{ min: Math.min(...ensCorrs) - 0.001, max: 1.0 }} }},
        plugins: {{ title: {{ display: true, text: 'Ensemble Diversity (lower = more different from best)' }} }}
    }}
}});

// Chart 4: Diversity vs Quality for pairs
const pairData = [];
const allNames = {json.dumps(names)};
const allLBs = {json.dumps([SUBMISSIONS[n]['lb'] for n in names])};
const corrMat = {json.dumps(corr_matrix.tolist())};
for (let i = 0; i < allNames.length; i++) {{
    for (let j = i + 1; j < allNames.length; j++) {{
        pairData.push({{
            x: corrMat[i][j],
            y: (allLBs[i] + allLBs[j]) / 2,
            label: allNames[i] + ' + ' + allNames[j]
        }});
    }}
}}
new Chart(document.getElementById('diversityChart'), {{
    type: 'scatter',
    data: {{
        datasets: [{{
            label: 'Submission Pairs',
            data: pairData,
            backgroundColor: pairData.map(d => d.y > 4.05 ? '#2e7d32' : (d.y > 3.95 ? '#ff9800' : '#f44336')),
            pointRadius: 8,
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            tooltip: {{
                callbacks: {{
                    label: function(ctx) {{
                        const d = pairData[ctx.dataIndex];
                        return d.label + ' (corr=' + d.x.toFixed(5) + ', avg LB=' + d.y.toFixed(3) + ')';
                    }}
                }}
            }},
            title: {{ display: true, text: 'Pair Quality vs Diversity — Bottom-right corner is ideal (high quality, low correlation)' }}
        }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Pairwise Correlation (lower = more diverse)' }} }},
            y: {{ title: {{ display: true, text: 'Average LB Score' }} }}
        }}
    }}
}});
</script>

</body>
</html>"""

report_path = BASE / "ensemble_analysis_report.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nHTML report saved: {report_path}")
print(f"Ensemble CSVs saved: {len(saved_files)} files")

# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nBest individual: {best_name} (LB={SUBMISSIONS[best_name]['lb']})")
print(f"Most diverse pair: {min_pair} (corr={min_corr:.5f})")
print(f"\nAll {len(ensemble_results)} ensemble strategies saved as CSV files.")
print(f"HTML report: {report_path}")
print("\nSubmit the top ensemble candidates to the LB to find the true best.")
