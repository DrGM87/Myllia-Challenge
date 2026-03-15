"""
V23 Submission Analysis & LB Prediction Model
===============================================
Analyzes 7 V23 submission files + historical submissions with known LB scores.
Builds a regression model to predict LB from submission properties.
Generates ensemble candidates and an HTML report.

IMPORTANT: Rows 61-120 are zeroed. Only first 60 rows contain scored predictions.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
import json

BASE = Path(r"c:\Users\MSI\Desktop\projects\Myllia_Challenge")

# ============================================================
# 1. DEFINE ALL SUBMISSIONS
# ============================================================

# Historical submissions with KNOWN LB scores (for building prediction model)
HISTORICAL = {
    "V20_RUN3": {
        "file": "submission_v20_loo_20260222_180450_0_9672x3.csv",
        "lb": 4.08996, "cv": 0.9672,
        "desc": "3x pooled LOO, N=30. ALL-TIME BEST.",
    },
    "V20_RUN8": {
        "file": "submission_v20_loo_20260225_090826.csv",
        "lb": 4.06765, "cv": 0.9692,
        "desc": "3x LOO, N_ENSEMBLE=40",
    },
    "V20_cv9642": {
        "file": "submission_v20_loo_20260225_235344_0_9642.csv",
        "lb": 4.06615, "cv": 0.9642,
        "desc": "3x LOO variant",
    },
    "V20_RUN4": {
        "file": "submission_v20_loo_20260223_114020_0_9806.csv",
        "lb": 4.03252, "cv": 0.9806,
        "desc": "2x LOO, N=30",
    },
    "V22": {
        "file": "submission_v22_loo_max_20260303_094459_0_9481.csv",
        "lb": 4.05101, "cv": 0.9481,
        "desc": "4x LOO + per-channel",
    },
    "V21": {
        "file": "submission_v21_sparse_20260226_081551.csv",
        "lb": 3.79172, "cv": 0.9306,
        "desc": "Targeted 2000 genes + threshold",
    },
    "V23_R4": {
        "file": "submission_v23_r4_weighted_20260305_061415_0_9770.csv",
        "lb": 3.77166, "cv": 0.9770,
        "desc": "Weighted LOO (quality-weighted samples)",
    },
    "V23_R1": {
        "file": "submission_v23_r1_nn_matched_20260305_061415_0_9769.csv",
        "lb": 3.85489, "cv": 0.9769,
        "desc": "NN-Matched LOO",
    },
    "V23_R2": {
        "file": "submission_v23_r2_residual_20260305_061415_0_8602.csv",
        "lb": 3.58958, "cv": 0.8602,
        "desc": "Residual LOO",
    },
    "V23_R5": {
        "file": "submission_v23_r5_baseline_repl_20260305_061415_0_9430.csv",
        "lb": 4.06386, "cv": 0.9430,
        "desc": "Baseline + Replogle augmentation",
    },
    "V23_R6": {
        "file": "submission_v23_r6_weighted_repl_20260305_061415_0_9679.csv",
        "lb": 3.87387, "cv": 0.9679,
        "desc": "Weighted + Replogle augmentation",
    },
}

# V23 submissions (some without LB)
V23_SUBS = {
    "V23_R0": {
        "file": "submission_v23_r0_baseline_20260305_061415_0_9685.csv",
        "lb": None, "cv": 0.9685,
        "desc": "Baseline (V20 RUN=3 repro)",
    },
    "V23_R1": {
        "file": "submission_v23_r1_nn_matched_20260305_061415_0_9769.csv",
        "lb": 3.85489, "cv": 0.9769,
        "desc": "NN-Matched LOO",
    },
    "V23_R2": {
        "file": "submission_v23_r2_residual_20260305_061415_0_8602.csv",
        "lb": 3.58958, "cv": 0.8602,
        "desc": "Residual LOO",
    },
    "V23_R3": {
        "file": "submission_v23_r3_adaptive_20260305_061415_0_9585.csv",
        "lb": None, "cv": 0.9585,
        "desc": "Adaptive LOO",
    },
    "V23_R4": {
        "file": "submission_v23_r4_weighted_20260305_061415_0_9770.csv",
        "lb": 3.77166, "cv": 0.9770,
        "desc": "Weighted LOO (submitted)",
    },
    "V23_R5": {
        "file": "submission_v23_r5_baseline_repl_20260305_061415_0_9430.csv",
        "lb": 4.06386, "cv": 0.9430,
        "desc": "Baseline + Replogle augmentation",
    },
    "V23_R6": {
        "file": "submission_v23_r6_weighted_repl_20260305_061415_0_9679.csv",
        "lb": 3.87387, "cv": 0.9679,
        "desc": "Weighted + Replogle augmentation",
    },
}

# Merge all unique submissions
ALL_SUBS = {}
ALL_SUBS.update(HISTORICAL)
for k, v in V23_SUBS.items():
    if k not in ALL_SUBS:
        ALL_SUBS[k] = v

# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading submissions...")
dfs = {}
for name, info in ALL_SUBS.items():
    path = BASE / info["file"]
    if not path.exists():
        print(f"  MISSING: {path.name}")
        continue
    df = pd.read_csv(path)
    dfs[name] = df
    lb_str = f"LB={info['lb']:.5f}" if info['lb'] else "LB=???"
    print(f"  {name:15s}: {df.shape} | {lb_str} | CV={info['cv']:.4f}")

SCORED_ROWS = 60
pert_ids = list(dfs.values())[0].iloc[:, 0].values
gene_names = list(dfs.values())[0].columns[1:]
n_genes = len(gene_names)

matrices_full = {}
matrices = {}
for name, df in dfs.items():
    full = df.iloc[:, 1:].values.astype(np.float64)
    matrices_full[name] = full
    matrices[name] = full[:SCORED_ROWS]

names = list(matrices.keys())
num_subs = len(names)
print(f"\nLoaded {num_subs} submissions, {SCORED_ROWS} scored rows × {n_genes} genes")

# ============================================================
# 3. COMPUTE FEATURES FOR EACH SUBMISSION
# ============================================================

print("\n" + "=" * 60)
print("COMPUTING SUBMISSION FEATURES")
print("=" * 60)

def compute_features(m):
    """Compute prediction-property features from a scored matrix."""
    return {
        "mean": float(np.mean(m)),
        "std": float(np.std(m)),
        "abs_mean": float(np.mean(np.abs(m))),
        "abs_median": float(np.median(np.abs(m))),
        "pct_near_zero": float(np.mean(np.abs(m) < 0.02) * 100),
        "pct_zero": float(np.mean(m == 0) * 100),
        "min_val": float(np.min(m)),
        "max_val": float(np.max(m)),
        "range": float(np.max(m) - np.min(m)),
        "skew": float(np.mean(((m - np.mean(m)) / (np.std(m) + 1e-12)) ** 3)),
        "kurtosis": float(np.mean(((m - np.mean(m)) / (np.std(m) + 1e-12)) ** 4)),
        "iqr": float(np.percentile(m, 75) - np.percentile(m, 25)),
        "pct_positive": float(np.mean(m > 0) * 100),
        "pct_negative": float(np.mean(m < 0) * 100),
        "row_std_mean": float(np.mean(np.std(m, axis=1))),
        "col_std_mean": float(np.mean(np.std(m, axis=0))),
    }

features = {}
for name in names:
    f = compute_features(matrices[name])
    f["cv"] = ALL_SUBS[name]["cv"]
    f["lb"] = ALL_SUBS[name]["lb"]
    features[name] = f
    lb_str = f"{f['lb']:.5f}" if f['lb'] else "???"
    print(f"  {name:15s}: |pred|={f['abs_mean']:.6f}, std={f['std']:.6f}, "
          f"near0={f['pct_near_zero']:.1f}%, CV={f['cv']:.4f}, LB={lb_str}")

# ============================================================
# 4. PAIRWISE CORRELATIONS
# ============================================================

print("\n" + "=" * 60)
print("PAIRWISE CORRELATION MATRIX")
print("=" * 60)

corr_matrix = np.zeros((num_subs, num_subs))
for i in range(num_subs):
    for j in range(num_subs):
        a = matrices[names[i]].ravel()
        b = matrices[names[j]].ravel()
        corr_matrix[i, j] = np.corrcoef(a, b)[0, 1]

# Print condensed
print(f"\n{'':15s}", end="")
for j in range(num_subs):
    print(f" {names[j][:8]:>8s}", end="")
print()
for i in range(num_subs):
    print(f"{names[i]:15s}", end="")
    for j in range(num_subs):
        print(f" {corr_matrix[i,j]:8.5f}", end="")
    print()

# Correlation of V23 runs with V20_RUN3 (best LB)
if "V20_RUN3" in names:
    best_idx = names.index("V20_RUN3")
    print(f"\nCorrelation with V20_RUN3 (best LB=4.090):")
    for i, nm in enumerate(names):
        if nm.startswith("V23"):
            print(f"  {nm:15s}: {corr_matrix[best_idx, i]:.6f}")

# ============================================================
# 5. LB PREDICTION MODEL
# ============================================================

print("\n" + "=" * 60)
print("LB PREDICTION MODEL")
print("=" * 60)

# Build feature matrix from submissions with known LB
known_names = [n for n in names if features[n]["lb"] is not None]
unknown_names = [n for n in names if features[n]["lb"] is None]

print(f"\nTraining set: {len(known_names)} submissions with known LB")
print(f"Prediction set: {len(unknown_names)} submissions without LB")

# Feature columns for regression
feat_cols = ["cv", "abs_mean", "std", "pct_near_zero", "pct_zero",
             "range", "skew", "kurtosis", "iqr", "row_std_mean", "col_std_mean",
             "pct_positive", "pct_negative"]

X_train = np.array([[features[n][c] for c in feat_cols] for n in known_names])
y_train = np.array([features[n]["lb"] for n in known_names])

# Also add: correlation with best submission as a feature
if "V20_RUN3" in names:
    best_idx = names.index("V20_RUN3")
    corr_to_best_train = np.array([corr_matrix[best_idx, names.index(n)] for n in known_names])
    corr_to_best_unknown = np.array([corr_matrix[best_idx, names.index(n)] for n in unknown_names])
    X_train = np.column_stack([X_train, corr_to_best_train])
    feat_cols_ext = feat_cols + ["corr_to_best"]
else:
    feat_cols_ext = feat_cols

# Standardize features
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-12
X_train_norm = (X_train - X_mean) / X_std

# Ridge regression (manual implementation to avoid sklearn dependency)
alpha_ridge = 1.0
n_feats = X_train_norm.shape[1]
I = np.eye(n_feats)
# Add bias column
X_train_b = np.column_stack([X_train_norm, np.ones(len(known_names))])
I_b = np.eye(n_feats + 1)
I_b[-1, -1] = 0  # Don't regularize bias

beta = np.linalg.solve(X_train_b.T @ X_train_b + alpha_ridge * I_b, X_train_b.T @ y_train)

# Training predictions (LOO cross-validation)
print("\nTraining fit (Ridge, alpha=1.0):")
y_pred_train = X_train_b @ beta
for i, n in enumerate(known_names):
    print(f"  {n:15s}: actual LB={y_train[i]:.5f}, predicted={y_pred_train[i]:.5f}, "
          f"error={y_pred_train[i]-y_train[i]:+.5f}")

train_rmse = np.sqrt(np.mean((y_pred_train - y_train) ** 2))
train_corr = np.corrcoef(y_pred_train, y_train)[0, 1] if len(known_names) > 2 else 0
print(f"\n  Training RMSE: {train_rmse:.5f}")
print(f"  Training R²: {train_corr**2:.4f}")

# LOO cross-validation for honest estimate
if len(known_names) >= 4:
    loo_errors = []
    for hold in range(len(known_names)):
        mask = np.ones(len(known_names), dtype=bool)
        mask[hold] = False
        X_loo = X_train_b[mask]
        y_loo = y_train[mask]
        beta_loo = np.linalg.solve(X_loo.T @ X_loo + alpha_ridge * I_b, X_loo.T @ y_loo)
        pred_hold = X_train_b[hold] @ beta_loo
        loo_errors.append((known_names[hold], y_train[hold], pred_hold, pred_hold - y_train[hold]))

    print(f"\n  LOO Cross-Validation:")
    for n, actual, pred, err in loo_errors:
        print(f"    {n:15s}: actual={actual:.5f}, LOO_pred={pred:.5f}, error={err:+.5f}")
    loo_rmse = np.sqrt(np.mean([e[3]**2 for e in loo_errors]))
    print(f"  LOO RMSE: {loo_rmse:.5f}")

# Predict unknown LB scores
if unknown_names:
    X_unknown = np.array([[features[n][c] for c in feat_cols] for n in unknown_names])
    if "V20_RUN3" in names:
        X_unknown = np.column_stack([X_unknown, corr_to_best_unknown])
    X_unknown_norm = (X_unknown - X_mean) / X_std
    X_unknown_b = np.column_stack([X_unknown_norm, np.ones(len(unknown_names))])
    y_pred_unknown = X_unknown_b @ beta

    print(f"\nPredicted LB for unsubmitted V23 runs:")
    print(f"  {'Name':15s} {'CV':>8s} {'Pred LB':>10s} {'vs Best':>10s}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*10}")
    for i, n in enumerate(unknown_names):
        delta = y_pred_unknown[i] - 4.08996
        print(f"  {n:15s} {features[n]['cv']:8.4f} {y_pred_unknown[i]:10.5f} {delta:+10.5f}")

# Feature importance
print(f"\nFeature importance (|beta|):")
for i, col in enumerate(feat_cols_ext):
    print(f"  {col:20s}: beta={beta[i]:+.4f}")

# ============================================================
# 6. SIMPLE HEURISTIC PREDICTION (CV→LB mapping)
# ============================================================

print("\n" + "=" * 60)
print("HEURISTIC CV→LB MAPPING")
print("=" * 60)

# Fit linear regression: LB = a * CV + b
cvs_known = np.array([features[n]["cv"] for n in known_names])
lbs_known = y_train

if len(known_names) >= 3:
    cv_corr = np.corrcoef(cvs_known, lbs_known)[0, 1]
    print(f"\nCV vs LB correlation: {cv_corr:.4f}")

    # Linear fit
    A_cv = np.column_stack([cvs_known, np.ones_like(cvs_known)])
    beta_cv = np.linalg.lstsq(A_cv, lbs_known, rcond=None)[0]
    print(f"Linear fit: LB = {beta_cv[0]:.4f} * CV + {beta_cv[1]:.4f}")

    for n in known_names:
        pred = beta_cv[0] * features[n]["cv"] + beta_cv[1]
        print(f"  {n:15s}: CV={features[n]['cv']:.4f} → pred LB={pred:.5f} (actual={features[n]['lb']:.5f}, err={pred-features[n]['lb']:+.5f})")

    if unknown_names:
        print(f"\nCV-based LB predictions for unsubmitted:")
        for n in unknown_names:
            pred = beta_cv[0] * features[n]["cv"] + beta_cv[1]
            print(f"  {n:15s}: CV={features[n]['cv']:.4f} → pred LB={pred:.5f}")

# ============================================================
# 7. CORRELATION-BASED PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION-BASED LB PREDICTION")
print("=" * 60)

if "V20_RUN3" in names:
    best_idx = names.index("V20_RUN3")
    print(f"\nUsing correlation with V20_RUN3 (LB=4.090) as predictor:")

    # For known submissions: plot corr_to_best vs LB
    for n in known_names:
        idx = names.index(n)
        c = corr_matrix[best_idx, idx]
        print(f"  {n:15s}: corr={c:.6f}, LB={features[n]['lb']:.5f}")

    corrs_known = np.array([corr_matrix[best_idx, names.index(n)] for n in known_names])
    corr_lb_corr = np.corrcoef(corrs_known, lbs_known)[0, 1]
    print(f"\n  Correlation(corr_to_best, LB) = {corr_lb_corr:.4f}")

    if unknown_names:
        print(f"\n  Unsubmitted V23 runs — similarity to best:")
        for n in unknown_names:
            idx = names.index(n)
            c = corr_matrix[best_idx, idx]
            print(f"    {n:15s}: corr_to_best={c:.6f}")

# ============================================================
# 8. V23 INTER-RUN ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("V23 INTER-RUN DIVERSITY ANALYSIS")
print("=" * 60)

v23_names = [n for n in names if n.startswith("V23")]
print(f"\nV23 runs: {len(v23_names)}")

if len(v23_names) > 1:
    print(f"\n{'':12s}", end="")
    for n in v23_names:
        print(f" {n[4:]:>8s}", end="")
    print()
    for i, ni in enumerate(v23_names):
        ii = names.index(ni)
        print(f"  {ni[4:]:8s}  ", end="")
        for j, nj in enumerate(v23_names):
            jj = names.index(nj)
            print(f" {corr_matrix[ii, jj]:8.5f}", end="")
        print()

    # Find most diverse V23 pair
    min_c = 1.0
    min_pair = None
    for i in range(len(v23_names)):
        for j in range(i+1, len(v23_names)):
            ii, jj = names.index(v23_names[i]), names.index(v23_names[j])
            if corr_matrix[ii, jj] < min_c:
                min_c = corr_matrix[ii, jj]
                min_pair = (v23_names[i], v23_names[j])
    if min_pair:
        print(f"\n  Most diverse V23 pair: {min_pair[0]} + {min_pair[1]} (corr={min_c:.6f})")

# ============================================================
# 9. ENSEMBLE STRATEGIES (V23 + Historical)
# ============================================================

print("\n" + "=" * 60)
print("ENSEMBLE STRATEGIES")
print("=" * 60)

ensemble_results = []

# Strategy 1: V23 R0 (baseline repro) — should be closest to V20_RUN3
if "V23_R0" in matrices:
    ensemble_results.append({
        "name": "V23_R0 (Baseline Repro)",
        "desc": "Should reproduce V20 RUN=3 (~4.09 LB)",
        "matrix": matrices["V23_R0"],
        "weights": {"V23_R0": 1.0},
    })

# Strategy 2: Average V23 R0 + V20_RUN3
if "V23_R0" in matrices and "V20_RUN3" in matrices:
    avg = 0.5 * matrices["V23_R0"] + 0.5 * matrices["V20_RUN3"]
    ensemble_results.append({
        "name": "V23_R0 + V20_RUN3 (0.5/0.5)",
        "desc": "Average of baseline repro with original best",
        "matrix": avg,
        "weights": {"V23_R0": 0.5, "V20_RUN3": 0.5},
    })

# Strategy 3: V20_RUN3 heavy + V23_R1 light (NN-matched, second best CV)
if "V23_R1" in matrices and "V20_RUN3" in matrices:
    avg = 0.7 * matrices["V20_RUN3"] + 0.3 * matrices["V23_R1"]
    ensemble_results.append({
        "name": "0.7×V20_RUN3 + 0.3×V23_R1",
        "desc": "Best LB dominant + NN-matched diversity",
        "matrix": avg,
        "weights": {"V20_RUN3": 0.7, "V23_R1": 0.3},
    })

# Strategy 4: Average all non-Replogle V23 runs
non_repl = [n for n in v23_names if "repl" not in n.lower() and "R2" not in n]
if non_repl:
    avg = np.mean([matrices[n] for n in non_repl], axis=0)
    ensemble_results.append({
        "name": f"V23 Non-Repl Average ({len(non_repl)} runs)",
        "desc": f"Average of {', '.join(non_repl)}",
        "matrix": avg,
        "weights": {n: 1.0/len(non_repl) for n in non_repl},
    })

# Strategy 5: Average all good V23 runs + V20_RUN3
good_v23 = [n for n in v23_names if n not in ("V23_R2", "V23_R5")]  # excl residual & Replogle
if good_v23 and "V20_RUN3" in matrices:
    all_good = good_v23 + ["V20_RUN3"]
    avg = np.mean([matrices[n] for n in all_good], axis=0)
    ensemble_results.append({
        "name": f"Good V23 + V20_RUN3 ({len(all_good)} subs)",
        "desc": f"Average of {', '.join(all_good)}",
        "matrix": avg,
        "weights": {n: 1.0/len(all_good) for n in all_good},
    })

# Strategy 6: LB-weighted historical + V23_R0
hist_with_lb = {n: ALL_SUBS[n] for n in known_names if n != "V23_R4"}
if hist_with_lb and "V23_R0" in matrices:
    lb_vals = np.array([ALL_SUBS[n]["lb"] for n in hist_with_lb])
    lb_w = lb_vals / lb_vals.sum()
    avg = np.zeros_like(matrices[names[0]])
    for i, n in enumerate(hist_with_lb):
        if n in matrices:
            avg += lb_w[i] * matrices[n]
    # Blend with V23_R0
    blend = 0.5 * avg + 0.5 * matrices["V23_R0"]
    ensemble_results.append({
        "name": "LB-Weighted Historical + V23_R0",
        "desc": "50% LB-weighted average of all with known LB + 50% V23 baseline",
        "matrix": blend,
        "weights": {"V23_R0": 0.5, **{n: 0.5*float(lb_w[i]) for i, n in enumerate(hist_with_lb) if n in matrices}},
    })

# Strategy 7: V20_RUN3 + V22 (two best LB, decent diversity)
if "V20_RUN3" in matrices and "V22" in matrices:
    avg = 0.6 * matrices["V20_RUN3"] + 0.4 * matrices["V22"]
    ensemble_results.append({
        "name": "0.6×V20_RUN3 + 0.4×V22",
        "desc": "Two best LB submissions blended",
        "matrix": avg,
        "weights": {"V20_RUN3": 0.6, "V22": 0.4},
    })

# Strategy 8: Top-3 historical by LB
sorted_hist = sorted(known_names, key=lambda n: ALL_SUBS[n]["lb"], reverse=True)
top3_hist = sorted_hist[:3]
if len(top3_hist) == 3:
    avg = np.mean([matrices[n] for n in top3_hist], axis=0)
    ensemble_results.append({
        "name": f"Top-3 Historical ({', '.join(top3_hist)})",
        "desc": "Average of 3 best LB submissions",
        "matrix": avg,
        "weights": {n: 1/3 for n in top3_hist},
    })

# Compute ensemble stats
print(f"\n{'#':>3s} {'Strategy':50s} {'Corr→Best':>10s} {'|Pred|':>8s} {'Near0%':>8s}")
print(f"{'---':>3s} {'-'*50} {'-'*10} {'-'*8} {'-'*8}")

if "V20_RUN3" in matrices:
    best_m = matrices["V20_RUN3"]
else:
    best_m = matrices[names[0]]

for i, ens in enumerate(ensemble_results):
    m = ens["matrix"]
    corr = np.corrcoef(m.ravel(), best_m.ravel())[0, 1]
    abs_mean = np.mean(np.abs(m))
    near0 = np.mean(np.abs(m) < 0.02) * 100
    ens["corr_to_best"] = float(corr)
    ens["abs_mean"] = float(abs_mean)
    ens["pct_near_zero"] = float(near0)
    print(f"{i+1:3d} {ens['name']:50s} {corr:10.6f} {abs_mean:8.6f} {near0:7.1f}%")

# ============================================================
# 10. SAVE ENSEMBLE SUBMISSIONS
# ============================================================

print("\n" + "=" * 60)
print("SAVING ENSEMBLE SUBMISSIONS")
print("=" * 60)

template_df = list(dfs.values())[0].copy()
saved_files = []

for i, ens in enumerate(ensemble_results):
    safe_name = ens["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
    fname = f"v23_ens_{i:02d}_{safe_name[:50]}.csv"
    out_path = BASE / fname

    # Build full 120-row submission
    full_matrix = np.zeros_like(matrices_full[names[0]])
    for nm, w in ens["weights"].items():
        if w > 0 and nm in matrices_full:
            full_matrix += w * matrices_full[nm]

    out_df = template_df.copy()
    out_df.iloc[:, 1:] = full_matrix
    out_df.to_csv(out_path, index=False)
    saved_files.append(fname)
    ens["saved_file"] = fname
    print(f"  Saved: {fname}")

# ============================================================
# 11. GENERATE HTML REPORT
# ============================================================

print("\n" + "=" * 60)
print("GENERATING HTML REPORT")
print("=" * 60)

# Build individual table rows
individual_rows = ""
for name in sorted(names, key=lambda n: ALL_SUBS[n]["lb"] if ALL_SUBS[n]["lb"] else 0, reverse=True):
    f = features[name]
    info = ALL_SUBS[name]
    lb_str = f"{info['lb']:.5f}" if info['lb'] else "<em>unknown</em>"
    highlight = ' class="highlight"' if info.get("lb") and info["lb"] > 4.05 else ""
    individual_rows += f"""
    <tr{highlight}>
        <td><strong>{name}</strong></td>
        <td>{lb_str}</td>
        <td>{info['cv']:.4f}</td>
        <td>{f['abs_mean']:.6f}</td>
        <td>{f['std']:.6f}</td>
        <td>{f['pct_near_zero']:.1f}%</td>
        <td>{info['desc']}</td>
    </tr>"""

# Build ensemble table rows
ens_rows = ""
for i, ens in enumerate(ensemble_results):
    weights_str = ", ".join(f"{k}:{v:.2f}" for k, v in ens["weights"].items() if v > 0)
    ens_rows += f"""
    <tr>
        <td>{i+1}</td>
        <td><strong>{ens['name']}</strong></td>
        <td>{ens['desc']}</td>
        <td>{ens['corr_to_best']:.6f}</td>
        <td>{ens['abs_mean']:.6f}</td>
        <td>{ens['pct_near_zero']:.1f}%</td>
        <td style="font-size:11px">{weights_str}</td>
        <td>{ens.get('saved_file','')}</td>
    </tr>"""

# Heatmap (V23 runs only)
v23_heatmap = ""
for i, ni in enumerate(v23_names):
    ii = names.index(ni)
    v23_heatmap += "<tr>"
    v23_heatmap += f"<td><strong>{ni}</strong></td>"
    for j, nj in enumerate(v23_names):
        jj = names.index(nj)
        val = corr_matrix[ii, jj]
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
        elif val > 0.95:
            color = "#ffe0b2"
        else:
            color = "#aaffaa"
        v23_heatmap += f'<td style="background:{color};text-align:center;font-size:12px;padding:4px">{val:.5f}</td>'
    v23_heatmap += "</tr>"

v23_headers = "".join(f"<th style='font-size:11px;padding:4px'>{n}</th>" for n in v23_names)

# LB prediction table
pred_rows = ""
if unknown_names:
    X_unknown = np.array([[features[n][c] for c in feat_cols] for n in unknown_names])
    if "V20_RUN3" in names:
        X_unknown = np.column_stack([X_unknown, corr_to_best_unknown])
    X_unknown_norm = (X_unknown - X_mean) / X_std
    X_unknown_b = np.column_stack([X_unknown_norm, np.ones(len(unknown_names))])
    y_pred_unknown = X_unknown_b @ beta

    # Also CV-based prediction
    for i, n in enumerate(unknown_names):
        ridge_pred = y_pred_unknown[i]
        cv_pred = beta_cv[0] * features[n]["cv"] + beta_cv[1] if len(known_names) >= 3 else None
        avg_pred = (ridge_pred + cv_pred) / 2 if cv_pred else ridge_pred
        corr_best = corr_matrix[best_idx, names.index(n)] if "V20_RUN3" in names else None
        corr_str = f"{corr_best:.6f}" if corr_best else "—"
        cv_str = f"{cv_pred:.5f}" if cv_pred else "—"
        pred_rows += f"""
        <tr>
            <td><strong>{n}</strong></td>
            <td>{features[n]['cv']:.4f}</td>
            <td>{corr_str}</td>
            <td>{ridge_pred:.5f}</td>
            <td>{cv_str}</td>
            <td><strong>{avg_pred:.5f}</strong></td>
            <td>{ALL_SUBS[n]['desc']}</td>
        </tr>"""

# Scatter data for chart
scatter_data = []
for n in names:
    scatter_data.append({
        "name": n,
        "lb": ALL_SUBS[n]["lb"] if ALL_SUBS[n]["lb"] else None,
        "cv": ALL_SUBS[n]["cv"],
        "abs_mean": features[n]["abs_mean"],
        "corr_to_best": float(corr_matrix[names.index("V20_RUN3") if "V20_RUN3" in names else 0, names.index(n)]),
        "is_v23": n.startswith("V23"),
    })
scatter_json = json.dumps(scatter_data)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>V23 Submission Analysis & LB Prediction</title>
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
    .highlight {{ background: #e8f5e9 !important; }}
    .metric {{ font-size: 28px; font-weight: bold; color: #1a237e; }}
    .metric-label {{ font-size: 12px; color: #666; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
    .insight {{ background: #e3f2fd; border-left: 4px solid #1565c0; padding: 12px 16px; margin: 10px 0; border-radius: 0 8px 8px 0; }}
    .warning {{ background: #fff3e0; border-left: 4px solid #e65100; padding: 12px 16px; margin: 10px 0; border-radius: 0 8px 8px 0; }}
    .recommendation {{ background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 12px 16px; margin: 10px 0; border-radius: 0 8px 8px 0; }}
    .chart-container {{ position: relative; height: 400px; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head>
<body>
<div class="container">

<h1>V23 Submission Analysis & LB Prediction</h1>
<p>Generated: 2026-03-05 | V23 runs: {len(v23_names)} | Historical: {len(known_names)} with known LB | Best LB: 4.08996 (V20_RUN3)</p>

<div class="grid">
    <div class="card" style="text-align:center">
        <div class="metric">4.08996</div>
        <div class="metric-label">Best Individual LB (V20_RUN3)</div>
    </div>
    <div class="card" style="text-align:center">
        <div class="metric">{num_subs}</div>
        <div class="metric-label">Total Submissions</div>
    </div>
    <div class="card" style="text-align:center">
        <div class="metric">{len(v23_names)}</div>
        <div class="metric-label">V23 Runs</div>
    </div>
    <div class="card" style="text-align:center">
        <div class="metric">{len(ensemble_results)}</div>
        <div class="metric-label">Ensemble Strategies</div>
    </div>
</div>

<h2>1. All Submissions</h2>
<div class="card">
    <table>
        <tr><th>Name</th><th>LB Score</th><th>CV</th><th>|Pred| Mean</th><th>Std</th><th>Near-Zero</th><th>Description</th></tr>
        {individual_rows}
    </table>
</div>

<h2>2. LB Predictions for Unsubmitted V23 Runs</h2>
<div class="card">
    <p>Ridge regression + CV-based prediction trained on {len(known_names)} submissions with known LB scores.</p>
    <table>
        <tr><th>Name</th><th>CV</th><th>Corr→Best</th><th>Ridge Pred</th><th>CV Pred</th><th>Avg Pred</th><th>Description</th></tr>
        {pred_rows}
    </table>
</div>

<div class="warning">
    <strong>Caveat:</strong> LB prediction models trained on {len(known_names)} samples are unreliable.
    The predictions are rough estimates. Only LB submission provides ground truth.
    The model captures broad patterns (e.g. low correlation with best → lower LB) but cannot predict precise scores.
</div>

<h2>3. V23 Inter-Run Correlation Heatmap</h2>
<div class="card">
    <p>How similar are the 7 V23 submissions to each other? Lower correlation = more diverse predictions.</p>
    <table style="width:auto">
        <tr><th></th>{v23_headers}</tr>
        {v23_heatmap}
    </table>
</div>

<h2>4. CV vs LB (Known + Predicted)</h2>
<div class="card">
    <div class="chart-container">
        <canvas id="cvLbChart"></canvas>
    </div>
</div>

<h2>5. Ensemble Strategies</h2>
<div class="card">
    <table>
        <tr><th>#</th><th>Strategy</th><th>Description</th><th>Corr→Best</th><th>|Pred|</th><th>Near-Zero</th><th>Weights</th><th>File</th></tr>
        {ens_rows}
    </table>
</div>

<h2>6. Recommendations</h2>

<div class="recommendation">
    <strong>Priority submissions to LB:</strong>
    <ol>
        <li><strong>V23_R0 (Baseline Repro)</strong> — Most important to submit. If this matches V20_RUN3's LB (~4.09), it confirms the V23 notebook is working correctly and the other V23 strategies genuinely helped/hurt.</li>
        <li><strong>V23_R1 (NN-Matched)</strong> — Second-best V23 CV (0.9769). If the CV→LB correlation holds, this should score close to R4 but with different predictions.</li>
        <li><strong>0.6×V20_RUN3 + 0.4×V22 ensemble</strong> — Blends the two best known-LB submissions with moderate diversity.</li>
    </ol>
</div>

<div class="insight">
    <strong>Key insight:</strong> V23 R4 (Weighted LOO, LB=3.77) scored 0.32 points below V20 RUN3 (LB=4.09) despite higher CV (0.977 vs 0.967).
    This confirms the pattern: LOO quality filtering inflates CV without improving LB.
    The simplest LOO approach (V20 RUN=3) remains the strongest on the leaderboard.
</div>

</div>

<script>
const data = {scatter_json};
const knownPts = data.filter(d => d.lb !== null);
const unknownPts = data.filter(d => d.lb === null);

new Chart(document.getElementById('cvLbChart'), {{
    type: 'scatter',
    data: {{
        datasets: [
            {{
                label: 'Known LB (Historical)',
                data: knownPts.filter(d => !d.is_v23).map(d => ({{x: d.cv, y: d.lb}})),
                backgroundColor: '#1565c0',
                pointRadius: 10,
            }},
            {{
                label: 'Known LB (V23)',
                data: knownPts.filter(d => d.is_v23).map(d => ({{x: d.cv, y: d.lb}})),
                backgroundColor: '#f44336',
                pointRadius: 12,
                pointStyle: 'triangle',
            }},
            {{
                label: 'Unknown LB (V23)',
                data: unknownPts.map(d => ({{x: d.cv, y: 3.9}})),
                backgroundColor: '#ff980080',
                pointRadius: 8,
                pointStyle: 'rect',
            }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            tooltip: {{
                callbacks: {{
                    label: function(ctx) {{
                        const idx = ctx.dataIndex;
                        const ds = ctx.dataset;
                        const subset = ds.label.includes('Unknown') ? unknownPts :
                                       (ds.label.includes('V23') ? knownPts.filter(d => d.is_v23) : knownPts.filter(d => !d.is_v23));
                        const d = subset[idx];
                        return d.name + ' (CV=' + d.cv.toFixed(4) + (d.lb ? ', LB=' + d.lb.toFixed(3) : ', LB=???') + ')';
                    }}
                }}
            }},
            title: {{ display: true, text: 'CV vs LB Score — Historical Pattern + V23 Results' }}
        }},
        scales: {{
            x: {{ title: {{ display: true, text: 'CV Score' }} }},
            y: {{ title: {{ display: true, text: 'LB Score' }}, min: 3.5, max: 4.2 }}
        }}
    }}
}});
</script>

</body>
</html>"""

report_path = BASE / "v23_analysis_report.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nHTML report saved: {report_path}")

# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nBest individual LB: V20_RUN3 (4.08996)")
print(f"V23 submitted: R4 Weighted (LB=3.77166, CV=0.9770)")
print(f"V23 best CV: R4 Weighted (0.9770) and R1 NN-Matched (0.9769)")
print(f"\n{len(ensemble_results)} ensemble strategies saved as CSV files")
print(f"HTML report: {report_path}")
print(f"\nRECOMMENDATION: Submit V23_R0 (baseline repro) to calibrate V23 vs V20 gap.")
