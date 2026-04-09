#!/usr/bin/env python3
"""
Two-stage S2648 benchmark:
  Stage 1: Build & save 83-dimensional features (cached to disk).
  Stage 2: Train all models on saved features (fast, no PDB I/O).
"""
import sys, os, gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

import logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')

OUT = Path('benchmark_results_s2648')
OUT.mkdir(exist_ok=True)

FEAT_CSV = OUT / 'feature_matrix.csv'
TGT_CSV  = OUT / 'target_values.csv'
IDX_CSV  = OUT / 'valid_indices.csv'

# ── Stage 1: Build features (or load cached) ──────────────────────
if FEAT_CSV.exists() and TGT_CSV.exists():
    print("✓ Cached features found — skipping feature building.")
    X = pd.read_csv(FEAT_CSV)
    y = pd.read_csv(TGT_CSV)['ddG_experimental'].values
    print(f"  {X.shape[0]} mutations × {X.shape[1]} features")
else:
    print("═" * 72)
    print("  STAGE 1: Build 83-D circuit features for S2648")
    print("═" * 72)
    from enzyme_resistance.data.downloader import prepare_dataset
    from enzyme_resistance.train import build_feature_matrix

    t0 = time.time()
    print("  Loading S2648 dataset...", flush=True)
    df, pdb_paths = prepare_dataset(dataset='s2648')
    print(f"  {len(df)} mutations, {len(pdb_paths)} proteins", flush=True)

    print("  Building features...", flush=True)
    X, y, valid_idx = build_feature_matrix(df, pdb_paths, conductance_method='exponential')
    print(f"  {len(X)}/{len(df)} mutations built  ({time.time()-t0:.1f}s)", flush=True)

    # Save for reuse
    X.to_csv(FEAT_CSV, index=False)
    pd.DataFrame({'ddG_experimental': y}).to_csv(TGT_CSV, index=False)
    pd.DataFrame({'valid_idx': valid_idx}).to_csv(IDX_CSV, index=False)

    # Also save per-mutation pdb_ids for group-based CV
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    df_valid[['pdb_id']].to_csv(OUT / 'valid_pdb_ids.csv', index=False)
    print(f"  Features saved to {FEAT_CSV}")
    del df, pdb_paths
    gc.collect()

# ── Stage 2: Train & evaluate models ──────────────────────────────
print()
print("═" * 72)
print("  STAGE 2: Evaluate 8 models × 4 resistance-based CVs")
print("═" * 72)
print(f"  {X.shape[0]} mutations, {X.shape[1]} features", flush=True)

from enzyme_resistance.resistance_cv import precompute_all_splits, RESISTANCE_CV_STRATEGIES
from enzyme_resistance.train import _make_model, ALL_MODEL_TYPES, _evaluate_on_splits, _empty_result
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr

# Load groups
groups_csv = OUT / 'valid_pdb_ids.csv'
if groups_csv.exists():
    groups = pd.read_csv(groups_csv)['pdb_id'].values
else:
    groups = None

n_splits = 5
print(f"  Pre-computing CV splits...", flush=True)
all_splits = precompute_all_splits(X, y, n_splits=n_splits, groups=groups)
for cv_name, splits in all_splits.items():
    cv_cls = RESISTANCE_CV_STRATEGIES.get(cv_name)
    desc = cv_cls.description if cv_cls else cv_name
    print(f"    ✓ {cv_name:<32s} ({len(splits)} folds)")

X_arr = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

rows = []
print(flush=True)
print(f"  {'Model':<25s} {'CV Strategy':<35s} {'Pearson r':>10s} {'RMSE':>8s} {'R²':>8s}", flush=True)
print(f"  {'─'*25} {'─'*35} {'─'*10} {'─'*8} {'─'*8}", flush=True)

for cv_name, splits in all_splits.items():
    for mt in ALL_MODEL_TYPES:
        try:
            result = _evaluate_on_splits(
                X_arr, y, X.columns.tolist(), splits, mt, cv_name,
            )
            rows.append(result)
            print(f"  {mt:<25s} {cv_name:<35s} "
                  f"{result['pearson_r']:>+10.4f} "
                  f"{result['cv_rmse_mean']:>8.3f} "
                  f"{result['cv_r2_mean']:>+8.3f}", flush=True)
        except Exception as e:
            print(f"  {mt:<25s} {cv_name:<35s} FAILED: {e}", flush=True)
            rows.append(_empty_result(mt, cv_name, len(y)))
        gc.collect()

df_results = pd.DataFrame(rows)
df_results.to_csv(OUT / 'model_cv_comparison.csv', index=False)

# ── Summary ───────────────────────────────────────────────────────
print()
print("═" * 72)
print("  S2648 RESULTS SUMMARY")
print("═" * 72)

# Best per model
best = df_results.loc[df_results.groupby('model_type')['pearson_r'].idxmax()]
best = best.sort_values('pearson_r', ascending=False)
print(f"\n  Best per model:")
print(f"  {'Model':<25s} {'CV Strategy':<35s} {'Pearson r':>10s} {'RMSE':>8s}")
print(f"  {'─'*25} {'─'*35} {'─'*10} {'─'*8}")
for _, row in best.iterrows():
    print(f"  {row['model_type']:<25s} {row['cv_strategy']:<35s} "
          f"{row['pearson_r']:>+10.4f} {row['cv_rmse_mean']:>8.3f}")

overall = df_results.loc[df_results['pearson_r'].idxmax()]
print(f"\n  ★ BEST OVERALL: {overall['model_type']} / {overall['cv_strategy']}")
print(f"    Pearson r = {overall['pearson_r']:.4f}")
print(f"    RMSE      = {overall['cv_rmse_mean']:.4f}")
print(f"    R²        = {overall['cv_r2_mean']:.4f}")

target = 0.65
if overall['pearson_r'] >= target:
    print(f"\n  ✅ TARGET REACHED! (r = {overall['pearson_r']:.4f} ≥ {target})")
else:
    gap = target - overall['pearson_r']
    print(f"\n  ⚠  Gap to target: {gap:.4f} (need r ≥ {target})")
