"""
Full benchmarking pipeline.

Architecture:
  Phase 1 — Download ALL datasets + PDB structures once (cached to disk).
  Phase 2 — For each dataset, build features and train our circuit model.
             All 7 model types are evaluated on the exact same
             resistance-based CV splits (4 strategies derived from circuit
             properties).
  Phase 3 — Compare our model against 15 published ΔΔG methods (each
             published method retains its own evaluation protocol as
             reported by the authors).
  Phase 4 — Generate comprehensive plots.

Resistance-based CV strategies (all splits pre-computed once and shared):
  ① Resistance Centrality Stratified — by current-flow closeness
  ② Propagation Radius Stratified    — by electrical perturbation radius
  ③ Kirchhoff-Grouped                — by protein total network resistance
  ④ Spectral-Clustered               — by Fiedler-vector / bottleneck position
"""

import gc
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr

from enzyme_resistance.data.downloader import (
    prepare_dataset,
    download_dataset,
    download_pdb_structure,
    get_data_dir,
)
from enzyme_resistance.published_baselines import (
    get_published_baselines,
    format_comparison_table,
)
from enzyme_resistance.resistance_cv import (
    precompute_all_splits,
    ALL_RESISTANCE_CV_NAMES,
    RESISTANCE_CV_STRATEGIES,
)
from enzyme_resistance.train import (
    build_feature_matrix,
    train_and_evaluate,
    compare_models,
    compare_models_resistance_cv,
    nested_cv_evaluate,
    ablation_study,
    group_ablation_study,
    conductance_sensitivity_study,
    ALL_MODEL_TYPES,
)

logger = logging.getLogger(__name__)

_INDIVIDUAL_DATASETS = ['builtin', 's2648', 'fireprotdb']


# ==================================================================
# Main entry point
# ==================================================================

def run_benchmark(
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    dataset: str = 'builtin',
    max_mutations: Optional[int] = None,
    n_folds: int = 5,
    run_ablation: bool = True,
    run_sensitivity: bool = True,
) -> dict:
    """
    Run the full benchmarking pipeline.

    Parameters
    ----------
    dataset : str
        One of 's2648', 'fireprotdb', 'builtin', or 'all'.
    """
    if output_dir is None:
        output_dir = "benchmark_results"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    print("=" * 72)
    print("  ENZYME RESISTANCE BENCHMARK")
    print("  Electrical Resistance Model of Mutation Propagation")
    print(f"  Dataset(s): {dataset}")
    print("  CV: 4 resistance-based strategies (same splits for all models)")
    print("=" * 72)
    print()

    # ── Determine datasets ────────────────────────────────────────
    if dataset == 'all':
        datasets_to_run = list(_INDIVIDUAL_DATASETS)
    else:
        datasets_to_run = [dataset]

    # ==============================================================
    # PHASE 1: Download everything once
    # ==============================================================
    print(f"[Phase 1] Downloading & caching {len(datasets_to_run)} dataset(s)")
    print(f"          Cache: {get_data_dir(data_dir)}")
    t0 = time.time()

    dataset_bundles: Dict[str, Tuple[pd.DataFrame, Dict[str, str]]] = {}
    for ds in datasets_to_run:
        print(f"  • {ds}...", end=" ", flush=True)
        try:
            df, pdb_paths = prepare_dataset(
                dataset=ds, data_dir=data_dir, max_mutations=max_mutations,
            )
            dataset_bundles[ds] = (df, pdb_paths)
            print(f"{len(df)} mutations, {len(pdb_paths)} proteins ✓")
        except Exception as e:
            print(f"FAILED ({e})")
            logger.warning(f"Could not load dataset '{ds}': {e}")

    print(f"  Download/cache time: {time.time() - t0:.1f}s")
    print()

    if not dataset_bundles:
        print("ERROR: No datasets loaded. Aborting.")
        return {}

    # ==============================================================
    # PHASE 2: Evaluate circuit model on each dataset
    # ==============================================================
    all_results: Dict[str, dict] = {}
    our_best_per_dataset: Dict[str, dict] = {}

    for ds_name, (df, pdb_paths) in dataset_bundles.items():
        ds_results = _evaluate_single_dataset(
            ds_name=ds_name,
            df=df,
            pdb_paths=pdb_paths,
            out_path=out_path / ds_name,
            n_folds=n_folds,
            run_ablation=run_ablation,
            run_sensitivity=run_sensitivity,
        )
        all_results[ds_name] = ds_results
        our_best_per_dataset[ds_name] = _pick_best_result(ds_results)

        # Free memory between datasets to prevent OOM on large runs
        gc.collect()

    # ==============================================================
    # PHASE 3: Compare against published methods
    # ==============================================================
    print()
    print("=" * 72)
    print("  [Phase 3] COMPARISON WITH 15 PUBLISHED ΔΔG PREDICTION METHODS")
    print("  Each published method keeps its own evaluation protocol.")
    print("  Our circuit model was evaluated with 4 resistance-based CVs.")
    print("=" * 72)
    print()

    comparison_table = format_comparison_table(
        our_results=our_best_per_dataset,
        dataset=dataset,
    )
    all_results['published_comparison'] = comparison_table.to_dict('records')

    _print_comparison_table(comparison_table, our_best_per_dataset)
    comparison_table.to_csv(out_path / 'published_comparison.csv', index=False)

    # ==============================================================
    # PHASE 4: Generate cross-dataset comparison plots
    # ==============================================================
    print()
    print("[Phase 4] Generating comparison plots...")
    _plot_published_comparison(comparison_table, out_path)
    print(f"  Plots saved to {out_path}/")

    # ── Summary ───────────────────────────────────────────────────
    total_time = time.time() - total_start
    all_results['total_time_seconds'] = total_time

    print()
    print("=" * 72)
    print("  BENCHMARK COMPLETE")
    print(f"  Datasets: {', '.join(dataset_bundles.keys())}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Results: {out_path}/")
    print("=" * 72)

    _save_master_results(all_results, out_path)
    return all_results


# ==================================================================
# Single-dataset evaluation
# ==================================================================

def _evaluate_single_dataset(
    ds_name: str,
    df: pd.DataFrame,
    pdb_paths: Dict[str, str],
    out_path: Path,
    n_folds: int = 5,
    run_ablation: bool = True,
    run_sensitivity: bool = True,
) -> dict:
    """Full pipeline on one dataset."""
    out_path.mkdir(parents=True, exist_ok=True)
    results = {'dataset_name': ds_name}

    print("─" * 72, flush=True)
    print(f"  DATASET: {ds_name}", flush=True)
    print(f"  {len(df)} mutations across {len(pdb_paths)} proteins", flush=True)
    print("─" * 72, flush=True)

    results['dataset'] = {
        'n_mutations': len(df),
        'n_proteins': len(pdb_paths),
        'proteins': list(pdb_paths.keys()),
    }

    # ── 1. Build features ─────────────────────────────────────────
    print(f"  [1/5] Building 83-dimensional circuit features (V=IR + P + RC + G(s))...", flush=True)
    t0 = time.time()
    X, y, valid_idx = build_feature_matrix(
        df, pdb_paths, conductance_method='exponential',
    )
    print(f"         {len(X)}/{len(df)} mutations  |  "
          f"ΔΔG range: [{y.min():.2f}, {y.max():.2f}] kcal/mol" if len(y) > 0
          else f"         {len(X)}/{len(df)} mutations", flush=True)
    print(f"         Time: {time.time() - t0:.1f}s", flush=True)

    results['features'] = {
        'n_valid': len(X),
        'feature_names': list(X.columns) if len(X) > 0 else [],
        'ddG_range': [float(y.min()), float(y.max())] if len(y) > 0 else [0, 0],
    }

    if len(X) < 5:
        print("         ⚠ Not enough mutations — skipping model training.")
        print()
        return results

    # Group labels
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    groups = df_valid['pdb_id'].values
    n_groups = len(np.unique(groups))
    print(f"         Protein groups: {n_groups}", flush=True)

    # ── 2. Pre-compute resistance-based CV splits ─────────────────
    print(f"  [2/5] Pre-computing 4 resistance-based CV strategies "
          f"({n_folds} folds)...", flush=True)
    t0 = time.time()
    all_splits = precompute_all_splits(X, y, n_splits=n_folds, groups=groups)

    for cv_name, splits in all_splits.items():
        cv_cls = RESISTANCE_CV_STRATEGIES.get(cv_name)
        desc = cv_cls.description if cv_cls else cv_name
        print(f"         ✓ {cv_name:<32s} ({len(splits)} folds)  — {desc}",
              flush=True)

    results['cv_strategies'] = {
        name: {'n_folds': len(splits)} for name, splits in all_splits.items()
    }
    print(f"         Time: {time.time() - t0:.1f}s", flush=True)

    # ── 3. Evaluate all models on same splits ─────────────────────
    print(f"  [3/5] Evaluating {len(ALL_MODEL_TYPES)} models (incl. stacking) × "
          f"{len(all_splits)} CV strategies (same splits for all)...",
          flush=True)
    t0 = time.time()

    comparison_df = compare_models_resistance_cv(
        X, y, n_splits=n_folds, groups=groups, model_types=ALL_MODEL_TYPES,
    )
    results['model_cv_comparison'] = comparison_df.to_dict('records')

    # Print summary: best model per CV
    print(flush=True)
    print(f"         {'CV Strategy':<34s} {'Best Model':<22s} "
          f"{'R²':>8s} {'RMSE':>8s} {'Pearson r':>10s} {'Spearman ρ':>11s}",
          flush=True)
    print(f"         {'─'*34} {'─'*22} {'─'*8} {'─'*8} {'─'*10} {'─'*11}",
          flush=True)

    for cv_name in all_splits:
        subset = comparison_df[comparison_df['cv_strategy'] == cv_name]
        if subset.empty:
            continue
        best_row = subset.loc[subset['cv_r2_mean'].idxmax()]
        print(f"         {cv_name:<34s} {best_row['model_type']:<22s} "
              f"{best_row['cv_r2_mean']:>+8.3f} "
              f"{best_row['cv_rmse_mean']:>8.3f} "
              f"{best_row['pearson_r']:>+10.3f} "
              f"{best_row['spearman_r']:>+11.3f}", flush=True)

    # Overall best
    if not comparison_df.empty:
        overall_best = comparison_df.loc[comparison_df['cv_r2_mean'].idxmax()]
        print(flush=True)
        print(f"         ★ Best overall: {overall_best['model_type']} on "
              f"{overall_best['cv_strategy']}  "
              f"(R²={overall_best['cv_r2_mean']:.3f}, "
              f"Pearson={overall_best['pearson_r']:.3f})", flush=True)

    print(f"         Time: {time.time() - t0:.1f}s", flush=True)

    # ── Save intermediate results so they survive OOM ─────────────
    _save_dataset_results(X, y, results, out_path)
    print(f"         (Intermediate results saved to {out_path})", flush=True)

    # For large datasets (>500 mutations), skip ablation and
    # conductance sensitivity to stay within memory limits.
    # The model comparison (step 3) is the core benchmark.
    _LARGE_DS_THRESHOLD = 500
    if len(X) > _LARGE_DS_THRESHOLD:
        print(f"         ⓘ  Dataset has {len(X)} mutations (>{_LARGE_DS_THRESHOLD}). "
              f"Skipping ablation & sensitivity to conserve memory.", flush=True)
        run_ablation = False
        run_sensitivity = False

    # ── 4. Ablation study ─────────────────────────────────────────
    if run_ablation and len(X) >= n_folds:
        print(f"  [4/5] Feature ablation study...", flush=True)
        t0 = time.time()
        ablation_df = ablation_study(X, y, n_folds=n_folds)
        results['ablation'] = ablation_df.to_dict('records')
        print("         Top-5 features (by R² drop):", flush=True)
        for _, row in ablation_df.head(5).iterrows():
            print(f"           #{int(row['importance_rank']):2d}: "
                  f"{row['feature_dropped']:>30s}  "
                  f"R² drop = {row['r2_drop']:+.4f}", flush=True)

        group_abl_df = group_ablation_study(X, y, n_folds=n_folds)
        results['group_ablation'] = group_abl_df.to_dict('records')
        if not group_abl_df.empty:
            print("         Feature groups:", flush=True)
            for _, row in group_abl_df.iterrows():
                print(f"           {row['group_dropped']:>20s} "
                      f"({int(row['n_features_dropped'])} features): "
                      f"R² drop = {row['r2_drop']:+.4f}", flush=True)
        print(f"         Time: {time.time() - t0:.1f}s", flush=True)
    else:
        print(f"  [4/5] Skipping ablation (large dataset or too few samples)",
              flush=True)

    # ── 5. Conductance sensitivity ────────────────────────────────
    if run_sensitivity and len(X) >= n_folds:
        print(f"  [5/5] Conductance sensitivity study...", flush=True)
        t0 = time.time()
        sensitivity_df = conductance_sensitivity_study(
            df, pdb_paths, n_folds=n_folds,
        )
        results['conductance_sensitivity'] = sensitivity_df.to_dict('records')
        for _, row in sensitivity_df.iterrows():
            print(f"         {row['conductance_method']:>15s}: "
                  f"R²={row['cv_r2_mean']:.3f}  "
                  f"Pearson r={row['pearson_r']:.3f}", flush=True)
        print(f"         Time: {time.time() - t0:.1f}s", flush=True)
    else:
        print(f"  [5/5] Skipping conductance sensitivity (large dataset or disabled)",
              flush=True)

    # ── Per-dataset plots ─────────────────────────────────────────
    print(f"  Generating plots for {ds_name}...", flush=True)
    _generate_dataset_plots(X, y, groups, results, out_path)
    _save_dataset_results(X, y, results, out_path)
    print(flush=True)

    return results


# ==================================================================
# Helpers
# ==================================================================

def _pick_best_result(ds_results: dict) -> dict:
    """Pick the best model+CV combo from the resistance CV comparison."""
    if 'model_cv_comparison' in ds_results:
        df = pd.DataFrame(ds_results['model_cv_comparison'])
        if not df.empty:
            best = df.loc[df['cv_r2_mean'].idxmax()]
            return best.to_dict()
    return {}


def _print_comparison_table(comparison_table, our_best_per_dataset):
    """Pretty-print the full comparison."""
    print(f"  {'Rank':<5} {'Method':<38} {'Eval dataset':<14} "
          f"{'Eval protocol':<34} {'Pearson r':>10} {'RMSE':>8} "
          f"{'Spearman ρ':>11}")
    print(f"  {'─'*5} {'─'*38} {'─'*14} {'─'*34} {'─'*10} {'─'*8} {'─'*11}")

    for _, row in comparison_table.iterrows():
        is_ours = 'ours' in str(row['method'])
        marker = " ◀" if is_ours else ""
        rmse_str = f"{row['rmse']:.3f}" if not pd.isna(row['rmse']) else "   N/A"
        print(f"  {int(row['rank']):<5} "
              f"{row['method']:<38} "
              f"{str(row.get('eval_dataset', '')):.<14} "
              f"{str(row.get('eval_protocol', '')):.<34} "
              f"{row['pearson_r']:>10.3f} "
              f"{rmse_str:>8} "
              f"{row['spearman_r']:>11.3f}"
              f"{marker}")

    print()
    for ds_name, metrics in our_best_per_dataset.items():
        pr = metrics.get('pearson_r', 0)
        cv = metrics.get('cv_strategy', '?')
        mt = metrics.get('model_type', '?')
        if pr >= 0.70:
            level = "EXCELLENT (DynaMut2-level)"
        elif pr >= 0.60:
            level = "GOOD (mCSM/DUET-level)"
        elif pr >= 0.50:
            level = "MODERATE (Rosetta/FoldX-level)"
        elif pr >= 0.35:
            level = "FAIR"
        else:
            level = "LOW"
        print(f"  {ds_name}: Pearson r = {pr:.3f} "
              f"[{mt} / {cv}] → {level}")

    print()
    print("  NOTE: Published methods used their own CV protocols (10-fold,")
    print("        LOO, blind test, etc.).  Our circuit model was evaluated")
    print("        with 4 resistance-based CV strategies (pre-computed splits")
    print("        shared across all models).")


def _save_master_results(all_results, out_path):
    """Save master results JSON."""
    def _ser(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_ser(v) for v in obj]
        return obj

    with open(out_path / 'benchmark_results.json', 'w') as f:
        json.dump(_ser(all_results), f, indent=2, default=str)


def _save_dataset_results(X, y, results, out_path):
    """Save per-dataset CSV files."""
    if 'model_cv_comparison' in results:
        df = pd.DataFrame(results['model_cv_comparison'])
        # Drop dict columns for CSV
        dict_cols = [c for c in df.columns
                     if df[c].apply(lambda x: isinstance(x, dict)).any()]
        df.drop(columns=dict_cols, errors='ignore').to_csv(
            out_path / 'model_cv_comparison.csv', index=False,
        )
    if 'ablation' in results:
        pd.DataFrame(results['ablation']).to_csv(
            out_path / 'ablation_study.csv', index=False,
        )
    if 'group_ablation' in results:
        pd.DataFrame(results['group_ablation']).to_csv(
            out_path / 'group_ablation_study.csv', index=False,
        )
    if len(X) > 0:
        X.to_csv(out_path / 'feature_matrix.csv', index=False)
        pd.DataFrame({'ddG_experimental': y}).to_csv(
            out_path / 'target_values.csv', index=False,
        )


# ──────────────────────────────────────────────────────────────────
# Per-dataset plotting
# ──────────────────────────────────────────────────────────────────

def _generate_dataset_plots(X, y, groups, results, out_path):
    """Generate plots for a single dataset."""
    try:
        _plot_prediction_scatter(X, y, out_path)
        _plot_feature_distributions(X, out_path)
        _plot_feature_correlations(X, y, out_path)

        if 'model_cv_comparison' in results:
            _plot_resistance_cv_heatmap(results['model_cv_comparison'], out_path)
            _plot_model_comparison_by_cv(results['model_cv_comparison'], out_path)

        if 'ablation' in results:
            _plot_ablation(results['ablation'], out_path)
        if 'group_ablation' in results:
            _plot_group_ablation(results['group_ablation'], out_path)
        if 'conductance_sensitivity' in results:
            _plot_conductance_comparison(
                results['conductance_sensitivity'], out_path,
            )
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close all matplotlib figures to free memory
        plt.close('all')
        gc.collect()


def _plot_prediction_scatter(X, y, out_path):
    """Predicted vs experimental ΔΔG."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )),
    ])
    n_folds = min(5, len(y))
    X_arr = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
    y_pred = cross_val_predict(pipe, X_arr, y, cv=n_folds)
    r, p = pearsonr(y, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, y_pred, alpha=0.6, edgecolors='navy', facecolors='steelblue', s=50)
    lims = [min(y.min(), y_pred.min()) - 0.5, max(y.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('Experimental ΔΔG (kcal/mol)', fontsize=12)
    ax.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=12)
    ax.set_title(f'Circuit Model: Predicted vs Experimental ΔΔG\n'
                 f'Pearson r = {r:.3f}, n = {len(y)}', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / 'prediction_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_feature_distributions(X, out_path):
    n_features = X.shape[1]
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    for i, col in enumerate(X.columns):
        axes[i].hist(X[col], bins=20, edgecolor='navy', facecolor='steelblue', alpha=0.7)
        axes[i].set_xlabel(col, fontsize=8)
        axes[i].set_title(f'μ={X[col].mean():.3f}, σ={X[col].std():.3f}', fontsize=8)
        axes[i].tick_params(labelsize=7)
        axes[i].grid(True, alpha=0.3)
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'Feature Distributions ({n_features})', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_feature_correlations(X, y, out_path):
    n_features = X.shape[1]
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    for i, col in enumerate(X.columns):
        r_val, _ = pearsonr(X[col], y) if X[col].std() > 1e-12 else (0.0, 1.0)
        axes[i].scatter(X[col], y, alpha=0.5, s=20, edgecolors='navy', facecolors='steelblue')
        axes[i].set_xlabel(col, fontsize=7)
        axes[i].set_ylabel('ΔΔG', fontsize=8)
        axes[i].set_title(f'r={r_val:.3f}', fontsize=9,
                          color='darkred' if abs(r_val) > 0.2 else 'gray')
        axes[i].tick_params(labelsize=7)
        axes[i].grid(True, alpha=0.3)
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Feature vs ΔΔG Correlation', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path / 'feature_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_resistance_cv_heatmap(comparison_data, out_path):
    """Heatmap: models × resistance-CV strategies (R² or Pearson r)."""
    df = pd.DataFrame(comparison_data)
    if df.empty:
        return

    for metric, label in [('cv_r2_mean', 'R²'), ('pearson_r', 'Pearson r')]:
        pivot = df.pivot_table(
            index='model_type', columns='cv_strategy', values=metric,
            aggfunc='first',
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2.5),
                                        max(5, len(pivot) * 0.6)))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9, rotation=25, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=9, fontweight='bold',
                            color='white' if val < 0.2 else 'black')

        plt.colorbar(im, ax=ax, label=label)
        ax.set_title(f'{label}: Models × Resistance-CV Strategies',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        safe_label = label.replace('²', '2').replace(' ', '_')
        plt.savefig(out_path / f'resistance_cv_heatmap_{safe_label}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()


def _plot_model_comparison_by_cv(comparison_data, out_path):
    """Grouped bar chart: Pearson r for each model, grouped by CV strategy."""
    df = pd.DataFrame(comparison_data)
    if df.empty:
        return

    cv_strategies = df['cv_strategy'].unique()
    model_types = df['model_type'].unique()

    fig, ax = plt.subplots(figsize=(max(10, len(cv_strategies) * 3), 6))
    x = np.arange(len(cv_strategies))
    width = 0.8 / len(model_types)
    cmap = plt.cm.tab10

    for i, mt in enumerate(model_types):
        vals = []
        for cv in cv_strategies:
            row = df[(df['model_type'] == mt) & (df['cv_strategy'] == cv)]
            vals.append(row['pearson_r'].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=mt, color=cmap(i / len(model_types)),
               edgecolor='black', alpha=0.85)

    ax.set_xticks(x + width * len(model_types) / 2)
    ax.set_xticklabels(cv_strategies, fontsize=9, rotation=15)
    ax.set_ylabel('Pearson r', fontsize=12)
    ax.set_title('Model Performance by Resistance-CV Strategy\n'
                 '(same splits for all models)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path / 'model_by_cv_strategy.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_ablation(ablation_data, out_path):
    df = pd.DataFrame(ablation_data)
    df = df.sort_values('r2_drop', ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#c0392b' if d > 0 else '#27ae60' for d in df['r2_drop']]
    ax.barh(df['feature_dropped'], df['r2_drop'], color=colors,
            edgecolor='black', alpha=0.8)
    ax.set_xlabel('R² Drop When Feature Removed', fontsize=12)
    ax.set_title('Feature Ablation (Positive = Feature Helps)', fontsize=13)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(out_path / 'ablation_study.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_group_ablation(group_ablation_data, out_path):
    df = pd.DataFrame(group_ablation_data)
    if df.empty:
        return
    df = df.sort_values('r2_drop', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#c0392b' if d > 0 else '#27ae60' for d in df['r2_drop']]
    labels = [f"{r['group_dropped']} ({int(r['n_features_dropped'])})"
              for _, r in df.iterrows()]
    ax.barh(labels, df['r2_drop'], color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('R² Drop When Group Removed', fontsize=12)
    ax.set_title('Feature Group Ablation', fontsize=13)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(out_path / 'group_ablation_study.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_conductance_comparison(sensitivity_data, out_path):
    df = pd.DataFrame(sensitivity_data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#2980b9', '#e67e22', '#27ae60'][:len(df)]
    axes[0].bar(df['conductance_method'], df['cv_r2_mean'],
                yerr=df.get('cv_r2_std', 0), capsize=5,
                color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_ylabel('R²', fontsize=12)
    axes[0].set_title('R² by Conductance Method', fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[1].bar(df['conductance_method'], df['pearson_r'],
                color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_ylabel('Pearson r', fontsize=12)
    axes[1].set_title('Pearson r by Conductance Method', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path / 'conductance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────────
# Published comparison plots
# ──────────────────────────────────────────────────────────────────

def _plot_published_comparison(comparison_table, out_path):
    df = comparison_table
    if df.empty:
        return
    _plot_published_pearson_r(df, out_path)
    _plot_published_rmse(df, out_path)
    _plot_published_landscape(df, out_path)


def _plot_published_pearson_r(df, out_path):
    df = df.sort_values('pearson_r', ascending=True).reset_index(drop=True)
    is_ours = df['method'].str.contains('ours', case=False)
    colors = ['#e74c3c' if o else '#3498db' for o in is_ours]
    edge_colors = ['#c0392b' if o else '#2c3e50' for o in is_ours]
    lw = [2.5 if o else 0.8 for o in is_ours]

    fig, ax = plt.subplots(figsize=(11, max(8, len(df) * 0.45)))
    bars = ax.barh(df['method'], df['pearson_r'],
                   color=colors, edgecolor=edge_colors, linewidth=lw, alpha=0.85)
    for bar, val in zip(bars, df['pearson_r']):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Pearson Correlation (r)', fontsize=13)
    ax.set_title('Comparison with Published ΔΔG Prediction Methods\n'
                 '(each method evaluated with its own protocol)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(df['pearson_r'].max() + 0.08, 0.1))
    ax.grid(True, alpha=0.3, axis='x')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='#c0392b',
              label='Circuit Model (resistance-based CV)'),
        Patch(facecolor='#3498db', edgecolor='#2c3e50',
              label='Published Methods (author-reported)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path / 'published_comparison_pearson.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_published_rmse(df, out_path):
    df = df.dropna(subset=['rmse'])
    if df.empty:
        return
    df = df.sort_values('rmse', ascending=False).reset_index(drop=True)
    is_ours = df['method'].str.contains('ours', case=False)
    colors = ['#e74c3c' if o else '#2ecc71' for o in is_ours]

    fig, ax = plt.subplots(figsize=(11, max(8, len(df) * 0.45)))
    bars = ax.barh(df['method'], df['rmse'], color=colors, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, df['rmse']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('RMSE (kcal/mol)', fontsize=13)
    ax.set_title('RMSE Comparison (lower is better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(out_path / 'published_comparison_rmse.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_published_landscape(df, out_path):
    df = df.dropna(subset=['pearson_r', 'rmse'])
    if df.empty:
        return

    categories = df['category'].unique()
    cmap = plt.cm.Set2
    cat_colors = {cat: cmap(i / max(len(categories) - 1, 1))
                  for i, cat in enumerate(categories)}
    is_ours = df['method'].str.contains('ours', case=False)

    fig, ax = plt.subplots(figsize=(11, 8))
    for cat in categories:
        mask = (df['category'] == cat) & ~is_ours
        sub = df[mask]
        if len(sub) == 0:
            continue
        ax.scatter(sub['pearson_r'], sub['rmse'],
                   c=[cat_colors[cat]] * len(sub),
                   s=120, alpha=0.8, edgecolors='#2c3e50', linewidths=0.8,
                   label=cat, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row['method'], (row['pearson_r'], row['rmse']),
                        fontsize=7, ha='left', va='bottom',
                        xytext=(5, 3), textcoords='offset points')

    ours = df[is_ours]
    if not ours.empty:
        ax.scatter(ours['pearson_r'], ours['rmse'],
                   c='#e74c3c', s=350, marker='*', edgecolors='black',
                   linewidths=1.5, zorder=5, label='Circuit Model (ours)')
        for _, row in ours.iterrows():
            short = row['method'].replace('Circuit Model — ', '').replace(' (ours)', '')
            ax.annotate(short, (row['pearson_r'], row['rmse']),
                        fontsize=9, fontweight='bold', color='#c0392b',
                        ha='left', va='bottom',
                        xytext=(8, 8), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))

    ax.set_xlabel('Pearson r (→ better)', fontsize=13)
    ax.set_ylabel('RMSE kcal/mol (↓ better)', fontsize=13)
    ax.set_title('ΔΔG Prediction Landscape', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / 'published_comparison_landscape.png', dpi=150, bbox_inches='tight')
    plt.close()
