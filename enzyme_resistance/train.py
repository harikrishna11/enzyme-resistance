"""
Step 5: Train Predictor.

Trains models on resistance-based features to predict experimental ΔΔG values.

Supports:
  - Multiple model types (GBR, RF, Ridge, ElasticNet, SVR, etc.)
  - Resistance-based CV strategies (all derived from circuit properties):
      * Resistance Centrality Stratified — stratify by current-flow closeness
      * Propagation Radius Stratified — stratify by electrical perturbation radius
      * Kirchhoff-Grouped — group by protein total network resistance
      * Spectral-Clustered — cluster by Fiedler-vector / bottleneck position
  - All models are evaluated on the exact same pre-computed splits
  - Nested CV, ablation, and conductance sensitivity studies
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    RepeatedKFold,
    LeaveOneGroupOut,
    cross_val_score,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from enzyme_resistance.contact_graph import build_contact_graph, get_residue_index
from enzyme_resistance.mutation import apply_mutation
from enzyme_resistance.features import extract_resistance_features, compute_wt_properties

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Feature building
# ──────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    pdb_paths: Dict[str, str],
    conductance_method: str = 'exponential',
    cutoff: float = 8.0,
    decay_constant: float = 5.0,
) -> Tuple[pd.DataFrame, np.ndarray, List[int]]:
    """
    Build feature matrix from mutation data and PDB structures.

    Memory-efficient: processes one protein at a time, freeing its graph
    and N×N resistance matrices before moving on to the next protein.

    Parameters
    ----------
    df : pd.DataFrame
        Mutation data with columns: pdb_id, chain, position, wild_type, mutation, ddG
    pdb_paths : dict
        Mapping of pdb_id -> local PDB file path.
    conductance_method : str
        Conductance weighting method.
    cutoff : float
        Contact distance cutoff in Angstroms.
    decay_constant : float
        Decay constant for exponential conductance.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target values (experimental ΔΔG).
    valid_indices : list
        Indices of successfully processed mutations.
    """
    import gc
    import networkx

    feature_rows = []
    valid_indices = []
    y_values = []
    failed_keys: set = set()

    # Group mutations by (pdb_id, chain) so we can process one protein
    # at a time, keeping only ONE graph + WT properties in memory.
    df_copy = df.copy()
    df_copy['_chain'] = df_copy.get('chain', pd.Series('A', index=df_copy.index))
    df_copy['_group_key'] = (
        df_copy['pdb_id'].astype(str) + '_' +
        df_copy['_chain'].astype(str) + '_' +
        conductance_method + '_' +
        str(cutoff)
    )

    grouped = df_copy.groupby('_group_key', sort=False)
    n_proteins = len(grouped)
    n_done = 0

    for cache_key, group in grouped:
        n_done += 1
        pdb_id = group['pdb_id'].iloc[0]
        chain = group['_chain'].iloc[0]

        if pdb_id not in pdb_paths:
            continue
        if cache_key in failed_keys:
            continue

        # ── Build graph & WT properties for this protein ──────────
        try:
            G_wt, residues = build_contact_graph(
                pdb_paths[pdb_id],
                cutoff=cutoff,
                conductance_method=conductance_method,
                decay_constant=decay_constant,
                chain_id=chain,
            )
            if G_wt.number_of_nodes() < 2:
                raise ValueError(
                    f"Graph for {pdb_id}:{chain} has only "
                    f"{G_wt.number_of_nodes()} nodes"
                )
            wt_props = compute_wt_properties(G_wt)
        except (ValueError, KeyError, IndexError,
                networkx.NetworkXError, np.linalg.LinAlgError) as e:
            logger.debug(f"Skipping protein {pdb_id}:{chain}: {e}")
            failed_keys.add(cache_key)
            continue

        # ── Process every mutation on this protein ────────────────
        for idx, row in group.iterrows():
            try:
                mut_site = get_residue_index(residues, row['position'], chain)
                G_mut = apply_mutation(
                    G_wt, mut_site, row['wild_type'], row['mutation'],
                )
                features = extract_resistance_features(
                    G_wt, G_mut, mut_site, wt_props=wt_props,
                )
                feature_rows.append(features)
                valid_indices.append(idx)
                y_values.append(float(row['ddG']))
            except (ValueError, KeyError, IndexError,
                    networkx.NetworkXError, np.linalg.LinAlgError) as e:
                logger.debug(
                    f"Skipping mutation {pdb_id}:{chain}:{row['position']} "
                    f"{row['wild_type']}->{row['mutation']}: {e}"
                )
                continue

        # ── Free this protein's data immediately ──────────────────
        del G_wt, residues, wt_props
        gc.collect()

        if n_done % 20 == 0:
            logger.info(
                f"  Processed {n_done}/{n_proteins} proteins "
                f"({len(feature_rows)} mutations so far)"
            )

    del df_copy
    gc.collect()

    if failed_keys:
        logger.info(f"Skipped {len(failed_keys)} protein(s) due to graph errors")

    X = pd.DataFrame(feature_rows)
    y = np.array(y_values)

    logger.info(f"Built features for {len(X)} / {len(df)} mutations")
    return X, y, valid_indices


# ──────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────

def _make_model(model_type: str, tuned: bool = False):
    """Create a model instance by name.

    Parameters
    ----------
    model_type : str
        One of: gradient_boosting, random_forest, extra_trees,
                ridge, elastic_net, lasso, svr, stacking.
    tuned : bool
        If True, return a GridSearchCV-wrapped model for automatic tuning.
    """
    if model_type == 'gradient_boosting':
        base = GradientBoostingRegressor(random_state=42)
        if tuned:
            param_grid = {
                'n_estimators': [200, 500],
                'max_depth': [4, 5],
                'learning_rate': [0.02, 0.05],
                'subsample': [0.8],
            }
            return GridSearchCV(base, param_grid, cv=3, scoring='r2',
                                n_jobs=-1, refit=True)
        base.set_params(n_estimators=500, max_depth=5, learning_rate=0.02,
                        subsample=0.8, min_samples_leaf=5)
        return base

    elif model_type == 'random_forest':
        base = RandomForestRegressor(random_state=42)
        if tuned:
            param_grid = {
                'n_estimators': [200, 500],
                'max_depth': [8, 12, None],
                'min_samples_leaf': [2, 5],
            }
            return GridSearchCV(base, param_grid, cv=3, scoring='r2',
                                n_jobs=-1, refit=True)
        base.set_params(n_estimators=300, max_depth=10, min_samples_leaf=3)
        return base

    elif model_type == 'extra_trees':
        base = ExtraTreesRegressor(random_state=42)
        base.set_params(n_estimators=300, max_depth=10, min_samples_leaf=3)
        return base

    elif model_type == 'ridge':
        if tuned:
            from sklearn.linear_model import RidgeCV
            return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        return Ridge(alpha=1.0)

    elif model_type == 'elastic_net':
        if tuned:
            param_grid = {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.2, 0.5, 0.8],
            }
            return GridSearchCV(ElasticNet(max_iter=5000, random_state=42),
                                param_grid, cv=3, scoring='r2', refit=True)
        return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)

    elif model_type == 'lasso':
        return Lasso(alpha=0.1, max_iter=5000, random_state=42)

    elif model_type == 'svr':
        if tuned:
            param_grid = {'C': [0.1, 1.0, 10.0], 'epsilon': [0.05, 0.1, 0.2]}
            return GridSearchCV(SVR(kernel='rbf'), param_grid, cv=3,
                                scoring='r2', refit=True)
        return SVR(kernel='rbf', C=1.0, epsilon=0.1)

    elif model_type == 'stacking':
        # Stacking ensemble: diverse base learners combined
        # by a Ridge meta-learner.  Typically +5–10% over individual models.
        # n_jobs=1 to avoid multiprocessing memory/resource issues.
        estimators = [
            ('gbr', GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42,
            )),
            ('rf', RandomForestRegressor(
                n_estimators=150, max_depth=8, min_samples_leaf=3,
                random_state=42,
            )),
            ('et', ExtraTreesRegressor(
                n_estimators=150, max_depth=8, min_samples_leaf=3,
                random_state=42,
            )),
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1)),
        ]
        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            passthrough=False,
            n_jobs=1,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ──────────────────────────────────────────────────────────────────
# CV strategy factory
# ──────────────────────────────────────────────────────────────────

def _make_cv(cv_strategy: str, n_folds: int = 5, groups: Optional[np.ndarray] = None):
    """Return a cross-validation splitter.

    Parameters
    ----------
    cv_strategy : str
        One of: kfold, group_kfold, repeated_kfold, logo (leave-one-group-out).
    n_folds : int
        Number of folds (ignored for logo).
    groups : array-like, optional
        Group labels (e.g. protein IDs). Required for group_kfold and logo.
    """
    if cv_strategy == 'kfold':
        return KFold(n_splits=n_folds, shuffle=True, random_state=42)

    elif cv_strategy == 'group_kfold':
        if groups is None:
            raise ValueError("group_kfold requires `groups`")
        n_groups = len(np.unique(groups))
        n_splits = min(n_folds, n_groups)
        if n_splits < 2:
            logger.warning(f"Only {n_groups} group(s); falling back to KFold")
            return KFold(n_splits=min(n_folds, len(groups)), shuffle=True,
                         random_state=42)
        return GroupKFold(n_splits=n_splits)

    elif cv_strategy == 'repeated_kfold':
        return RepeatedKFold(n_splits=n_folds, n_repeats=3, random_state=42)

    elif cv_strategy == 'logo':
        if groups is None:
            raise ValueError("logo requires `groups`")
        n_groups = len(np.unique(groups))
        if n_groups < 3:
            logger.warning(f"Only {n_groups} group(s); falling back to KFold")
            return KFold(n_splits=min(n_folds, len(groups)), shuffle=True,
                         random_state=42)
        return LeaveOneGroupOut()

    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}")


# ──────────────────────────────────────────────────────────────────
# Core train & evaluate
# ──────────────────────────────────────────────────────────────────

def train_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    model_type: str = 'gradient_boosting',
    cv_strategy: str = 'kfold',
    groups: Optional[np.ndarray] = None,
    tune_hyperparams: bool = False,
) -> Dict[str, float]:
    """
    Train a model and evaluate with cross-validation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target values.
    n_folds : int
        Number of cross-validation folds (ignored for logo).
    model_type : str
        Model type: 'gradient_boosting', 'random_forest', 'extra_trees',
        'ridge', 'elastic_net', 'lasso', 'svr'.
    cv_strategy : str
        CV strategy: 'kfold', 'group_kfold', 'repeated_kfold', 'logo'.
    groups : array-like, optional
        Group labels (protein IDs) for group-aware CV.
    tune_hyperparams : bool
        Whether to use GridSearchCV for the model.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    if len(X) < 4:
        return _empty_result(model_type, cv_strategy, len(X))

    effective_folds = n_folds
    if cv_strategy in ('group_kfold', 'logo') and groups is not None:
        n_groups = len(np.unique(groups))
        if cv_strategy == 'logo':
            effective_folds = n_groups
        else:
            effective_folds = min(n_folds, n_groups)

    if len(X) < effective_folds:
        logger.warning(f"Only {len(X)} samples for {effective_folds} splits; "
                       f"reducing folds")
        effective_folds = max(2, len(X))
        if cv_strategy == 'logo':
            cv_strategy = 'kfold'

    # Build pipeline: scale -> model
    model = _make_model(model_type, tuned=tune_hyperparams)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])

    cv = _make_cv(cv_strategy, n_folds=effective_folds, groups=groups)

    X_arr = np.asarray(X, dtype=np.float64)
    # Replace any NaN/Inf in features with 0
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        r2_scores = cross_val_score(
            pipe, X_arr, y, cv=cv, groups=groups, scoring='r2',
        )
        neg_mse_scores = cross_val_score(
            pipe, X_arr, y, cv=cv, groups=groups,
            scoring='neg_mean_squared_error',
        )
        neg_mae_scores = cross_val_score(
            pipe, X_arr, y, cv=cv, groups=groups,
            scoring='neg_mean_absolute_error',
        )

        # Cross-validated predictions for correlation
        # RepeatedKFold has overlapping test sets → cross_val_predict fails.
        # Fall back to a plain KFold for predictions in that case.
        try:
            y_pred = cross_val_predict(pipe, X_arr, y, cv=cv, groups=groups)
        except ValueError:
            pred_cv = KFold(n_splits=min(effective_folds, len(y)),
                            shuffle=True, random_state=42)
            y_pred = cross_val_predict(pipe, X_arr, y, cv=pred_cv)

    # Correlation metrics
    if np.std(y_pred) < 1e-12 or np.std(y) < 1e-12:
        pearson_r_val, pearson_p = 0.0, 1.0
        spearman_r_val, spearman_p = 0.0, 1.0
    else:
        pearson_r_val, pearson_p = pearsonr(y, y_pred)
        spearman_r_val, spearman_p = spearmanr(y, y_pred)

    results = {
        'cv_r2_mean': float(np.mean(r2_scores)),
        'cv_r2_std': float(np.std(r2_scores)),
        'cv_rmse_mean': float(np.sqrt(np.clip(-np.mean(neg_mse_scores), 0, None))),
        'cv_mae_mean': float(-np.mean(neg_mae_scores)),
        'pearson_r': float(pearson_r_val),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r_val),
        'spearman_p': float(spearman_p),
        'n_samples': len(y),
        'n_features': X.shape[1],
        'model_type': model_type,
        'cv_strategy': cv_strategy,
    }

    # Feature importances (if tree-based model, re-fit on full data)
    try:
        pipe.fit(X_arr, y)
        fitted_model = pipe.named_steps['model']
        # Unwrap GridSearchCV if needed
        if hasattr(fitted_model, 'best_estimator_'):
            fitted_model = fitted_model.best_estimator_
        if hasattr(fitted_model, 'feature_importances_'):
            importances = fitted_model.feature_importances_
            if isinstance(X, pd.DataFrame):
                results['feature_importances'] = dict(zip(X.columns, importances))
        elif hasattr(fitted_model, 'coef_'):
            coefs = fitted_model.coef_
            if isinstance(X, pd.DataFrame):
                results['feature_coefficients'] = dict(zip(X.columns, coefs))
    except Exception:
        pass

    return results


def _empty_result(model_type, cv_strategy, n_samples):
    """Return a placeholder result when there's not enough data."""
    return {
        'cv_r2_mean': float('nan'),
        'cv_r2_std': float('nan'),
        'cv_rmse_mean': float('nan'),
        'cv_mae_mean': float('nan'),
        'pearson_r': float('nan'),
        'pearson_p': float('nan'),
        'spearman_r': float('nan'),
        'spearman_p': float('nan'),
        'n_samples': n_samples,
        'n_features': 0,
        'model_type': model_type,
        'cv_strategy': cv_strategy,
    }


# ──────────────────────────────────────────────────────────────────
# Multi-CV evaluation
# ──────────────────────────────────────────────────────────────────

def multi_cv_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_folds: int = 5,
    model_type: str = 'gradient_boosting',
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate one model under multiple CV strategies.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target values.
    groups : array-like, optional
        Group labels (protein IDs).
    n_folds : int
        Number of folds for fold-based CVs.
    model_type : str
        Model type.

    Returns
    -------
    dict[str, dict]
        CV strategy name -> evaluation metrics.
    """
    strategies = ['kfold', 'repeated_kfold']
    if groups is not None and len(np.unique(groups)) >= 3:
        strategies.extend(['group_kfold', 'logo'])

    results = {}
    for strat in strategies:
        try:
            results[strat] = train_and_evaluate(
                X, y,
                n_folds=n_folds,
                model_type=model_type,
                cv_strategy=strat,
                groups=groups,
            )
        except Exception as e:
            logger.warning(f"CV strategy '{strat}' failed: {e}")
            results[strat] = _empty_result(model_type, strat, len(y))

    return results


# ──────────────────────────────────────────────────────────────────
# Model comparison
# ──────────────────────────────────────────────────────────────────

ALL_MODEL_TYPES = [
    'gradient_boosting', 'random_forest', 'extra_trees',
    'ridge', 'elastic_net', 'lasso', 'svr', 'stacking',
]


def compare_models(
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    cv_strategy: str = 'kfold',
    groups: Optional[np.ndarray] = None,
    model_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare multiple model types under the same CV.

    Returns a DataFrame with one row per model.
    """
    if model_types is None:
        model_types = ALL_MODEL_TYPES

    rows = []
    for mt in model_types:
        try:
            metrics = train_and_evaluate(
                X, y, n_folds=n_folds,
                model_type=mt, cv_strategy=cv_strategy, groups=groups,
            )
            rows.append(metrics)
        except Exception as e:
            logger.warning(f"Model {mt} failed: {e}")
            rows.append(_empty_result(mt, cv_strategy, len(y)))

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Resistance-CV model comparison (same splits for all models)
# ──────────────────────────────────────────────────────────────────

def compare_models_resistance_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    groups: Optional[np.ndarray] = None,
    model_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare ALL model types across ALL resistance-based CV strategies,
    using the exact same pre-computed splits for every model.

    This is the core evaluation: every model sees identical train/test
    partitions so performance differences are purely due to the model,
    not the split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (83 circuit-theoretic features).
    y : np.ndarray
        Experimental ΔΔG values.
    n_splits : int
        Number of folds.
    groups : np.ndarray, optional
        Protein group labels (for kirchhoff_grouped).
    model_types : list[str], optional
        Model names to evaluate.  Default: ALL_MODEL_TYPES.

    Returns
    -------
    pd.DataFrame
        One row per (model, cv_strategy) combination.
    """
    from enzyme_resistance.resistance_cv import precompute_all_splits

    if model_types is None:
        model_types = ALL_MODEL_TYPES

    # Step 1: pre-compute splits ONCE
    all_splits = precompute_all_splits(X, y, n_splits=n_splits, groups=groups)

    if not all_splits:
        logger.warning("No resistance-based CV splits could be computed.")
        return pd.DataFrame()

    X_arr = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0,
                          posinf=0.0, neginf=0.0)

    import gc

    rows = []
    for cv_name, splits in all_splits.items():
        for mt in model_types:
            try:
                result = _evaluate_on_splits(
                    X_arr, y, X.columns.tolist(), splits, mt, cv_name,
                )
                rows.append(result)
            except Exception as e:
                logger.warning(f"{mt} on {cv_name} failed: {e}")
                result = _empty_result(mt, cv_name, len(y))
                rows.append(result)
        # Free memory between CV strategies
        gc.collect()

    return pd.DataFrame(rows)


def _evaluate_on_splits(
    X_arr: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    splits: List[Tuple],
    model_type: str,
    cv_name: str,
) -> Dict[str, float]:
    """Evaluate one model on pre-computed splits."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    model_factory = lambda: Pipeline([
        ('scaler', StandardScaler()),
        ('model', _make_model(model_type)),
    ])

    r2_scores = []
    mse_scores = []
    mae_scores = []
    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in splits:
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = model_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    if np.std(all_y_pred) < 1e-12 or np.std(all_y_true) < 1e-12:
        pr, pp = 0.0, 1.0
        sr, sp = 0.0, 1.0
    else:
        pr, pp = pearsonr(all_y_true, all_y_pred)
        sr, sp = spearmanr(all_y_true, all_y_pred)

    # Feature importances (fit on full data)
    fi = {}
    try:
        full_pipe = model_factory()
        full_pipe.fit(X_arr, y)
        fitted = full_pipe.named_steps['model']
        if hasattr(fitted, 'best_estimator_'):
            fitted = fitted.best_estimator_
        if hasattr(fitted, 'feature_importances_'):
            fi = dict(zip(feature_names, fitted.feature_importances_))
        elif hasattr(fitted, 'coef_'):
            fi = dict(zip(feature_names, fitted.coef_))
    except Exception:
        pass

    result = {
        'model_type': model_type,
        'cv_strategy': cv_name,
        'cv_r2_mean': float(np.mean(r2_scores)),
        'cv_r2_std': float(np.std(r2_scores)),
        'cv_rmse_mean': float(np.sqrt(np.clip(np.mean(mse_scores), 0, None))),
        'cv_mae_mean': float(np.mean(mae_scores)),
        'pearson_r': float(pr),
        'pearson_p': float(pp),
        'spearman_r': float(sr),
        'spearman_p': float(sp),
        'n_samples': len(y),
        'n_features': X_arr.shape[1],
        'n_folds': len(splits),
    }
    if fi:
        result['feature_importances'] = fi

    return result


# ──────────────────────────────────────────────────────────────────
# Nested CV (hyperparameter tuning + honest evaluation)
# ──────────────────────────────────────────────────────────────────

def nested_cv_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    n_outer_folds: int = 5,
    model_type: str = 'gradient_boosting',
    groups: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Nested cross-validation: outer loop for evaluation, inner GridSearchCV
    for hyperparameter tuning.

    This gives an unbiased estimate of generalisation performance
    when tuning is involved.
    """
    if len(X) < n_outer_folds:
        return _empty_result(model_type, 'nested_cv', len(y))

    model = _make_model(model_type, tuned=True)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])

    X_arr = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0,
                          posinf=0.0, neginf=0.0)

    outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2_scores = cross_val_score(pipe, X_arr, y, cv=outer_cv, scoring='r2')
        neg_mse = cross_val_score(pipe, X_arr, y, cv=outer_cv,
                                  scoring='neg_mean_squared_error')
        y_pred = cross_val_predict(pipe, X_arr, y, cv=outer_cv)

    if np.std(y_pred) < 1e-12 or np.std(y) < 1e-12:
        pr, pp = 0.0, 1.0
        sr, sp = 0.0, 1.0
    else:
        pr, pp = pearsonr(y, y_pred)
        sr, sp = spearmanr(y, y_pred)

    return {
        'cv_r2_mean': float(np.mean(r2_scores)),
        'cv_r2_std': float(np.std(r2_scores)),
        'cv_rmse_mean': float(np.sqrt(np.clip(-np.mean(neg_mse), 0, None))),
        'pearson_r': float(pr),
        'spearman_r': float(sr),
        'n_samples': len(y),
        'n_features': X.shape[1],
        'model_type': model_type,
        'cv_strategy': 'nested_cv',
    }


# ──────────────────────────────────────────────────────────────────
# Ablation study (drop-one-feature and drop-one-group)
# ──────────────────────────────────────────────────────────────────

def ablation_study(
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    cv_strategy: str = 'kfold',
    groups: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Drop-one-feature ablation study to identify which resistance feature
    matters most for prediction.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target values.
    n_folds : int
        Number of CV folds.
    cv_strategy : str
        CV strategy for evaluation.
    groups : array-like, optional
        Group labels.

    Returns
    -------
    pd.DataFrame
        Ablation results showing R² drop when each feature is removed.
    """
    # Full model baseline
    baseline = train_and_evaluate(
        X, y, n_folds=n_folds, cv_strategy=cv_strategy, groups=groups,
    )
    baseline_r2 = baseline['cv_r2_mean']

    results = []
    for feature in X.columns:
        X_ablated = X.drop(columns=[feature])
        metrics = train_and_evaluate(
            X_ablated, y, n_folds=n_folds,
            cv_strategy=cv_strategy, groups=groups,
        )
        r2_drop = baseline_r2 - metrics['cv_r2_mean']
        results.append({
            'feature_dropped': feature,
            'r2_without': metrics['cv_r2_mean'],
            'r2_drop': r2_drop,
            'importance_rank': 0,
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('r2_drop', ascending=False)
    df_results['importance_rank'] = range(1, len(df_results) + 1)
    return df_results


def group_ablation_study(
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    cv_strategy: str = 'kfold',
    groups: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Drop-one-feature-group ablation to see which *category* of features
    matters most (resistance vs centrality vs topology vs physicochemical).
    """
    from enzyme_resistance.features import FEATURE_GROUPS

    baseline = train_and_evaluate(
        X, y, n_folds=n_folds, cv_strategy=cv_strategy, groups=groups,
    )
    baseline_r2 = baseline['cv_r2_mean']

    results = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        cols_to_drop = [c for c in group_cols if c in X.columns]
        if not cols_to_drop:
            continue
        X_ablated = X.drop(columns=cols_to_drop)
        if X_ablated.shape[1] == 0:
            continue
        metrics = train_and_evaluate(
            X_ablated, y, n_folds=n_folds,
            cv_strategy=cv_strategy, groups=groups,
        )
        results.append({
            'group_dropped': group_name,
            'n_features_dropped': len(cols_to_drop),
            'r2_without': metrics['cv_r2_mean'],
            'r2_drop': baseline_r2 - metrics['cv_r2_mean'],
        })

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values('r2_drop', ascending=False)
    return df_results


# ──────────────────────────────────────────────────────────────────
# Conductance sensitivity study
# ──────────────────────────────────────────────────────────────────

def conductance_sensitivity_study(
    df: pd.DataFrame,
    pdb_paths: Dict[str, str],
    n_folds: int = 5,
    cv_strategy: str = 'kfold',
    groups: Optional[np.ndarray] = None,
    max_mutations: int = 300,
) -> pd.DataFrame:
    """
    Test different conductance functions to see how much the physics
    of the edge model matters.

    Tests: exponential, inverse_square, and binary (contact-only) weighting.

    For large datasets, a random subsample is used to keep memory manageable
    (the sensitivity *ranking* of methods is stable even on subsets).

    Parameters
    ----------
    max_mutations : int
        Maximum number of mutations to use.  If the dataset is larger,
        a stratified-by-protein subsample is taken.
    """
    import gc

    # Subsample for large datasets to avoid OOM
    if len(df) > max_mutations:
        logger.info(f"Subsampling {max_mutations} mutations (from {len(df)}) "
                     f"for conductance sensitivity study")
        # Stratified sample by protein for diversity
        df_sample = (
            df.groupby('pdb_id', group_keys=False)
            .apply(lambda x: x.sample(
                n=min(len(x), max(1, max_mutations // df['pdb_id'].nunique())),
                random_state=42,
            ))
        ).head(max_mutations)
    else:
        df_sample = df

    methods = ['exponential', 'inverse_square', 'binary']
    results = []

    for method in methods:
        logger.info(f"\n--- Conductance method: {method} ---")
        X, y, valid_idx = build_feature_matrix(df_sample, pdb_paths,
                                               conductance_method=method)

        if len(X) < n_folds:
            logger.warning(f"Not enough samples for {method}")
            continue

        # Build groups for valid indices if available
        g = None
        if groups is not None and len(groups) >= len(df_sample):
            g = groups[valid_idx] if max(valid_idx) < len(groups) else None

        metrics = train_and_evaluate(
            X, y, n_folds=n_folds, cv_strategy=cv_strategy, groups=g,
        )
        metrics['conductance_method'] = method
        results.append(metrics)

        # Free memory between iterations
        del X, y, valid_idx
        gc.collect()

    return pd.DataFrame(results)
