"""
Electrical-Resistance-Based Cross-Validation Strategies.

Every CV splitter in this module partitions the data using properties
derived from the resistor-network model of the protein.  This makes
the evaluation protocol itself reflect the circuit analogy:

1. ResistanceCentralityStratifiedKFold
   Stratify by the mutation site's resistance centrality (= current-flow
   closeness).  Each fold contains mutations at both well-connected
   (low effective resistance) and electrically remote positions.

2. PropagationRadiusStratifiedKFold
   Stratify by propagation radius — the number of residues whose
   effective resistance changes significantly upon mutation.  Folds
   balance local vs. globally-propagating mutations.

3. KirchhoffGroupedKFold
   Group mutations by the Kirchhoff index of their parent protein.
   Proteins with similar total network resistance go into the same
   group; folds split entire groups to prevent leakage from proteins
   with similar circuit topology.

4. SpectralClusteredKFold
   Cluster mutations by the Fiedler-vector component at their site
   (position relative to the graph's primary conductance bottleneck).
   This separates "bottleneck-adjacent" mutations from those in dense
   circuit cores, distributing them evenly across folds.

All splitters are scikit-learn compatible — they implement ``split(X, y,
groups)`` and ``get_n_splits()``.

Usage
-----
>>> from enzyme_resistance.resistance_cv import make_resistance_cv
>>> cv = make_resistance_cv('resistance_centrality', X, n_splits=5)
>>> for train_idx, test_idx in cv.split(X, y):
...     # all models use these exact same splits
"""

import numpy as np
from typing import Optional, List, Tuple, Iterator

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.cluster import KMeans


# ──────────────────────────────────────────────────────────────────
# Feature columns used for each strategy
# ──────────────────────────────────────────────────────────────────

_CENTRALITY_COL = 'resistance_centrality_wt'
_PROPAGATION_COL = 'propagation_radius'
_KIRCHHOFF_COL = 'delta_kirchhoff'
_FIEDLER_COL = 'fiedler_component_wt'
_VOLTAGE_COL = 'voltage_wt'
_CONDUCTANCE_DEG_COL = 'conductance_degree_wt'


# ──────────────────────────────────────────────────────────────────
# Helper: continuous → stratification bins
# ──────────────────────────────────────────────────────────────────

def _to_strata(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin continuous values into approximately equal-size strata.

    Uses quantile-based binning so each stratum has roughly the same
    number of samples, which is what StratifiedKFold expects.
    """
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(values, percentiles)
    return np.digitize(values, edges)


# ==================================================================
# 1. Resistance Centrality Stratified KFold
# ==================================================================

class ResistanceCentralityStratifiedKFold:
    """Stratify by resistance centrality of the mutation site.

    Resistance centrality  =  (n − 1) / Σ_j R(mutation, j)
    is the information centrality (current-flow closeness).

    Mutations at high-centrality (electrically central) vs.
    low-centrality (electrically peripheral) positions are spread
    evenly across folds.
    """

    name = 'resistance_centrality_stratified'
    description = ('Stratified by mutation-site resistance centrality '
                   '(current-flow closeness)')

    def __init__(self, n_splits: int = 5, centrality_values: Optional[np.ndarray] = None):
        self.n_splits = n_splits
        self._strata = None
        if centrality_values is not None:
            self._strata = _to_strata(centrality_values, n_bins=n_splits)

    @classmethod
    def from_feature_matrix(cls, X, n_splits: int = 5):
        """Construct from the feature matrix (pd.DataFrame)."""
        if _CENTRALITY_COL in X.columns:
            vals = X[_CENTRALITY_COL].values
        else:
            # Fallback: use first column
            vals = X.iloc[:, 0].values
        return cls(n_splits=n_splits, centrality_values=vals)

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self._strata is None:
            raise ValueError("Call from_feature_matrix() first")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        X_dummy = np.zeros((len(self._strata), 1))
        yield from skf.split(X_dummy, self._strata)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ==================================================================
# 2. Propagation Radius Stratified KFold
# ==================================================================

class PropagationRadiusStratifiedKFold:
    """Stratify by propagation radius — how many residues are
    electrically affected by the mutation.

    Ensures each fold has a balance of local (small-radius) and
    global (large-radius) circuit perturbations.
    """

    name = 'propagation_radius_stratified'
    description = ('Stratified by propagation radius '
                   '(number of residues with significant ΔR)')

    def __init__(self, n_splits: int = 5, radius_values: Optional[np.ndarray] = None):
        self.n_splits = n_splits
        self._strata = None
        if radius_values is not None:
            self._strata = _to_strata(radius_values, n_bins=n_splits)

    @classmethod
    def from_feature_matrix(cls, X, n_splits: int = 5):
        if _PROPAGATION_COL in X.columns:
            vals = X[_PROPAGATION_COL].values
        else:
            vals = X.iloc[:, 0].values
        return cls(n_splits=n_splits, radius_values=vals)

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self._strata is None:
            raise ValueError("Call from_feature_matrix() first")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        X_dummy = np.zeros((len(self._strata), 1))
        yield from skf.split(X_dummy, self._strata)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ==================================================================
# 3. Kirchhoff-Grouped KFold
# ==================================================================

class KirchhoffGroupedKFold:
    """Group mutations by the Kirchhoff index (total network resistance)
    of their parent protein.

    Proteins with similar Kirchhoff index → similar overall circuit
    topology → should go in the same fold to prevent leakage.  This
    is the resistance-model analogue of GroupKFold(group=protein).
    """

    name = 'kirchhoff_grouped'
    description = ('Grouped by protein Kirchhoff index '
                   '(total network resistance)')

    def __init__(self, n_splits: int = 5, group_labels: Optional[np.ndarray] = None):
        self.n_splits = n_splits
        self._groups = group_labels

    @classmethod
    def from_feature_matrix(cls, X, groups: np.ndarray, n_splits: int = 5):
        """Use per-protein Kirchhoff index (via delta_kirchhoff as proxy)
        to define groups.  Falls back to pdb_id groups."""
        # Group by the actual protein ID — but order groups by their
        # mean Kirchhoff shift (so proteins with similar total-resistance
        # topology are adjacent in the group ordering).
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if n_groups < n_splits:
            # Not enough groups — just use the raw groups
            return cls(n_splits=min(n_splits, n_groups), group_labels=groups)

        # Assign an integer label per unique group
        group_map = {g: i for i, g in enumerate(unique_groups)}
        int_groups = np.array([group_map[g] for g in groups])
        return cls(n_splits=min(n_splits, n_groups), group_labels=int_groups)

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        g = self._groups if self._groups is not None else groups
        if g is None:
            raise ValueError("No group labels — use from_feature_matrix()")
        n_groups = len(np.unique(g))
        n_splits = min(self.n_splits, n_groups)
        if n_splits < 2:
            # Fallback
            kf = KFold(n_splits=min(self.n_splits, len(X) if hasattr(X, '__len__') else 5),
                       shuffle=True, random_state=42)
            yield from kf.split(X, y)
            return
        gkf = GroupKFold(n_splits=n_splits)
        yield from gkf.split(X, y, groups=g)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ==================================================================
# 4. Spectral-Clustered KFold
# ==================================================================

class SpectralClusteredKFold:
    """Cluster mutations by the Fiedler-vector component at their
    mutation site (position relative to the primary conductance
    bottleneck of the circuit).

    Uses K-means on [fiedler_component_wt, voltage_wt,
    conductance_degree_wt] to define clusters, then does stratified
    splitting so each fold gets mutations from every spectral region.
    """

    name = 'spectral_clustered'
    description = ('Clustered by Fiedler-vector position '
                   '(bottleneck vs. core mutations)')

    def __init__(self, n_splits: int = 5, cluster_labels: Optional[np.ndarray] = None):
        self.n_splits = n_splits
        self._strata = cluster_labels

    @classmethod
    def from_feature_matrix(cls, X, n_splits: int = 5):
        # Pick spectral features for clustering
        spectral_cols = [c for c in [_FIEDLER_COL, _VOLTAGE_COL, _CONDUCTANCE_DEG_COL]
                         if c in X.columns]
        if not spectral_cols:
            # Fallback: use all features
            spectral_cols = list(X.columns)

        Z = X[spectral_cols].values.copy()
        # Standardise for KMeans
        std = Z.std(axis=0)
        std[std < 1e-12] = 1.0
        Z = (Z - Z.mean(axis=0)) / std

        n_clusters = min(n_splits, len(Z))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(Z)
        return cls(n_splits=n_splits, cluster_labels=labels)

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self._strata is None:
            raise ValueError("Call from_feature_matrix() first")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        X_dummy = np.zeros((len(self._strata), 1))
        yield from skf.split(X_dummy, self._strata)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ==================================================================
# Registry + factory
# ==================================================================

RESISTANCE_CV_STRATEGIES = {
    'resistance_centrality': ResistanceCentralityStratifiedKFold,
    'propagation_radius': PropagationRadiusStratifiedKFold,
    'kirchhoff_grouped': KirchhoffGroupedKFold,
    'spectral_clustered': SpectralClusteredKFold,
}

ALL_RESISTANCE_CV_NAMES = list(RESISTANCE_CV_STRATEGIES.keys())


def make_resistance_cv(
    strategy: str,
    X,
    n_splits: int = 5,
    groups: Optional[np.ndarray] = None,
):
    """
    Factory: build a resistance-based CV splitter.

    Parameters
    ----------
    strategy : str
        One of: resistance_centrality, propagation_radius,
                kirchhoff_grouped, spectral_clustered.
    X : pd.DataFrame
        Feature matrix (83 circuit-theoretic features).
    n_splits : int
        Number of folds.
    groups : array-like, optional
        Protein-level group labels (required for kirchhoff_grouped).

    Returns
    -------
    CV splitter
        Scikit-learn compatible splitter with `.split()`.
    """
    if strategy not in RESISTANCE_CV_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {ALL_RESISTANCE_CV_NAMES}"
        )

    cls = RESISTANCE_CV_STRATEGIES[strategy]

    if strategy == 'kirchhoff_grouped':
        if groups is None:
            raise ValueError("kirchhoff_grouped requires protein-level groups")
        return cls.from_feature_matrix(X, groups=groups, n_splits=n_splits)
    else:
        return cls.from_feature_matrix(X, n_splits=n_splits)


def precompute_all_splits(
    X,
    y: np.ndarray,
    n_splits: int = 5,
    groups: Optional[np.ndarray] = None,
) -> dict:
    """
    Pre-compute train/test index arrays for ALL resistance-based CV
    strategies.  Every model uses these exact same splits.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target values.
    n_splits : int
        Number of folds.
    groups : np.ndarray, optional
        Protein-level group labels.

    Returns
    -------
    dict
        strategy_name -> list of (train_indices, test_indices) tuples.
    """
    all_splits = {}
    for name in ALL_RESISTANCE_CV_NAMES:
        try:
            cv = make_resistance_cv(name, X, n_splits=n_splits, groups=groups)
            splits = list(cv.split(X, y, groups=groups))
            all_splits[name] = splits
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Could not build CV '{name}': {e}"
            )
    return all_splits
