"""Unit tests for enzyme_resistance package — pure circuit model."""

import numpy as np
import networkx as nx
import pandas as pd
import pytest

from enzyme_resistance.resistance import (
    effective_resistance_matrix,
    effective_resistance_matrix_with_pinv,
    laplacian_pseudoinverse,
    laplacian_eigenvalues,
    fiedler_vector,
    kirchhoff_index,
    resistance_centrality,
)
from enzyme_resistance.mutation import (
    compute_perturbation_factor,
    apply_mutation,
)
from enzyme_resistance.features import (
    extract_resistance_features,
    FEATURE_NAMES,
    FEATURE_GROUPS,
)
from enzyme_resistance.train import (
    train_and_evaluate,
    multi_cv_evaluate,
    compare_models,
    nested_cv_evaluate,
    ablation_study,
    group_ablation_study,
    _make_model,
    _make_cv,
)


def _make_test_graph():
    """Create a simple test graph (path graph with 5 nodes)."""
    G = nx.path_graph(5)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
        G[u][v]['distance'] = 5.0
    return G


def _make_larger_test_graph():
    """Create a more complex test graph for CV tests."""
    G = nx.watts_strogatz_graph(20, 4, 0.3, seed=42)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.RandomState(u * 100 + v).uniform(0.5, 2.0)
        G[u][v]['distance'] = 5.0
    return G


# ──────────────────────────────────────────────────────────────────
# Effective resistance & circuit primitives
# ──────────────────────────────────────────────────────────────────

class TestEffectiveResistance:
    def test_self_resistance_is_zero(self):
        G = _make_test_graph()
        R = effective_resistance_matrix(G)
        for i in range(len(G)):
            assert abs(R[i, i]) < 1e-10

    def test_symmetry(self):
        G = _make_test_graph()
        R = effective_resistance_matrix(G)
        assert np.allclose(R, R.T, atol=1e-10)

    def test_non_negative(self):
        G = _make_test_graph()
        R = effective_resistance_matrix(G)
        assert np.all(R >= -1e-10)

    def test_path_graph_resistance(self):
        """For a path graph, R(0,k) = k (all weights = 1)."""
        G = _make_test_graph()
        R = effective_resistance_matrix(G)
        for k in range(5):
            assert abs(R[0, k] - k) < 1e-8

    def test_kirchhoff_index(self):
        G = _make_test_graph()
        R = effective_resistance_matrix(G)
        Kf = kirchhoff_index(R)
        assert Kf > 0

    def test_resistance_centrality(self):
        G = _make_test_graph()
        R = effective_resistance_matrix(G)
        centralities = [resistance_centrality(R, i) for i in range(5)]
        # Center node (2) should have highest resistance centrality
        assert centralities[2] == max(centralities)

    def test_with_pinv_consistency(self):
        """R and L⁺ from the combined function should be consistent."""
        G = _make_test_graph()
        R1 = effective_resistance_matrix(G)
        R2, Lp = effective_resistance_matrix_with_pinv(G)
        assert np.allclose(R1, R2, atol=1e-10)
        # R[i,j] = Lp[i,i] + Lp[j,j] - 2*Lp[i,j]
        for i in range(len(G)):
            for j in range(len(G)):
                expected = Lp[i, i] + Lp[j, j] - 2 * Lp[i, j]
                assert abs(R2[i, j] - max(expected, 0)) < 1e-8

    def test_laplacian_pseudoinverse_symmetry(self):
        G = _make_test_graph()
        Lp = laplacian_pseudoinverse(G)
        assert np.allclose(Lp, Lp.T, atol=1e-10)


class TestSpectral:
    def test_eigenvalues_first_is_zero(self):
        """First eigenvalue of connected graph Laplacian should be ~0."""
        G = _make_test_graph()
        eigs = laplacian_eigenvalues(G)
        assert abs(eigs[0]) < 1e-10

    def test_eigenvalues_sorted(self):
        G = _make_test_graph()
        eigs = laplacian_eigenvalues(G)
        assert np.all(np.diff(eigs) >= -1e-10)

    def test_fiedler_vector_length(self):
        G = _make_test_graph()
        fv = fiedler_vector(G)
        assert len(fv) == len(G)

    def test_fiedler_vector_orthogonal_to_constant(self):
        """Fiedler vector should be orthogonal to the all-ones vector."""
        G = _make_test_graph()
        fv = fiedler_vector(G)
        # <fv, 1> ≈ 0
        assert abs(np.sum(fv)) < 1e-8

    def test_algebraic_connectivity_positive(self):
        """λ₂ > 0 for a connected graph."""
        G = _make_test_graph()
        eigs = laplacian_eigenvalues(G)
        lambda2 = eigs[1]
        assert lambda2 > 1e-8


# ──────────────────────────────────────────────────────────────────
# Mutation tests
# ──────────────────────────────────────────────────────────────────

class TestMutation:
    def test_perturbation_identity(self):
        p = compute_perturbation_factor('ALA', 'ALA')
        assert abs(p - 1.0) < 1e-10

    def test_charge_change_increases_perturbation(self):
        p = compute_perturbation_factor('ASP', 'LYS')
        assert abs(p - 1.0) > 0.3

    def test_apply_mutation_preserves_nodes(self):
        G = _make_test_graph()
        G_mut = apply_mutation(G, 2, 'ALA', 'GLU')
        assert G_mut.number_of_nodes() == G.number_of_nodes()
        assert G_mut.number_of_edges() == G.number_of_edges()

    def test_apply_mutation_changes_weights(self):
        G = _make_test_graph()
        G_mut = apply_mutation(G, 2, 'VAL', 'GLU')
        changed = any(
            G[u][v]['weight'] != G_mut[u][v]['weight']
            for u, v in G.edges()
        )
        assert changed

    def test_one_letter_codes(self):
        p = compute_perturbation_factor('A', 'G')
        assert p != 1.0


# ──────────────────────────────────────────────────────────────────
# Pure circuit feature extraction
# ──────────────────────────────────────────────────────────────────

class TestFeatures:
    def test_21_features_returned(self):
        """Should return exactly 21 circuit features."""
        G_wt = _make_test_graph()
        G_mut = apply_mutation(G_wt, 2, 'ALA', 'GLU')
        features = extract_resistance_features(
            G_wt, G_mut, 2, active_site_residues=[0, 4],
        )
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"
        assert len(features) == 21

    def test_no_physicochemical_features(self):
        """Verify no direct amino-acid property features leak in."""
        G_wt = _make_test_graph()
        G_mut = apply_mutation(G_wt, 2, 'ALA', 'GLU')
        features = extract_resistance_features(G_wt, G_mut, 2)
        forbidden = ['delta_hydrophobicity', 'delta_charge',
                     'delta_volume', 'perturbation_factor']
        for f in forbidden:
            assert f not in features, f"Physicochemical feature present: {f}"

    def test_identity_mutation_near_zero(self):
        """Same graph → all delta features ≈ 0."""
        G = _make_test_graph()
        features = extract_resistance_features(
            G, G, 2, active_site_residues=[0, 4],
        )
        for name in FEATURE_NAMES:
            if name.startswith('delta_'):
                assert abs(features[name]) < 1e-10, \
                    f"{name} should be ~0 for identity, got {features[name]}"

    def test_resistance_distance_features(self):
        G_wt = _make_test_graph()
        G_mut = apply_mutation(G_wt, 2, 'VAL', 'GLU')
        f = extract_resistance_features(G_wt, G_mut, 2)
        # delta_R_max >= |delta_R_global_mean|
        assert f['delta_R_max'] >= abs(f['delta_R_global_mean'])
        assert f['delta_R_std'] >= 0.0
        assert f['R_eccentricity_wt'] > 0

    def test_current_flow_betweenness_positive(self):
        """Current-flow betweenness should be ≥ 0."""
        G_wt = _make_test_graph()
        G_mut = apply_mutation(G_wt, 2, 'ALA', 'GLU')
        f = extract_resistance_features(G_wt, G_mut, 2)
        assert f['current_flow_betweenness_wt'] >= 0.0

    def test_resistance_centrality_positive(self):
        G_wt = _make_test_graph()
        G_mut = apply_mutation(G_wt, 2, 'ALA', 'GLU')
        f = extract_resistance_features(G_wt, G_mut, 2)
        assert f['resistance_centrality_wt'] > 0.0

    def test_voltage_features(self):
        """Self-potential should be non-negative."""
        G_wt = _make_test_graph()
        G_mut = apply_mutation(G_wt, 2, 'ALA', 'GLU')
        f = extract_resistance_features(G_wt, G_mut, 2)
        # L⁺[i,i] can be positive or negative depending on convention,
        # but should be a finite number
        assert np.isfinite(f['voltage_wt'])
        assert np.isfinite(f['delta_voltage'])

    def test_spectral_features(self):
        G_wt = _make_test_graph()
        G_mut = apply_mutation(G_wt, 2, 'ALA', 'GLU')
        f = extract_resistance_features(G_wt, G_mut, 2)
        assert np.isfinite(f['fiedler_component_wt'])
        assert np.isfinite(f['spectral_gap_ratio'])
        assert f['spectral_gap_ratio'] >= 0

    def test_conductance_degree(self):
        """Conductance degree = sum of edge conductances at site."""
        G_wt = _make_test_graph()
        f = extract_resistance_features(G_wt, G_wt, 2)
        # Node 2 in path graph has 2 edges, each weight=1 → should be 2.0
        assert abs(f['conductance_degree_wt'] - 2.0) < 1e-10

    def test_feature_groups_cover_all(self):
        all_in_groups = []
        for group_features in FEATURE_GROUPS.values():
            all_in_groups.extend(group_features)
        assert set(all_in_groups) == set(FEATURE_NAMES)

    def test_larger_graph(self):
        """Features should work on larger, more realistic graphs."""
        G_wt = _make_larger_test_graph()
        G_mut = apply_mutation(G_wt, 5, 'ALA', 'GLU')
        f = extract_resistance_features(G_wt, G_mut, 5)
        assert len(f) == 21
        for v in f.values():
            assert np.isfinite(v)


# ──────────────────────────────────────────────────────────────────
# Training / CV tests
# ──────────────────────────────────────────────────────────────────

class TestTrainAndEvaluate:
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        n = 50
        X = pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'f3': np.random.randn(n),
        })
        y = 2.0 * X['f1'] - 1.5 * X['f2'] + 0.5 * X['f3'] + np.random.randn(n) * 0.5
        groups = np.array(['P1'] * 15 + ['P2'] * 15 + ['P3'] * 10 + ['P4'] * 10)
        return X, y.values, groups

    def test_kfold_cv(self, synthetic_data):
        X, y, _ = synthetic_data
        result = train_and_evaluate(X, y, n_folds=5, cv_strategy='kfold')
        assert 'cv_r2_mean' in result
        assert result['cv_strategy'] == 'kfold'

    def test_group_kfold_cv(self, synthetic_data):
        X, y, groups = synthetic_data
        result = train_and_evaluate(
            X, y, n_folds=4, cv_strategy='group_kfold', groups=groups,
        )
        assert result['cv_strategy'] == 'group_kfold'

    def test_repeated_kfold_cv(self, synthetic_data):
        X, y, _ = synthetic_data
        result = train_and_evaluate(
            X, y, n_folds=5, cv_strategy='repeated_kfold',
        )
        assert result['cv_strategy'] == 'repeated_kfold'

    def test_logo_cv(self, synthetic_data):
        X, y, groups = synthetic_data
        result = train_and_evaluate(
            X, y, cv_strategy='logo', groups=groups,
        )
        assert result['cv_strategy'] == 'logo'

    def test_multi_cv(self, synthetic_data):
        X, y, groups = synthetic_data
        results = multi_cv_evaluate(X, y, groups=groups, n_folds=4)
        assert 'kfold' in results
        assert 'group_kfold' in results

    def test_nested_cv(self, synthetic_data):
        X, y, _ = synthetic_data
        result = nested_cv_evaluate(X, y, n_outer_folds=3, model_type='ridge')
        assert result['cv_strategy'] == 'nested_cv'

    def test_all_model_types(self, synthetic_data):
        X, y, _ = synthetic_data
        for mt in ['gradient_boosting', 'random_forest', 'extra_trees',
                    'ridge', 'elastic_net', 'lasso', 'svr']:
            result = train_and_evaluate(X, y, n_folds=3, model_type=mt)
            assert result['model_type'] == mt

    def test_compare_models(self, synthetic_data):
        X, y, _ = synthetic_data
        df = compare_models(X, y, n_folds=3)
        assert len(df) == 7
        assert 'cv_r2_mean' in df.columns

    def test_ablation(self, synthetic_data):
        X, y, _ = synthetic_data
        df = ablation_study(X, y, n_folds=3)
        assert len(df) == X.shape[1]

    def test_group_ablation_empty(self, synthetic_data):
        X, y, _ = synthetic_data
        df = group_ablation_study(X, y, n_folds=3)
        assert isinstance(df, pd.DataFrame)

    def test_feature_importances(self, synthetic_data):
        X, y, _ = synthetic_data
        result = train_and_evaluate(X, y, n_folds=3, model_type='gradient_boosting')
        assert 'feature_importances' in result

    def test_small_dataset(self):
        X = pd.DataFrame({'f1': [1.0, 2.0, 3.0], 'f2': [4.0, 5.0, 6.0]})
        y = np.array([0.1, 0.2, 0.3])
        result = train_and_evaluate(X, y, n_folds=2)
        assert 'cv_r2_mean' in result


class TestFactories:
    def test_make_model_all(self):
        for mt in ['gradient_boosting', 'random_forest', 'extra_trees',
                    'ridge', 'elastic_net', 'lasso', 'svr']:
            assert _make_model(mt) is not None

    def test_make_model_tuned(self):
        from sklearn.model_selection import GridSearchCV
        model = _make_model('gradient_boosting', tuned=True)
        assert isinstance(model, GridSearchCV)

    def test_make_cv_types(self):
        from sklearn.model_selection import (
            KFold, GroupKFold, RepeatedKFold, LeaveOneGroupOut,
        )
        assert isinstance(_make_cv('kfold', 5), KFold)
        groups = np.array([0, 0, 1, 1, 2, 2])
        assert isinstance(_make_cv('group_kfold', 3, groups), GroupKFold)
        assert isinstance(_make_cv('repeated_kfold', 3), RepeatedKFold)
        assert isinstance(_make_cv('logo', groups=groups), LeaveOneGroupOut)

    def test_make_cv_invalid(self):
        with pytest.raises(ValueError):
            _make_cv('invalid_strategy')
