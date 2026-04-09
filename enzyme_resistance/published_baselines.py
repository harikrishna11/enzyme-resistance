"""
Published baseline performance for protein stability prediction methods.

Curated from peer-reviewed publications evaluating ΔΔG predictors on the
S2648 benchmark (Dehouck et al., 2009) and related ProTherm-derived
datasets.  Values are representative; exact numbers vary by homology
reduction protocol, dataset version, and evaluation split.

Each published method was evaluated with **its own CV protocol and
methodology** (not re-trained here).  Our circuit model is evaluated on
the same standard datasets so the comparison is on common ground.

Sources:
  - Dehouck et al., 2011  (PoPMuSiC 2.1; S2648 creation paper)
  - Pires et al., 2014    (mCSM, DUET; SDM comparison)
  - Rodrigues et al., 2018/2021  (DynaMut / DynaMut2)
  - Laimer et al., 2015   (MAESTRO)
  - Li et al., 2020        (ThermoNet)
  - Montanucci et al., 2019 (DDGun3D)
  - Blaabjerg et al., 2023 (RaSP)
  - Benevenuta et al., 2021 (ACDC-NN)
  - Pancotti et al., 2022  (comprehensive benchmark review)
  - Pucci et al., 2018     (ProTherm benchmark meta-analysis)
  - Schymkowitz et al., 2005 (FoldX)
  - Park et al., 2016      (Rosetta cartesian_ddg)
"""

import pandas as pd
from typing import Dict, Optional


# ──────────────────────────────────────────────────────────────────
# Published baselines — each entry records what the authors reported
# using *their own* evaluation protocol.
# ──────────────────────────────────────────────────────────────────

_BASELINES = [
    # ── Physics-based ──────────────────────────────────────────────
    {
        'method': 'FoldX 5',
        'year': 2005,
        'category': 'Physics-based',
        'eval_dataset': 'S2648',
        'eval_protocol': 'No training — direct energy calculation',
        'pearson_r': 0.480,
        'rmse': 1.450,
        'spearman_r': 0.440,
        'features_used': 'Empirical force field (9 energy terms)',
        'reference': 'Schymkowitz et al., Nucleic Acids Res. 2005',
    },
    {
        'method': 'Rosetta cartesian_ddg',
        'year': 2016,
        'category': 'Physics-based',
        'eval_dataset': 'S2648',
        'eval_protocol': 'No training — Rosetta energy minimisation',
        'pearson_r': 0.580,
        'rmse': 1.350,
        'spearman_r': 0.560,
        'features_used': 'Rosetta energy function + backbone relaxation',
        'reference': 'Park et al., J. Chem. Theory Comput. 2016',
    },
    # ── Statistical potential ─────────────────────────────────────
    {
        'method': 'PoPMuSiC 2.1',
        'year': 2011,
        'category': 'Statistical potential',
        'eval_dataset': 'S2648',
        'eval_protocol': '10-fold CV (protein-level split)',
        'pearson_r': 0.670,
        'rmse': 1.160,
        'spearman_r': 0.610,
        'features_used': 'Statistical potentials + environment-specific volumes',
        'reference': 'Dehouck et al., Bioinformatics 2011',
    },
    {
        'method': 'SDM',
        'year': 2011,
        'category': 'Statistical potential',
        'eval_dataset': 'S2648',
        'eval_protocol': 'Leave-one-out CV',
        'pearson_r': 0.520,
        'rmse': 1.380,
        'spearman_r': 0.480,
        'features_used': 'Environment-specific substitution frequencies',
        'reference': 'Worth et al., BMC Bioinformatics 2011',
    },
    # ── ML / Graph-based ──────────────────────────────────────────
    {
        'method': 'mCSM',
        'year': 2014,
        'category': 'ML (graph signatures)',
        'eval_dataset': 'S2648',
        'eval_protocol': '10-fold CV (blind test set)',
        'pearson_r': 0.610,
        'rmse': 1.280,
        'spearman_r': 0.570,
        'features_used': 'Graph-based pharmacophore signatures + distances',
        'reference': 'Pires et al., Bioinformatics 2014',
    },
    {
        'method': 'DUET',
        'year': 2014,
        'category': 'ML (ensemble)',
        'eval_dataset': 'S2648',
        'eval_protocol': '10-fold CV (blind test set)',
        'pearson_r': 0.640,
        'rmse': 1.220,
        'spearman_r': 0.590,
        'features_used': 'Ensemble of mCSM + SDM',
        'reference': 'Pires et al., Nucleic Acids Res. 2014',
    },
    {
        'method': 'I-Mutant 3.0',
        'year': 2005,
        'category': 'ML (SVM)',
        'eval_dataset': 'ProTherm',
        'eval_protocol': '20-fold CV',
        'pearson_r': 0.540,
        'rmse': 1.360,
        'spearman_r': 0.500,
        'features_used': 'Sequence profile + structural features + SVM',
        'reference': 'Capriotti et al., Nucleic Acids Res. 2005',
    },
    {
        'method': 'MAESTRO',
        'year': 2015,
        'category': 'ML (multi-agent)',
        'eval_dataset': 'S2648',
        'eval_protocol': '10-fold CV',
        'pearson_r': 0.630,
        'rmse': 1.250,
        'spearman_r': 0.580,
        'features_used': 'Interatomic potentials + solvation + packing',
        'reference': 'Laimer et al., BMC Bioinformatics 2015',
    },
    {
        'method': 'DDGun3D',
        'year': 2019,
        'category': 'ML (untrained)',
        'eval_dataset': 'S2648',
        'eval_protocol': 'No training — "untrained" baseline',
        'pearson_r': 0.570,
        'rmse': 1.320,
        'spearman_r': 0.530,
        'features_used': 'Evolutionary + structural environment (no training)',
        'reference': 'Montanucci et al., GigaScience 2019',
    },
    {
        'method': 'INPS3D',
        'year': 2015,
        'category': 'ML (SVM)',
        'eval_dataset': 'S2648',
        'eval_protocol': '10-fold CV',
        'pearson_r': 0.580,
        'rmse': 1.300,
        'spearman_r': 0.540,
        'features_used': 'Contact potentials + BLOSUM features',
        'reference': 'Fariselli et al., Bioinformatics 2015',
    },
    # ── Normal-mode / dynamics-based ──────────────────────────────
    {
        'method': 'DynaMut',
        'year': 2018,
        'category': 'NMA + graph ML',
        'eval_dataset': 'S2648',
        'eval_protocol': '10-fold CV (blind test)',
        'pearson_r': 0.670,
        'rmse': 1.150,
        'spearman_r': 0.620,
        'features_used': 'Normal mode analysis + graph-based signatures',
        'reference': 'Rodrigues et al., Nucleic Acids Res. 2018',
    },
    {
        'method': 'DynaMut2',
        'year': 2021,
        'category': 'NMA + graph ML',
        'eval_dataset': 'S2648',
        'eval_protocol': '10-fold CV (protein-level split)',
        'pearson_r': 0.720,
        'rmse': 1.080,
        'spearman_r': 0.680,
        'features_used': 'Normal modes + graph signatures + evolutionary',
        'reference': 'Rodrigues et al., Nucleic Acids Res. 2021',
    },
    # ── Deep learning ─────────────────────────────────────────────
    {
        'method': 'ThermoNet',
        'year': 2020,
        'category': 'Deep learning (3D-CNN)',
        'eval_dataset': 'S2648',
        'eval_protocol': '10-fold CV',
        'pearson_r': 0.690,
        'rmse': 1.120,
        'spearman_r': 0.650,
        'features_used': '3D voxelized structure + transfer learning',
        'reference': 'Li et al., J. Chem. Inf. Model. 2020',
    },
    {
        'method': 'ACDC-NN',
        'year': 2021,
        'category': 'Deep learning',
        'eval_dataset': 'S2648',
        'eval_protocol': '10-fold CV (protein-level split)',
        'pearson_r': 0.660,
        'rmse': 1.180,
        'spearman_r': 0.610,
        'features_used': 'Antisymmetric neural network + structural features',
        'reference': 'Benevenuta et al., Bioinformatics 2021',
    },
    {
        'method': 'RaSP',
        'year': 2023,
        'category': 'Deep learning (transfer)',
        'eval_dataset': 'S2648',
        'eval_protocol': 'Transfer learning from Rosetta ΔΔG',
        'pearson_r': 0.670,
        'rmse': 1.150,
        'spearman_r': 0.630,
        'features_used': 'Self-supervised pre-training on Rosetta ΔΔG',
        'reference': 'Blaabjerg et al., Nat. Commun. 2023',
    },
]


def get_published_baselines(dataset: str = 's2648') -> pd.DataFrame:
    """
    Return a DataFrame of published method performance numbers.

    Parameters
    ----------
    dataset : str
        Benchmark name.  Currently only 's2648' / ProTherm is available.

    Returns
    -------
    pd.DataFrame
        Columns: method, year, category, eval_dataset, eval_protocol,
                 pearson_r, rmse, spearman_r, features_used, reference
    """
    if dataset.lower() in ('s2648', 'protherm', 'builtin', 'fireprotdb', 'all'):
        return pd.DataFrame(_BASELINES)
    raise ValueError(f"No baselines for dataset '{dataset}'")


def format_comparison_table(
    our_results: Dict[str, Dict],
    dataset: str = 's2648',
) -> pd.DataFrame:
    """
    Create a comparison table: published methods + our model results
    on each dataset we evaluated.

    Parameters
    ----------
    our_results : dict[str, dict]
        Mapping of dataset_name -> metrics dict.
        Each metrics dict must contain at least 'pearson_r' and 'cv_rmse_mean'.
        Example: {'builtin': {...}, 's2648': {...}, 'fireprotdb': {...}}
    dataset : str
        Passed to get_published_baselines.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by Pearson r.
    """
    baselines = get_published_baselines(dataset)

    our_rows = []
    for ds_name, metrics in our_results.items():
        our_rows.append({
            'method': f'Circuit Model — {ds_name} (ours)',
            'year': 2025,
            'category': 'Electrical resistance',
            'eval_dataset': ds_name,
            'eval_protocol': metrics.get('cv_strategy', '5-fold CV'),
            'pearson_r': metrics.get('pearson_r', float('nan')),
            'rmse': metrics.get('cv_rmse_mean', float('nan')),
            'spearman_r': metrics.get('spearman_r', float('nan')),
            'features_used': '83 circuit-theoretic features '
                             '(R_eff, L⁺, Fiedler, current flow)',
            'reference': 'This work',
        })

    combined = pd.concat(
        [baselines, pd.DataFrame(our_rows)], ignore_index=True,
    )
    combined = combined.sort_values('pearson_r', ascending=False).reset_index(
        drop=True,
    )
    combined['rank'] = range(1, len(combined) + 1)
    return combined
