"""
Step 3: Model the Mutation.

A mutation at residue k changes the physicochemical properties of that node,
which alters the weights (conductances) of all edges incident to k.

Perturbation model:
    perturbation_factor = 1.0 + alpha * delta_hydrophobicity + beta * |delta_charge| + gamma * delta_volume
    new_conductance = old_conductance * perturbation_factor

This captures the physical reality that mutations change local packing,
electrostatics, and hydrophobic interactions.
"""

import networkx as nx
from typing import Optional


# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'ALA': 1.8,  'VAL': 4.2,  'LEU': 3.8,  'ILE': 4.5,
    'PHE': 2.8,  'TRP': -0.9, 'MET': 1.9,  'PRO': -1.6,
    'GLY': -0.4, 'SER': -0.8, 'THR': -0.7, 'CYS': 2.5,
    'TYR': -1.3, 'HIS': -3.2, 'ASP': -3.5, 'GLU': -3.5,
    'ASN': -3.5, 'GLN': -3.5, 'LYS': -3.9, 'ARG': -4.5,
}

# Formal charge at pH 7
CHARGE = {
    'ASP': -1.0, 'GLU': -1.0, 'LYS': 1.0, 'ARG': 1.0, 'HIS': 0.5,
    'ALA': 0.0, 'VAL': 0.0, 'LEU': 0.0, 'ILE': 0.0, 'PHE': 0.0,
    'TRP': 0.0, 'MET': 0.0, 'PRO': 0.0, 'GLY': 0.0, 'SER': 0.0,
    'THR': 0.0, 'CYS': 0.0, 'TYR': 0.0, 'ASN': 0.0, 'GLN': 0.0,
}

# Side-chain volume (Å³) — Zamyatnin scale
VOLUME = {
    'GLY': 60.1,  'ALA': 88.6,  'VAL': 140.0, 'LEU': 166.7,
    'ILE': 166.7, 'PRO': 112.7, 'PHE': 189.9, 'TRP': 227.8,
    'MET': 162.9, 'SER': 89.0,  'THR': 116.1, 'CYS': 108.5,
    'TYR': 193.6, 'HIS': 153.2, 'ASP': 111.1, 'GLU': 138.4,
    'ASN': 114.1, 'GLN': 143.8, 'LYS': 168.6, 'ARG': 173.4,
}

# One-letter to three-letter
AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


def _normalize_aa(aa: str) -> str:
    """Convert single-letter AA code to three-letter if needed."""
    if len(aa) == 1:
        return AA_1TO3.get(aa.upper(), aa.upper())
    return aa.upper()


def compute_perturbation_factor(
    aa_from: str,
    aa_to: str,
    alpha: float = 0.1,
    beta: float = 0.3,
    gamma: float = 0.002,
) -> float:
    """
    Compute the perturbation factor for a mutation.

    Parameters
    ----------
    aa_from : str
        Wild-type amino acid (1-letter or 3-letter code).
    aa_to : str
        Mutant amino acid (1-letter or 3-letter code).
    alpha : float
        Weight for hydrophobicity change.
    beta : float
        Weight for charge change.
    gamma : float
        Weight for volume change.

    Returns
    -------
    float
        Perturbation factor (> 1 means increased conductance, < 1 means decreased).
    """
    aa_from = _normalize_aa(aa_from)
    aa_to = _normalize_aa(aa_to)

    delta_hydro = HYDROPHOBICITY.get(aa_to, 0.0) - HYDROPHOBICITY.get(aa_from, 0.0)
    delta_charge = abs(CHARGE.get(aa_to, 0.0) - CHARGE.get(aa_from, 0.0))
    delta_volume = abs(VOLUME.get(aa_to, 100.0) - VOLUME.get(aa_from, 100.0))

    perturbation = 1.0 + alpha * delta_hydro + beta * delta_charge + gamma * delta_volume
    return perturbation


def apply_mutation(
    G: nx.Graph,
    mutation_site: int,
    aa_from: str,
    aa_to: str,
    alpha: float = 0.1,
    beta: float = 0.3,
    gamma: float = 0.002,
) -> nx.Graph:
    """
    Apply a mutation to the contact graph by perturbing edge weights at the mutation site.

    Parameters
    ----------
    G : nx.Graph
        Wild-type contact graph.
    mutation_site : int
        Node index of the mutated residue.
    aa_from : str
        Wild-type amino acid (1-letter or 3-letter code).
    aa_to : str
        Mutant amino acid (1-letter or 3-letter code).
    alpha, beta, gamma : float
        Perturbation weights for hydrophobicity, charge, and volume changes.

    Returns
    -------
    nx.Graph
        Mutant contact graph with perturbed edge weights.
    """
    G_mut = G.copy()

    perturbation = compute_perturbation_factor(aa_from, aa_to, alpha, beta, gamma)

    for neighbor in G_mut.neighbors(mutation_site):
        old_weight = G_mut[mutation_site][neighbor]['weight']
        G_mut[mutation_site][neighbor]['weight'] = old_weight * perturbation

    return G_mut
