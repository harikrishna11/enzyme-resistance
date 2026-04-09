"""
Step 1: Build the Protein Contact Graph.

Constructs a weighted graph from a PDB file where:
  - Nodes = amino acid residues (Cα atoms)
  - Edges = non-covalent contacts within a distance cutoff
  - Edge weight = conductance (distance-decayed)

Supports multiple conductance models:
  - 'exponential': exp(-d / decay_constant)   [default, physics-based]
  - 'inverse_square': 1 / d²
  - 'binary': 1 if d < cutoff, else 0
"""

import numpy as np
import networkx as nx
from Bio.PDB import PDBParser
from typing import List, Tuple, Optional


# Three-letter to one-letter amino acid code mapping
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

AA_1TO3 = {v: k for k, v in AA_3TO1.items()}


def _compute_conductance(distance: float, method: str = 'exponential',
                          decay_constant: float = 5.0) -> float:
    """Compute conductance from inter-residue distance."""
    if method == 'exponential':
        return np.exp(-distance / decay_constant)
    elif method == 'inverse_square':
        return 1.0 / (distance ** 2)
    elif method == 'binary':
        return 1.0
    else:
        raise ValueError(f"Unknown conductance method: {method}")


def build_contact_graph(
    pdb_file: str,
    cutoff: float = 8.0,
    conductance_method: str = 'exponential',
    decay_constant: float = 5.0,
    chain_id: Optional[str] = None,
) -> Tuple[nx.Graph, list]:
    """
    Build a protein contact graph from a PDB file.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file.
    cutoff : float
        Distance cutoff in Angstroms for defining contacts.
    conductance_method : str
        Method for computing edge conductance: 'exponential', 'inverse_square', or 'binary'.
    decay_constant : float
        Decay constant for exponential conductance (only used with 'exponential').
    chain_id : str, optional
        If specified, only use residues from this chain.

    Returns
    -------
    G : nx.Graph
        Weighted contact graph.
    residues : list
        List of Bio.PDB Residue objects in graph node order.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('enzyme', pdb_file)

    # Collect standard amino acid residues
    residues = []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.get_id() != chain_id:
                continue
            for residue in chain:
                # Skip hetero atoms and water
                if residue.get_id()[0] != ' ':
                    continue
                # Only include residues with a Cα atom
                if 'CA' in residue:
                    residues.append(residue)
        break  # Only first model

    G = nx.Graph()

    # Add nodes
    for i, res in enumerate(residues):
        G.add_node(
            i,
            resname=res.resname,
            resid=res.get_id()[1],
            chain=res.get_parent().get_id(),
            aa1=AA_3TO1.get(res.resname, 'X'),
        )

    # Add edges based on Cα-Cα distance
    ca_coords = []
    for res in residues:
        ca_coords.append(res['CA'].get_vector().get_array())
    ca_coords = np.array(ca_coords)

    n = len(residues)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if dist < cutoff:
                conductance = _compute_conductance(dist, conductance_method, decay_constant)
                G.add_edge(i, j, weight=conductance, distance=dist)

    return G, residues


def get_residue_index(residues: list, resid: int, chain: Optional[str] = None) -> int:
    """
    Find the graph node index for a given PDB residue number.

    Parameters
    ----------
    residues : list
        List of Bio.PDB Residue objects.
    resid : int
        PDB residue number.
    chain : str, optional
        Chain identifier.

    Returns
    -------
    int
        Node index in the contact graph.
    """
    for i, res in enumerate(residues):
        if res.get_id()[1] == resid:
            if chain is None or res.get_parent().get_id() == chain:
                return i
    raise ValueError(f"Residue {resid} (chain={chain}) not found in structure.")


def get_active_site_indices(residues: list, active_site_resids: List[int],
                             chain: Optional[str] = None) -> List[int]:
    """Convert a list of PDB residue numbers to graph node indices."""
    return [get_residue_index(residues, r, chain) for r in active_site_resids]
