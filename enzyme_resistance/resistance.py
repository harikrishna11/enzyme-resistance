"""
Step 2: Compute Effective Resistance (The Key Physics).

Effective resistance between nodes i and j in a weighted graph:
    R_eff(i,j) = (e_i - e_j)^T * L^+ * (e_i - e_j)
               = L^+_ii + L^+_jj - 2 * L^+_ij

Where L^+ is the Moore-Penrose pseudoinverse of the weighted Laplacian.

Physical meaning: measures how "isolated" two residues are in terms of
information/force flow through the protein contact network.

Additional circuit primitives exposed here:
    - L^+ (Laplacian pseudoinverse) — "Green's function" of the network;
      its diagonal entries are self-potentials and off-diagonals are
      transfer voltages.
    - Laplacian eigenvalues — the spectrum of the circuit; the smallest
      non-zero eigenvalue (Fiedler value / algebraic connectivity) is the
      bottleneck conductance.
"""

import numpy as np
import networkx as nx
from typing import Tuple


# ──────────────────────────────────────────────────────────────────
# Core matrices
# ──────────────────────────────────────────────────────────────────

def laplacian_pseudoinverse(G: nx.Graph) -> np.ndarray:
    """
    Compute the Moore-Penrose pseudoinverse of the graph Laplacian.

    L^+ has direct circuit interpretation:
      - L^+[i,i] = "self-potential" at node i (voltage when unit charge
        is uniformly distributed and removed at i)
      - L^+[i,j] = "transfer voltage" (voltage at i due to charge at j)
      - R_eff(i,j) = L^+[i,i] + L^+[j,j] - 2*L^+[i,j]

    Parameters
    ----------
    G : nx.Graph
        Weighted contact graph (edge 'weight' = conductance).

    Returns
    -------
    L_pinv : np.ndarray
        n × n pseudoinverse of the weighted Laplacian.
    """
    L = nx.laplacian_matrix(G, weight='weight').toarray().astype(np.float64)
    return np.linalg.pinv(L)


def effective_resistance_matrix(G: nx.Graph) -> np.ndarray:
    """
    Compute the full effective resistance matrix for a weighted graph.

    Parameters
    ----------
    G : nx.Graph
        Weighted contact graph (edge attribute 'weight' = conductance).

    Returns
    -------
    R : np.ndarray
        n x n matrix where R[i,j] = effective resistance between nodes i and j.
    """
    L_pinv = laplacian_pseudoinverse(G)

    # Vectorized: R[i,j] = L_pinv[i,i] + L_pinv[j,j] - 2*L_pinv[i,j]
    diag = np.diag(L_pinv)
    R = diag[:, np.newaxis] + diag[np.newaxis, :] - 2.0 * L_pinv

    # Ensure non-negative (numerical precision)
    R = np.maximum(R, 0.0)
    return R


def effective_resistance_matrix_with_pinv(
    G: nx.Graph,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both R and L^+ in one pass (avoids redundant pseudoinverse).

    Returns
    -------
    R : np.ndarray   — effective resistance matrix
    L_pinv : np.ndarray — Laplacian pseudoinverse
    """
    L_pinv = laplacian_pseudoinverse(G)
    diag = np.diag(L_pinv)
    R = diag[:, np.newaxis] + diag[np.newaxis, :] - 2.0 * L_pinv
    R = np.maximum(R, 0.0)
    return R, L_pinv


# ──────────────────────────────────────────────────────────────────
# Spectral helpers
# ──────────────────────────────────────────────────────────────────

def laplacian_eigenvalues(G: nx.Graph) -> np.ndarray:
    """
    Sorted eigenvalues of the weighted Laplacian (ascending).

    The spectrum encodes the circuit's resonance structure:
      λ₁ = 0           (ground mode — connected component)
      λ₂ = algebraic connectivity = bottleneck conductance
      λ₃, λ₄, ...     higher modes
    """
    L = nx.laplacian_matrix(G, weight='weight').toarray().astype(np.float64)
    eigvals = np.linalg.eigvalsh(L)
    return np.sort(eigvals)


def fiedler_vector(G: nx.Graph) -> np.ndarray:
    """
    The Fiedler vector — eigenvector for λ₂ (smallest non-zero eigenvalue).

    Interpretation: the dominant mode of "oscillation" in the circuit.
    Nodes with opposite signs are on different sides of the electrical
    bottleneck.  Nodes near 0 are AT the bottleneck.
    """
    L = nx.laplacian_matrix(G, weight='weight').toarray().astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(L)
    # λ₂ is the second-smallest eigenvalue
    idx = np.argsort(eigvals)
    return eigvecs[:, idx[1]]


# ──────────────────────────────────────────────────────────────────
# Scalar circuit quantities
# ──────────────────────────────────────────────────────────────────

def effective_resistance_pair(G: nx.Graph, i: int, j: int) -> float:
    """
    Compute effective resistance between a single pair of nodes.
    """
    L_pinv = laplacian_pseudoinverse(G)
    return float(L_pinv[i, i] + L_pinv[j, j] - 2.0 * L_pinv[i, j])


def kirchhoff_index(R: np.ndarray) -> float:
    """
    Kirchhoff index = ½ Σ R_eff(i,j).

    This equals n × Σ(1/λ_k) for non-zero eigenvalues.
    It is the total effective resistance of the circuit,
    analogous to total resistance in a resistor network.
    """
    return float(np.sum(R) / 2.0)


def resistance_centrality(R: np.ndarray, node: int) -> float:
    """
    Resistance centrality = (n-1) / Σ_j R_eff(node, j).

    This is the "effective conductance" centrality — how easily current
    can reach all other nodes from this node.  Equivalent to
    information centrality / current-flow closeness centrality.
    """
    n = R.shape[0]
    total_R = np.sum(R[node])
    return float((n - 1) / total_R) if total_R > 0 else 0.0


# ──────────────────────────────────────────────────────────────────
# Current-flow primitives (Kirchhoff's laws)
# ──────────────────────────────────────────────────────────────────

def compute_node_voltages(
    Lp: np.ndarray,
    source: int,
    sinks: list,
) -> np.ndarray:
    """
    Compute node voltages when +1 A is injected at *source* and
    removed uniformly across *sinks*.

    By superposition:
        V = L⁺ · I_ext  =  L⁺[:, source] − (1/|sinks|) Σ_{s ∈ sinks} L⁺[:, s]

    This is O(n) since L⁺ is already computed.

    Parameters
    ----------
    Lp : np.ndarray
        n × n Laplacian pseudoinverse.
    source : int
        Node where +1 A is injected.
    sinks : list of int
        Nodes where current is collected (each receives −1/|sinks| A).

    Returns
    -------
    V : np.ndarray
        n-vector of node voltages.
    """
    V = Lp[:, source].copy()
    if sinks:
        sink_mean = np.mean(Lp[:, sinks], axis=1)
        V -= sink_mean
    return V


def compute_edge_currents(
    G: nx.Graph,
    voltages: np.ndarray,
) -> np.ndarray:
    """
    Compute Kirchhoff edge currents from node voltages.

    For each edge (u, v) with conductance G_uv:
        I_{uv} = G_uv · (V_u − V_v)

    Positive current means flow from u → v.

    Parameters
    ----------
    G : nx.Graph
        Graph with 'weight' = conductance on edges.
    voltages : np.ndarray
        Node voltage vector (from compute_node_voltages).

    Returns
    -------
    edge_currents : np.ndarray
        1-D array of signed currents, one per edge
        (in G.edges() order).
    """
    edges = list(G.edges(data=True))
    currents = np.empty(len(edges))
    for k, (u, v, d) in enumerate(edges):
        g_uv = d.get('weight', 1.0)
        currents[k] = g_uv * (voltages[u] - voltages[v])
    return currents


def compute_edge_current_dict(
    G: nx.Graph,
    voltages: np.ndarray,
) -> dict:
    """
    Same as compute_edge_currents but returns {(u,v): I} dict.
    """
    return {
        (u, v): d.get('weight', 1.0) * (voltages[u] - voltages[v])
        for u, v, d in G.edges(data=True)
    }


def site_edge_currents(
    G: nx.Graph,
    voltages: np.ndarray,
    node: int,
) -> np.ndarray:
    """
    Return absolute currents through edges incident to *node*.

    This tells us how much current passes through the mutation site.
    """
    return np.array([
        abs(d.get('weight', 1.0) * (voltages[node] - voltages[v]))
        for v, d in G[node].items()
    ])


def current_entropy(edge_currents: np.ndarray) -> float:
    """
    Shannon entropy of the normalised |edge current| distribution.

    High entropy → current distributed evenly (robust circuit).
    Low entropy  → current funnelled through few wires (fragile).
    """
    abs_I = np.abs(edge_currents)
    total = abs_I.sum()
    if total < 1e-15:
        return 0.0
    p = abs_I / total
    p = p[p > 1e-15]           # avoid log(0)
    return float(-np.sum(p * np.log(p)))


def herfindahl_index(values: np.ndarray) -> float:
    """
    Herfindahl–Hirschman Index of concentration.

    HHI = Σ (s_i)²  where s_i = |v_i| / Σ|v|.
    HHI = 1   → all flow through one element (monopoly).
    HHI = 1/n → perfectly uniform.
    """
    abs_v = np.abs(values)
    total = abs_v.sum()
    if total < 1e-15:
        return 0.0
    shares = abs_v / total
    return float(np.sum(shares ** 2))


# ──────────────────────────────────────────────────────────────────
# Power dissipation primitives (P = I²/G = I·V per edge)
# ──────────────────────────────────────────────────────────────────

def compute_edge_power(
    G: nx.Graph,
    voltages: np.ndarray,
) -> np.ndarray:
    """
    Compute Joule heating (power dissipation) at each edge.

    P_e = I_e² / G_e = G_e · (V_u − V_v)²

    Units: watts when V in volts and G in siemens.

    Returns
    -------
    edge_power : np.ndarray
        1-D array of non-negative power values, one per edge.
    """
    edges = list(G.edges(data=True))
    power = np.empty(len(edges))
    for k, (u, v, d) in enumerate(edges):
        g_uv = d.get('weight', 1.0)
        dv = voltages[u] - voltages[v]
        power[k] = g_uv * dv * dv          # G·ΔV²
    return power


def power_entropy(edge_power: np.ndarray) -> float:
    """
    Shannon entropy of the normalised edge-power distribution.

    Uniform power → high entropy (no bottleneck).
    Concentrated power → low entropy (one wire does all the work).
    """
    total = edge_power.sum()
    if total < 1e-15:
        return 0.0
    p = edge_power / total
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def site_edge_power(
    G: nx.Graph,
    voltages: np.ndarray,
    node: int,
) -> np.ndarray:
    """
    Power dissipated at each edge incident to *node*.
    """
    return np.array([
        d.get('weight', 1.0) * (voltages[node] - voltages[v]) ** 2
        for v, d in G[node].items()
    ])


# ──────────────────────────────────────────────────────────────────
# Multi-scale resistance (shell analysis)
# ──────────────────────────────────────────────────────────────────

def resistance_shells(
    R: np.ndarray,
    G: nx.Graph,
    node: int,
    max_hops: int = 3,
) -> dict:
    """
    Mean effective resistance to graph-distance shells around *node*.

    Shell k = nodes exactly k hops away in the graph topology.

    Returns dict  {1: mean_R_shell_1, 2: mean_R_shell_2, ...}
    """
    # BFS to find shells
    shells: dict = {}
    visited = {node}
    frontier = {node}
    for hop in range(1, max_hops + 1):
        next_frontier = set()
        for u in frontier:
            for v in G.neighbors(u):
                if v not in visited:
                    next_frontier.add(v)
                    visited.add(v)
        if not next_frontier:
            break
        r_vals = [R[node, v] for v in next_frontier]
        shells[hop] = float(np.mean(r_vals))
        frontier = next_frontier
    return shells


def voltage_influence_radius(
    Lp: np.ndarray,
    node: int,
    threshold_frac: float = 0.1,
) -> int:
    """
    Number of nodes where |L⁺[j, node]| > threshold × L⁺[node, node].

    Measures how far the voltage influence of *node* extends.
    Small radius → isolated node.  Large radius → influential hub.
    """
    self_pot = abs(Lp[node, node])
    if self_pot < 1e-15:
        return 0
    col = np.abs(Lp[:, node])
    return int(np.sum(col > threshold_frac * self_pot)) - 1   # exclude self


def total_conductance(G: nx.Graph) -> float:
    """
    Sum of all edge conductances = total 'wire material' in the circuit.
    Used for commute-time computation: T_commute(i,j) = 2·m·R_eff(i,j).
    """
    return float(sum(d.get('weight', 1.0) for _, _, d in G.edges(data=True)))


# ──────────────────────────────────────────────────────────────────
# Green's function (L⁺) row statistics
# ──────────────────────────────────────────────────────────────────

def green_row_entropy(Lp: np.ndarray, node: int) -> float:
    """
    Shannon entropy of |L⁺[node, :]| (normalised).

    High entropy → voltage influence spread evenly across the network.
    Low entropy  → concentrated near a few nodes.
    """
    row = np.abs(Lp[node, :])
    total = row.sum()
    if total < 1e-15:
        return 0.0
    p = row / total
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def green_anisotropy(Lp: np.ndarray, node: int) -> float:
    """
    Coefficient of variation of |L⁺[node, :]|.

    Measures directional bias: high → voltage influence is very
    non-uniform (the node is near one 'side' of the circuit).
    """
    row = np.abs(Lp[node, :])
    mean_val = row.mean()
    if mean_val < 1e-15:
        return 0.0
    return float(row.std() / mean_val)


# ──────────────────────────────────────────────────────────────────
# Spectral descriptors
# ──────────────────────────────────────────────────────────────────

def eigenvalue_entropy(eigs: np.ndarray) -> float:
    """
    Shannon entropy of normalised non-zero Laplacian eigenvalues.

    High entropy → eigenvalues spread evenly (homogeneous circuit).
    Low entropy  → a few modes dominate (structured / modular circuit).
    """
    nonzero = eigs[eigs > 1e-10]
    if len(nonzero) == 0:
        return 0.0
    total = nonzero.sum()
    if total < 1e-15:
        return 0.0
    p = nonzero / total
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


# ──────────────────────────────────────────────────────────────────
# Resistance distance geometry helpers
# ──────────────────────────────────────────────────────────────────

def resistance_row_harmonic_mean(R: np.ndarray, node: int) -> float:
    """
    Harmonic mean of R_eff(node, j) for j ≠ node.

    Biased toward small R values — captures how close the node is to
    its nearest electrical neighbours.  Analogous to harmonic closeness
    centrality but in resistance-distance space.
    """
    row = np.delete(R[node], node)
    row = row[row > 1e-15]
    if len(row) == 0:
        return 0.0
    return float(len(row) / np.sum(1.0 / row))


def resistance_row_statistics(R: np.ndarray, node: int) -> dict:
    """
    Compute median, skewness, and kurtosis of R_eff(node, :).

    These higher-order statistics capture the *shape* of the resistance
    landscape around the node — information lost by just using mean/std.
    """
    from scipy.stats import skew, kurtosis
    row = np.delete(R[node], node)
    if len(row) < 3:
        return {'median': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}
    return {
        'median': float(np.median(row)),
        'skewness': float(skew(row)),
        'kurtosis': float(kurtosis(row)),
    }
