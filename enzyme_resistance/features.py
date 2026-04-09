"""
Step 4: Extract Pure Circuit-Theoretic Features.

Every feature is derived from treating the protein contact graph as an
electrical resistor network.  No direct physicochemical amino-acid
properties are used — those are already encoded in the edge-weight
perturbation model (mutation.py).  Here we only read the *circuit*
response.

Ohm's law has three pillars:  V = I · R.  Plus power P = I²R, and
circuit dynamics (RC time constants, commute times).

83 features in 15 groups:

A. RESISTANCE DISTANCE  (6)   — R_eff landscape shifts
B. GLOBAL CIRCUIT HEALTH  (4) — Kirchhoff, λ₂, spectral gap
C. CURRENT FLOW CENTRALITY  (4) — betweenness, closeness
D. VOLTAGE / POTENTIAL  (4)   — L⁺ Green's function
E. SPECTRAL MODE  (3)         — Fiedler, conductance degree
F. CURRENT FLOW PATTERNS  (12) — Kirchhoff edge currents
G. VOLTAGE TRANSFER & COUPLING  (5) — signal transmission efficiency
H. POWER DISSIPATION  (5)    — Joule heating landscape P = I²/G
I. MULTI-SCALE RESISTANCE & DYNAMICS  (5) — shells, commute time, redundancy
J. GREEN'S FUNCTION PROPAGATOR  (5) — L⁺ row statistics
K. THEVENIN / NORTON EQUIVALENT  (4) — equivalent circuit parameters
L. SPECTRAL CIRCUIT DESCRIPTORS  (4) — eigenvalue distribution
M. RESISTANCE DISTANCE GEOMETRY  (8) — higher-order R statistics
N. FOLD-CHANGE RATIOS  (5)   — multiplicative mutation effects
O. MULTI-SCALE EXTENDED  (4) — 3-hop shell, gradient, local/global
P. CROSS-FEATURE PRODUCTS  (5) — dimensionless physical combinations
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from enzyme_resistance.resistance import (
    effective_resistance_matrix_with_pinv,
    kirchhoff_index,
    resistance_centrality,
    laplacian_eigenvalues,
    fiedler_vector,
    compute_node_voltages,
    compute_edge_currents,
    site_edge_currents,
    current_entropy,
    herfindahl_index,
    compute_edge_power,
    power_entropy,
    site_edge_power,
    resistance_shells,
    voltage_influence_radius,
    total_conductance,
    green_row_entropy,
    green_anisotropy,
    eigenvalue_entropy,
    resistance_row_harmonic_mean,
    resistance_row_statistics,
)


@dataclass
class WTProperties:
    """Pre-computed wild-type circuit properties.

    Computing the pseudoinverse, eigenvalues, Fiedler vector, and
    current-flow betweenness is O(N³).  These only depend on the WT
    graph, so we compute them once per protein and reuse across all
    mutations on that protein.
    """
    R_wt: np.ndarray
    Lp_wt: np.ndarray
    eigs_wt: np.ndarray
    fv_wt: np.ndarray
    cfb_wt: Dict[int, float]
    active_site_residues: List[int]
    Kf_wt: float


def compute_wt_properties(G_wt: nx.Graph) -> WTProperties:
    """Compute all expensive WT-only circuit properties once.

    Raises ValueError if the graph has fewer than 2 nodes.
    """
    n = G_wt.number_of_nodes()
    if n < 2:
        raise ValueError(f"Graph too small ({n} nodes) for circuit analysis")

    R_wt, Lp_wt = effective_resistance_matrix_with_pinv(G_wt)
    eigs_wt = laplacian_eigenvalues(G_wt)

    try:
        fv_wt = fiedler_vector(G_wt)
    except (np.linalg.LinAlgError, nx.NetworkXError):
        fv_wt = np.zeros(n)

    try:
        cfb_wt = nx.current_flow_betweenness_centrality(
            G_wt, weight='weight', normalized=True,
        )
    except (nx.NetworkXError, np.linalg.LinAlgError):
        cfb_wt = {}

    # Proxy active site: top-5 betweenness-centrality nodes
    bc = nx.betweenness_centrality(G_wt, weight='weight')
    sorted_nodes = sorted(bc.keys(), key=lambda x: bc[x], reverse=True)
    active_site_residues = sorted_nodes[:5]

    Kf_wt = kirchhoff_index(R_wt)

    return WTProperties(
        R_wt=R_wt,
        Lp_wt=Lp_wt,
        eigs_wt=eigs_wt,
        fv_wt=fv_wt,
        cfb_wt=cfb_wt,
        active_site_residues=active_site_residues,
        Kf_wt=Kf_wt,
    )


def extract_resistance_features(
    G_wt: nx.Graph,
    G_mut: nx.Graph,
    mutation_site: int,
    active_site_residues: Optional[List[int]] = None,
    perturbation_threshold: float = 0.01,
    wt_props: Optional[WTProperties] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Extract 48 pure circuit-theoretic features comparing wild-type
    and mutant resistor networks.

    Parameters
    ----------
    G_wt : nx.Graph
        Wild-type contact graph.
    G_mut : nx.Graph
        Mutant contact graph (edges perturbed at mutation_site).
    mutation_site : int
        Node index of the mutated residue.
    active_site_residues : list of int, optional
        Functional-hub node indices.  If None, the 5 highest
        betweenness-centrality nodes are used as a proxy.
    perturbation_threshold : float
        Threshold for counting affected residues (propagation_radius).
    wt_props : WTProperties, optional
        Pre-computed WT circuit properties.  When supplied, the expensive
        WT computations (pseudoinverse, eigenvalues, Fiedler vector,
        current-flow betweenness) are skipped.

    Returns
    -------
    dict
        Feature name → float value.
    """
    n = G_wt.number_of_nodes()

    # ── Use cached WT properties or compute fresh ─────────────────
    if wt_props is not None:
        R_wt = wt_props.R_wt
        Lp_wt = wt_props.Lp_wt
    else:
        R_wt, Lp_wt = effective_resistance_matrix_with_pinv(G_wt)

    R_mut, Lp_mut = effective_resistance_matrix_with_pinv(G_mut)

    features: Dict[str, float] = {}

    # ── Proxy active site ─────────────────────────────────────────
    if active_site_residues is None or len(active_site_residues) == 0:
        if wt_props is not None:
            active_site_residues = [nd for nd in wt_props.active_site_residues
                                    if nd != mutation_site]
        else:
            bc = nx.betweenness_centrality(G_wt, weight='weight')
            sorted_nodes = sorted(bc.keys(), key=lambda x: bc[x], reverse=True)
            active_site_residues = [nd for nd in sorted_nodes[:5]
                                    if nd != mutation_site]

    # ═══════════════════════════════════════════════════════════════
    # A. RESISTANCE DISTANCE  (6)
    # ═══════════════════════════════════════════════════════════════

    delta_row = R_mut[mutation_site] - R_wt[mutation_site]
    abs_delta_row = np.abs(delta_row)

    # 1. Mean R change to active-site residues
    if len(active_site_residues) > 0:
        features['delta_R_active_site'] = float(np.mean([
            delta_row[a] for a in active_site_residues
            if a < n and mutation_site < n
        ]))
    else:
        features['delta_R_active_site'] = 0.0

    # 2. Mean R change to all residues
    features['delta_R_global_mean'] = float(np.mean(delta_row))

    # 3. Max absolute R change
    features['delta_R_max'] = float(np.max(abs_delta_row))

    # 4. Spread of R perturbation (anisotropy)
    features['delta_R_std'] = float(np.std(delta_row))

    # 5. Resistance eccentricity in WT (electrical remoteness)
    features['R_eccentricity_wt'] = float(np.max(R_wt[mutation_site]))

    # 6. Change in eccentricity
    features['delta_R_eccentricity'] = float(
        np.max(R_mut[mutation_site]) - np.max(R_wt[mutation_site])
    )

    # ═══════════════════════════════════════════════════════════════
    # B. GLOBAL CIRCUIT HEALTH  (4)
    # ═══════════════════════════════════════════════════════════════

    # 7. Kirchhoff index shift (total network resistance)
    Kf_wt = wt_props.Kf_wt if wt_props is not None else kirchhoff_index(R_wt)
    Kf_mut = kirchhoff_index(R_mut)
    features['delta_kirchhoff'] = Kf_mut - Kf_wt

    # 8. Propagation radius (# residues with significant R change)
    features['propagation_radius'] = float(
        np.sum(abs_delta_row > perturbation_threshold)
    )

    # 9. Algebraic connectivity shift (bottleneck conductance)
    eigs_wt = wt_props.eigs_wt if wt_props is not None else laplacian_eigenvalues(G_wt)
    eigs_mut = laplacian_eigenvalues(G_mut)
    lambda2_wt = _lambda_k(eigs_wt, 2)
    lambda2_mut = _lambda_k(eigs_mut, 2)
    features['delta_algebraic_connectivity'] = lambda2_mut - lambda2_wt

    # 10. Spectral gap ratio λ₂/λ₃ in WT (single-bottleneck indicator)
    lambda3_wt = _lambda_k(eigs_wt, 3)
    features['spectral_gap_ratio'] = (
        lambda2_wt / lambda3_wt if lambda3_wt > 1e-12 else 0.0
    )

    # ═══════════════════════════════════════════════════════════════
    # C. CURRENT FLOW CENTRALITY  (4)
    # ═══════════════════════════════════════════════════════════════

    # 11–12. Current-flow betweenness (proportion of current through
    #        the mutation site when unit current flows between all
    #        pairs — the true Kirchhoff-law circuit betweenness).
    try:
        if wt_props is not None:
            cfb_wt = wt_props.cfb_wt
        else:
            cfb_wt = nx.current_flow_betweenness_centrality(
                G_wt, weight='weight', normalized=True,
            )
        cfb_mut = nx.current_flow_betweenness_centrality(
            G_mut, weight='weight', normalized=True,
        )
        features['current_flow_betweenness_wt'] = cfb_wt.get(mutation_site, 0.0)
        features['delta_current_flow_betweenness'] = (
            cfb_mut.get(mutation_site, 0.0) - cfb_wt.get(mutation_site, 0.0)
        )
    except (nx.NetworkXError, np.linalg.LinAlgError):
        features['current_flow_betweenness_wt'] = 0.0
        features['delta_current_flow_betweenness'] = 0.0

    # 13. Resistance centrality in WT  [ (n-1) / Σ_j R(i,j) ]
    #     = information centrality = current-flow closeness
    features['resistance_centrality_wt'] = resistance_centrality(
        R_wt, mutation_site,
    )

    # 14. Change in resistance centrality
    rc_wt = resistance_centrality(R_wt, mutation_site)
    rc_mut = resistance_centrality(R_mut, mutation_site)
    features['delta_resistance_centrality'] = rc_mut - rc_wt

    # ═══════════════════════════════════════════════════════════════
    # D. VOLTAGE / POTENTIAL  (4)
    #    L⁺ is the Green's function of the circuit.
    # ═══════════════════════════════════════════════════════════════

    # 15. Self-potential at mutation site in WT
    features['voltage_wt'] = float(Lp_wt[mutation_site, mutation_site])

    # 16. Change in self-potential
    features['delta_voltage'] = float(
        Lp_mut[mutation_site, mutation_site] - Lp_wt[mutation_site, mutation_site]
    )

    # 17–18. Transfer voltage to active site
    if len(active_site_residues) > 0:
        tv_wt = float(np.mean([
            Lp_wt[mutation_site, a] for a in active_site_residues if a < n
        ]))
        tv_mut = float(np.mean([
            Lp_mut[mutation_site, a] for a in active_site_residues if a < n
        ]))
        features['transfer_voltage_active_wt'] = tv_wt
        features['delta_transfer_voltage_active'] = tv_mut - tv_wt
    else:
        features['transfer_voltage_active_wt'] = 0.0
        features['delta_transfer_voltage_active'] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # E. SPECTRAL MODE  (3)
    # ═══════════════════════════════════════════════════════════════

    # 19–20. Fiedler vector component at mutation site
    #        (position relative to circuit bottleneck)
    try:
        fv_wt = wt_props.fv_wt if wt_props is not None else fiedler_vector(G_wt)
        fv_mut = fiedler_vector(G_mut)
        features['fiedler_component_wt'] = float(fv_wt[mutation_site])
        features['delta_fiedler_component'] = float(
            fv_mut[mutation_site] - fv_wt[mutation_site]
        )
    except (np.linalg.LinAlgError, nx.NetworkXError):
        features['fiedler_component_wt'] = 0.0
        features['delta_fiedler_component'] = 0.0

    # 21. Conductance degree (sum of edge conductances at mutation site)
    #     = local "wire gauge" / how well-connected this node is
    features['conductance_degree_wt'] = float(
        sum(d['weight'] for _, _, d in G_wt.edges(mutation_site, data=True))
    )

    # ═══════════════════════════════════════════════════════════════
    # F. CURRENT FLOW PATTERNS  (12)
    #    The missing third of Ohm's law: I = V / R.
    #    Inject +1 A at mutation site, sink at active-site residues,
    #    and solve Kirchhoff's equations for actual edge currents.
    # ═══════════════════════════════════════════════════════════════

    # Determine sink nodes (active site residues, excluding mut site)
    current_sinks = [a for a in active_site_residues
                     if a != mutation_site and a < n]
    if not current_sinks:
        # Fallback: use all nodes except mutation site
        current_sinks = [j for j in range(n) if j != mutation_site]

    try:
        # ── Compute node voltages (V = L⁺ · I_ext) ───────────────
        V_wt = compute_node_voltages(Lp_wt, mutation_site, current_sinks)
        V_mut = compute_node_voltages(Lp_mut, mutation_site, current_sinks)

        # ── Compute edge currents (I = G · ΔV) ───────────────────
        I_wt = compute_edge_currents(G_wt, V_wt)
        I_mut = compute_edge_currents(G_mut, V_mut)

        # ── Site-level edge currents ──────────────────────────────
        site_I_wt = site_edge_currents(G_wt, V_wt, mutation_site)
        site_I_mut = site_edge_currents(G_mut, V_mut, mutation_site)

        # 22–23. Effective conductance to active site  (G_eff = 1/R_eff)
        R_to_active_wt = float(np.mean([
            R_wt[mutation_site, a] for a in current_sinks
        ])) if current_sinks else 1.0
        R_to_active_mut = float(np.mean([
            R_mut[mutation_site, a] for a in current_sinks
        ])) if current_sinks else 1.0
        G_eff_wt = 1.0 / R_to_active_wt if R_to_active_wt > 1e-15 else 0.0
        G_eff_mut = 1.0 / R_to_active_mut if R_to_active_mut > 1e-15 else 0.0
        features['effective_conductance_to_active_wt'] = G_eff_wt
        features['delta_effective_conductance_to_active'] = G_eff_mut - G_eff_wt

        # 24–25. Bottleneck current (max |I| through any single edge)
        abs_I_wt = np.abs(I_wt)
        abs_I_mut = np.abs(I_mut)
        features['max_edge_current_wt'] = float(abs_I_wt.max()) if len(I_wt) > 0 else 0.0
        features['delta_max_edge_current'] = (
            float(abs_I_mut.max()) - float(abs_I_wt.max())
        ) if len(I_wt) > 0 and len(I_mut) > 0 else 0.0

        # 26–27. Current entropy (uniformity of current distribution)
        features['current_entropy_wt'] = current_entropy(I_wt)
        features['delta_current_entropy'] = (
            current_entropy(I_mut) - current_entropy(I_wt)
        )

        # 28–29. Site current fraction (how much flows through mut site)
        total_I_wt = abs_I_wt.sum()
        total_I_mut = abs_I_mut.sum()
        site_frac_wt = (
            site_I_wt.sum() / total_I_wt if total_I_wt > 1e-15 else 0.0
        )
        site_frac_mut = (
            site_I_mut.sum() / total_I_mut if total_I_mut > 1e-15 else 0.0
        )
        features['site_current_fraction_wt'] = float(site_frac_wt)
        features['delta_site_current_fraction'] = float(site_frac_mut - site_frac_wt)

        # 30. Current redistribution (global rerouting measure)
        #     ‖I_mut − I_wt‖₂ / ‖I_wt‖₂
        norm_wt = np.linalg.norm(I_wt)
        if norm_wt > 1e-15 and len(I_wt) == len(I_mut):
            features['current_redistribution'] = float(
                np.linalg.norm(I_mut - I_wt) / norm_wt
            )
        else:
            features['current_redistribution'] = 0.0

        # 31–32. Current concentration at mutation site (Herfindahl index)
        features['current_concentration_wt'] = herfindahl_index(site_I_wt)
        features['delta_current_concentration'] = (
            herfindahl_index(site_I_mut) - herfindahl_index(site_I_wt)
        )

        # 33. Neighbor current asymmetry (directional bias)
        if len(site_I_wt) >= 2:
            sorted_I = np.sort(site_I_wt)[::-1]
            min_I = sorted_I[-1]
            features['neighbor_current_asymmetry_wt'] = float(
                sorted_I[0] / min_I if min_I > 1e-15
                else sorted_I[0] / 1e-15
            )
        else:
            features['neighbor_current_asymmetry_wt'] = 1.0

    except (np.linalg.LinAlgError, nx.NetworkXError, ValueError,
            IndexError, ZeroDivisionError):
        # Safe fallback for all current-flow features
        features['effective_conductance_to_active_wt'] = 0.0
        features['delta_effective_conductance_to_active'] = 0.0
        features['max_edge_current_wt'] = 0.0
        features['delta_max_edge_current'] = 0.0
        features['current_entropy_wt'] = 0.0
        features['delta_current_entropy'] = 0.0
        features['site_current_fraction_wt'] = 0.0
        features['delta_site_current_fraction'] = 0.0
        features['current_redistribution'] = 0.0
        features['current_concentration_wt'] = 0.0
        features['delta_current_concentration'] = 0.0
        features['neighbor_current_asymmetry_wt'] = 1.0
        V_wt = V_mut = I_wt = I_mut = None        # mark unavailable

    # ═══════════════════════════════════════════════════════════════
    # G. VOLTAGE TRANSFER & COUPLING  (5)
    #    Analogous to S-parameters / transfer matrices.
    #    How efficiently does the mutation site's voltage reach
    #    the active site?
    # ═══════════════════════════════════════════════════════════════

    try:
        # Pick a representative active-site node
        active_rep = current_sinks[0] if current_sinks else 0

        # 34–35. Voltage transfer ratio  =  L⁺[active, mut] / L⁺[mut, mut]
        self_pot_wt = Lp_wt[mutation_site, mutation_site]
        self_pot_mut = Lp_mut[mutation_site, mutation_site]
        vtr_wt = (
            float(np.mean([Lp_wt[a, mutation_site] for a in current_sinks]))
            / self_pot_wt if abs(self_pot_wt) > 1e-15 else 0.0
        )
        vtr_mut = (
            float(np.mean([Lp_mut[a, mutation_site] for a in current_sinks]))
            / self_pot_mut if abs(self_pot_mut) > 1e-15 else 0.0
        )
        features['voltage_transfer_ratio_wt'] = float(vtr_wt)
        features['delta_voltage_transfer_ratio'] = float(vtr_mut - vtr_wt)

        # 36–37. Mutual conductance (normalised coupling ∈ [-1, 1])
        #        G_mutual = L⁺[i,j] / √(L⁺[i,i]·L⁺[j,j])
        Lp_act_wt = float(np.mean([
            Lp_wt[mutation_site, a] for a in current_sinks
        ]))
        Lp_act_act_wt = float(np.mean([
            Lp_wt[a, a] for a in current_sinks
        ]))
        denom_wt = np.sqrt(abs(self_pot_wt * Lp_act_act_wt))
        mc_wt = Lp_act_wt / denom_wt if denom_wt > 1e-15 else 0.0

        Lp_act_mut = float(np.mean([
            Lp_mut[mutation_site, a] for a in current_sinks
        ]))
        Lp_act_act_mut = float(np.mean([
            Lp_mut[a, a] for a in current_sinks
        ]))
        denom_mut = np.sqrt(abs(self_pot_mut * Lp_act_act_mut))
        mc_mut = Lp_act_mut / denom_mut if denom_mut > 1e-15 else 0.0

        features['mutual_conductance_wt'] = float(mc_wt)
        features['delta_mutual_conductance'] = float(mc_mut - mc_wt)

        # 38. Voltage influence radius
        features['voltage_influence_radius_wt'] = float(
            voltage_influence_radius(Lp_wt, mutation_site, threshold_frac=0.1)
        )
    except (IndexError, ValueError, ZeroDivisionError):
        features['voltage_transfer_ratio_wt'] = 0.0
        features['delta_voltage_transfer_ratio'] = 0.0
        features['mutual_conductance_wt'] = 0.0
        features['delta_mutual_conductance'] = 0.0
        features['voltage_influence_radius_wt'] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # H. POWER DISSIPATION LANDSCAPE  (5)
    #    P = I²/G = G·(ΔV)² at each edge.
    #    WHERE the circuit dissipates energy reveals the bottlenecks
    #    that a mutation can disrupt.
    # ═══════════════════════════════════════════════════════════════

    try:
        if V_wt is not None and V_mut is not None:
            P_wt = compute_edge_power(G_wt, V_wt)
            P_mut = compute_edge_power(G_mut, V_mut)

            # 39–40. Power local fraction (mutation-site Joule heating)
            P_site_wt = site_edge_power(G_wt, V_wt, mutation_site)
            P_site_mut = site_edge_power(G_mut, V_mut, mutation_site)
            P_total_wt = P_wt.sum()
            P_total_mut = P_mut.sum()
            plf_wt = P_site_wt.sum() / P_total_wt if P_total_wt > 1e-15 else 0.0
            plf_mut = P_site_mut.sum() / P_total_mut if P_total_mut > 1e-15 else 0.0
            features['power_local_fraction_wt'] = float(plf_wt)
            features['delta_power_local_fraction'] = float(plf_mut - plf_wt)

            # 41–42. Power entropy (uniformity of energy dissipation)
            pe_wt = power_entropy(P_wt)
            pe_mut = power_entropy(P_mut)
            features['power_entropy_wt'] = pe_wt
            features['delta_power_entropy'] = pe_mut - pe_wt

            # 43. Max power edge (hottest single wire)
            features['max_power_edge_wt'] = float(P_wt.max()) if len(P_wt) > 0 else 0.0
        else:
            raise ValueError("Voltages not available")
    except (ValueError, IndexError, ZeroDivisionError):
        features['power_local_fraction_wt'] = 0.0
        features['delta_power_local_fraction'] = 0.0
        features['power_entropy_wt'] = 0.0
        features['delta_power_entropy'] = 0.0
        features['max_power_edge_wt'] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # I. MULTI-SCALE RESISTANCE & DYNAMICS  (5)
    #    Distance-shell analysis reveals how the mutation's electrical
    #    effect decays with topological distance.  Commute time bridges
    #    resistance → random-walk dynamics.  Path redundancy tells us
    #    how many parallel wires carry current.
    # ═══════════════════════════════════════════════════════════════

    try:
        # 44–46. Resistance shells (1-hop & 2-hop mean R)
        shells_wt = resistance_shells(R_wt, G_wt, mutation_site, max_hops=2)
        shells_mut = resistance_shells(R_mut, G_mut, mutation_site, max_hops=2)
        r1_wt = shells_wt.get(1, 0.0)
        r2_wt = shells_wt.get(2, 0.0)
        r1_mut = shells_mut.get(1, 0.0)
        r2_mut = shells_mut.get(2, 0.0)
        features['R_shell_1_mean_wt'] = float(r1_wt)
        features['R_shell_2_mean_wt'] = float(r2_wt)

        # Shell ratio change = how the resistance gradient shifts
        ratio_wt = r2_wt / r1_wt if r1_wt > 1e-15 else 0.0
        ratio_mut = r2_mut / r1_mut if r1_mut > 1e-15 else 0.0
        features['delta_R_shell_ratio'] = float(ratio_mut - ratio_wt)

        # 47. Commute time to active site
        #     T_commute(i,j) = 2 × total_conductance × R_eff(i,j)
        m_wt = total_conductance(G_wt)
        R_to_active = float(np.mean([
            R_wt[mutation_site, a] for a in current_sinks
        ])) if current_sinks else 0.0
        features['commute_time_to_active_wt'] = 2.0 * m_wt * R_to_active

        # 48. Effective path count = 1 / HHI(edge_currents)
        #     = effective number of parallel paths carrying current
        #     High → robust (many paths).  Low → fragile (one path).
        if I_wt is not None and len(I_wt) > 0:
            hhi_wt = herfindahl_index(I_wt)
            features['effective_path_count_wt'] = (
                1.0 / hhi_wt if hhi_wt > 1e-15 else float(len(I_wt))
            )
        else:
            features['effective_path_count_wt'] = 0.0

    except (ValueError, IndexError, ZeroDivisionError, KeyError):
        features['R_shell_1_mean_wt'] = 0.0
        features['R_shell_2_mean_wt'] = 0.0
        features['delta_R_shell_ratio'] = 0.0
        features['commute_time_to_active_wt'] = 0.0
        features['effective_path_count_wt'] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # J. GREEN'S FUNCTION PROPAGATOR  (5)
    #    L⁺ is the Green's function of the resistor network.
    #    Its row at the mutation site encodes how voltage influence
    #    spreads — entropy, anisotropy, and trace fraction capture
    #    complementary information.
    # ═══════════════════════════════════════════════════════════════

    try:
        # 49. Green's function trace fraction = L⁺[i,i] / Tr(L⁺)
        #     = relative self-potential = "electrical leverage"
        trace_wt = np.trace(Lp_wt)
        features['green_trace_fraction_wt'] = (
            float(Lp_wt[mutation_site, mutation_site] / trace_wt)
            if abs(trace_wt) > 1e-15 else 0.0
        )

        # 50–51. Green's function row entropy (spread of voltage influence)
        features['green_row_entropy_wt'] = green_row_entropy(Lp_wt, mutation_site)
        features['delta_green_row_entropy'] = (
            green_row_entropy(Lp_mut, mutation_site)
            - green_row_entropy(Lp_wt, mutation_site)
        )

        # 52–53. Green's function anisotropy (directional bias)
        features['green_anisotropy_wt'] = green_anisotropy(Lp_wt, mutation_site)
        features['delta_green_anisotropy'] = (
            green_anisotropy(Lp_mut, mutation_site)
            - green_anisotropy(Lp_wt, mutation_site)
        )
    except (ValueError, IndexError, ZeroDivisionError):
        features['green_trace_fraction_wt'] = 0.0
        features['green_row_entropy_wt'] = 0.0
        features['delta_green_row_entropy'] = 0.0
        features['green_anisotropy_wt'] = 0.0
        features['delta_green_anisotropy'] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # K. THEVENIN / NORTON EQUIVALENT CIRCUIT  (4)
    #    Every two-terminal sub-circuit has a Thevenin equivalent:
    #    an ideal voltage source in series with R_Thevenin.
    #    These features capture the equivalent circuit between the
    #    mutation site and the active site.
    # ═══════════════════════════════════════════════════════════════

    try:
        # 54. Absolute WT resistance to active site (Thevenin R)
        R_active_wt_abs = float(np.mean([
            R_wt[mutation_site, a] for a in current_sinks
        ])) if current_sinks else 0.0
        features['R_active_site_wt'] = R_active_wt_abs

        # 55–56. Circuit gain = V_active / V_self
        #        = fraction of the mutation site's self-potential
        #          that actually reaches the active site
        V_self_wt = Lp_wt[mutation_site, mutation_site]
        V_self_mut = Lp_mut[mutation_site, mutation_site]
        if current_sinks and abs(V_self_wt) > 1e-15:
            V_active_mean_wt = float(np.mean([
                Lp_wt[a, mutation_site] for a in current_sinks
            ]))
            gain_wt = V_active_mean_wt / V_self_wt
        else:
            gain_wt = 0.0

        if current_sinks and abs(V_self_mut) > 1e-15:
            V_active_mean_mut = float(np.mean([
                Lp_mut[a, mutation_site] for a in current_sinks
            ]))
            gain_mut = V_active_mean_mut / V_self_mut
        else:
            gain_mut = 0.0

        features['circuit_gain_wt'] = float(gain_wt)
        features['delta_circuit_gain'] = float(gain_mut - gain_wt)

        # 57. Norton current capacity = V_self / R_active
        #     = maximum current the mutation site can drive to active
        features['norton_current_wt'] = (
            float(V_self_wt / R_active_wt_abs)
            if R_active_wt_abs > 1e-15 else 0.0
        )
    except (ValueError, IndexError, ZeroDivisionError):
        features['R_active_site_wt'] = 0.0
        features['circuit_gain_wt'] = 0.0
        features['delta_circuit_gain'] = 0.0
        features['norton_current_wt'] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # L. SPECTRAL CIRCUIT DESCRIPTORS  (4)
    #    Beyond λ₂ and λ₃ — the full eigenvalue distribution encodes
    #    the circuit's resonance structure and homogeneity.
    # ═══════════════════════════════════════════════════════════════

    try:
        # 58. Spectral radius ratio = λ_max / λ₂
        #     Large → many "resonance modes" between bottleneck and
        #     densest region.  Circuit has wide dynamic range.
        lambda_max_wt = float(eigs_wt[-1]) if len(eigs_wt) > 0 else 0.0
        features['spectral_radius_ratio_wt'] = (
            lambda_max_wt / lambda2_wt if lambda2_wt > 1e-12 else 0.0
        )

        # 59–60. Eigenvalue entropy (spectral homogeneity)
        eig_ent_wt = eigenvalue_entropy(eigs_wt)
        eig_ent_mut = eigenvalue_entropy(eigs_mut)
        features['eigenvalue_entropy_wt'] = eig_ent_wt
        features['delta_eigenvalue_entropy'] = eig_ent_mut - eig_ent_wt

        # 61. Effective graph resistance = Kf / (n*(n-1)/2)
        #     = average pairwise effective resistance
        n_pairs = n * (n - 1) / 2.0
        features['effective_graph_resistance_wt'] = (
            Kf_wt / n_pairs if n_pairs > 0 else 0.0
        )
    except (ValueError, IndexError, ZeroDivisionError):
        features['spectral_radius_ratio_wt'] = 0.0
        features['eigenvalue_entropy_wt'] = 0.0
        features['delta_eigenvalue_entropy'] = 0.0
        features['effective_graph_resistance_wt'] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # M. RESISTANCE DISTANCE GEOMETRY  (8)
    #    Higher-order statistics of the resistance distance
    #    distribution around the mutation site.
    #    Mean/std miss the *shape* — median, harmonic mean,
    #    skewness, and kurtosis capture it.
    # ═══════════════════════════════════════════════════════════════

    try:
        # 62–63. Harmonic mean of R (biased toward small R = near neighbours)
        R_hm_wt = resistance_row_harmonic_mean(R_wt, mutation_site)
        R_hm_mut = resistance_row_harmonic_mean(R_mut, mutation_site)
        features['R_harmonic_mean_wt'] = R_hm_wt
        features['delta_R_harmonic_mean'] = R_hm_mut - R_hm_wt

        # 64–65. Median R (robust central tendency)
        stats_wt = resistance_row_statistics(R_wt, mutation_site)
        stats_mut = resistance_row_statistics(R_mut, mutation_site)
        features['R_median_wt'] = stats_wt['median']
        features['delta_R_median'] = stats_mut['median'] - stats_wt['median']

        # 66–67. Skewness of R distribution (asymmetry)
        features['R_skewness_wt'] = stats_wt['skewness']
        features['delta_R_skewness'] = stats_mut['skewness'] - stats_wt['skewness']

        # 68–69. Kurtosis of R distribution (tail weight)
        features['R_kurtosis_wt'] = stats_wt['kurtosis']
        features['delta_R_kurtosis'] = stats_mut['kurtosis'] - stats_wt['kurtosis']
    except (ValueError, IndexError, ZeroDivisionError):
        features['R_harmonic_mean_wt'] = 0.0
        features['delta_R_harmonic_mean'] = 0.0
        features['R_median_wt'] = 0.0
        features['delta_R_median'] = 0.0
        features['R_skewness_wt'] = 0.0
        features['delta_R_skewness'] = 0.0
        features['R_kurtosis_wt'] = 0.0
        features['delta_R_kurtosis'] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # N. FOLD-CHANGE RATIOS  (5)
    #    Additive deltas miss multiplicative effects.
    #    log₂(mut/wt) is the natural scale for fold changes and
    #    is symmetric: doubling R gives +1, halving gives −1.
    # ═══════════════════════════════════════════════════════════════

    try:
        _eps = 1e-12  # avoid log(0) and division by zero

        # 70. R_active fold change
        R_active_mut_abs = float(np.mean([
            R_mut[mutation_site, a] for a in current_sinks
        ])) if current_sinks else _eps
        R_active_wt_val = features.get('R_active_site_wt', _eps)
        if R_active_wt_val < _eps:
            R_active_wt_val = _eps
        features['R_active_fold_change'] = float(
            np.log2(max(R_active_mut_abs, _eps) / R_active_wt_val)
        )

        # 71. Kirchhoff fold change
        features['kirchhoff_fold_change'] = float(
            np.log2(max(Kf_mut, _eps) / max(Kf_wt, _eps))
        )

        # 72. Conductance degree fold change
        G_deg_wt = features.get('conductance_degree_wt', _eps)
        G_deg_mut = float(sum(
            d['weight'] for _, _, d in G_mut.edges(mutation_site, data=True)
        ))
        features['conductance_degree_fold_change'] = float(
            np.log2(max(G_deg_mut, _eps) / max(G_deg_wt, _eps))
        )

        # 73. R global fold change
        R_global_wt_mean = float(np.mean(R_wt[mutation_site]))
        R_global_mut_mean = float(np.mean(R_mut[mutation_site]))
        features['R_global_fold_change'] = float(
            np.log2(max(R_global_mut_mean, _eps) / max(R_global_wt_mean, _eps))
        )

        # 74. Voltage (self-potential) fold change
        features['voltage_fold_change'] = float(
            np.log2(
                max(abs(Lp_mut[mutation_site, mutation_site]), _eps)
                / max(abs(Lp_wt[mutation_site, mutation_site]), _eps)
            )
        )
    except (ValueError, IndexError, ZeroDivisionError):
        features['R_active_fold_change'] = 0.0
        features['kirchhoff_fold_change'] = 0.0
        features['conductance_degree_fold_change'] = 0.0
        features['R_global_fold_change'] = 0.0
        features['voltage_fold_change'] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # O. MULTI-SCALE EXTENDED  (4)
    #    3-hop shell, resistance gradient, and local/global ratio.
    # ═══════════════════════════════════════════════════════════════

    try:
        # Reuse shells computed in section I (extend to 3 hops)
        shells_wt_3 = resistance_shells(R_wt, G_wt, mutation_site, max_hops=3)
        shells_mut_3 = resistance_shells(R_mut, G_mut, mutation_site, max_hops=3)

        # 75. R shell 3 mean (3-hop neighbours)
        features['R_shell_3_mean_wt'] = float(shells_wt_3.get(3, 0.0))

        # 76–77. Shell gradient = (R2 - R1) / R1 = rate of R increase
        _r1_wt = shells_wt_3.get(1, _eps)
        _r2_wt = shells_wt_3.get(2, 0.0)
        _r1_mut = shells_mut_3.get(1, _eps)
        _r2_mut = shells_mut_3.get(2, 0.0)
        grad_wt = (_r2_wt - _r1_wt) / max(_r1_wt, _eps)
        grad_mut = (_r2_mut - _r1_mut) / max(_r1_mut, _eps)
        features['shell_gradient_wt'] = float(grad_wt)
        features['delta_shell_gradient'] = float(grad_mut - grad_wt)

        # 78. Local/global resistance ratio = R_shell_1 / R_mean_all
        R_mean_all_wt = float(np.mean(R_wt[mutation_site]))
        features['R_local_global_ratio_wt'] = (
            float(_r1_wt / R_mean_all_wt) if R_mean_all_wt > _eps else 0.0
        )
    except (ValueError, IndexError, ZeroDivisionError, KeyError):
        features['R_shell_3_mean_wt'] = 0.0
        features['shell_gradient_wt'] = 0.0
        features['delta_shell_gradient'] = 0.0
        features['R_local_global_ratio_wt'] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # P. CROSS-FEATURE PHYSICAL PRODUCTS  (5)
    #    Dimensionless products of circuit quantities that have
    #    clear physical interpretation.  These capture non-linear
    #    interactions between features that a linear model would miss
    #    and help tree-based models find splits more easily.
    # ═══════════════════════════════════════════════════════════════

    try:
        # 79. Conductance × R_active (dimensionless coupling strength)
        #     High G_degree but high R_active → good local wiring
        #     but poor long-range coupling to active site.
        features['conductance_R_active_product'] = (
            features.get('conductance_degree_wt', 0.0)
            * features.get('R_active_site_wt', 0.0)
        )

        # 80. Fiedler² (importance of bottleneck position)
        features['fiedler_squared_wt'] = (
            features.get('fiedler_component_wt', 0.0) ** 2
        )

        # 81. Current × R product (effective power at active)
        features['current_R_product_wt'] = (
            features.get('max_edge_current_wt', 0.0)
            * features.get('R_active_site_wt', 0.0)
        )

        # 82. Voltage × conductance degree (Norton current proxy)
        features['voltage_conductance_product_wt'] = (
            features.get('voltage_wt', 0.0)
            * features.get('conductance_degree_wt', 0.0)
        )

        # 83. Power fraction × delta_R (disruption × bottleneck impact)
        features['power_deltaR_product'] = (
            features.get('power_local_fraction_wt', 0.0)
            * features.get('delta_R_active_site', 0.0)
        )
    except (ValueError, IndexError, ZeroDivisionError):
        features['conductance_R_active_product'] = 0.0
        features['fiedler_squared_wt'] = 0.0
        features['current_R_product_wt'] = 0.0
        features['voltage_conductance_product_wt'] = 0.0
        features['power_deltaR_product'] = 0.0

    return features


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _lambda_k(eigenvalues: np.ndarray, k: int) -> float:
    """Return the k-th smallest *non-zero* eigenvalue (1-indexed).

    λ₁ is the smallest non-zero eigenvalue (algebraic connectivity),
    λ₂ is the next, etc.
    """
    nonzero = eigenvalues[eigenvalues > 1e-10]
    if len(nonzero) >= k:
        return float(np.sort(nonzero)[k - 1])
    return 0.0


# ──────────────────────────────────────────────────────────────────
# Feature metadata
# ──────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # A – Resistance distance (6)
    'delta_R_active_site',
    'delta_R_global_mean',
    'delta_R_max',
    'delta_R_std',
    'R_eccentricity_wt',
    'delta_R_eccentricity',
    # B – Global circuit health (4)
    'delta_kirchhoff',
    'propagation_radius',
    'delta_algebraic_connectivity',
    'spectral_gap_ratio',
    # C – Current flow centrality (4)
    'current_flow_betweenness_wt',
    'delta_current_flow_betweenness',
    'resistance_centrality_wt',
    'delta_resistance_centrality',
    # D – Voltage / potential (4)
    'voltage_wt',
    'delta_voltage',
    'transfer_voltage_active_wt',
    'delta_transfer_voltage_active',
    # E – Spectral mode (3)
    'fiedler_component_wt',
    'delta_fiedler_component',
    'conductance_degree_wt',
    # F – Current flow patterns (12)
    'effective_conductance_to_active_wt',
    'delta_effective_conductance_to_active',
    'max_edge_current_wt',
    'delta_max_edge_current',
    'current_entropy_wt',
    'delta_current_entropy',
    'site_current_fraction_wt',
    'delta_site_current_fraction',
    'current_redistribution',
    'current_concentration_wt',
    'delta_current_concentration',
    'neighbor_current_asymmetry_wt',
    # G – Voltage transfer & coupling (5)
    'voltage_transfer_ratio_wt',
    'delta_voltage_transfer_ratio',
    'mutual_conductance_wt',
    'delta_mutual_conductance',
    'voltage_influence_radius_wt',
    # H – Power dissipation landscape (5)
    'power_local_fraction_wt',
    'delta_power_local_fraction',
    'power_entropy_wt',
    'delta_power_entropy',
    'max_power_edge_wt',
    # I – Multi-scale resistance & dynamics (5)
    'R_shell_1_mean_wt',
    'R_shell_2_mean_wt',
    'delta_R_shell_ratio',
    'commute_time_to_active_wt',
    'effective_path_count_wt',
    # J – Green's function propagator (5)
    'green_trace_fraction_wt',
    'green_row_entropy_wt',
    'delta_green_row_entropy',
    'green_anisotropy_wt',
    'delta_green_anisotropy',
    # K – Thevenin / Norton equivalent (4)
    'R_active_site_wt',
    'circuit_gain_wt',
    'delta_circuit_gain',
    'norton_current_wt',
    # L – Spectral circuit descriptors (4)
    'spectral_radius_ratio_wt',
    'eigenvalue_entropy_wt',
    'delta_eigenvalue_entropy',
    'effective_graph_resistance_wt',
    # M – Resistance distance geometry (8)
    'R_harmonic_mean_wt',
    'delta_R_harmonic_mean',
    'R_median_wt',
    'delta_R_median',
    'R_skewness_wt',
    'delta_R_skewness',
    'R_kurtosis_wt',
    'delta_R_kurtosis',
    # N – Fold-change ratios (5)
    'R_active_fold_change',
    'kirchhoff_fold_change',
    'conductance_degree_fold_change',
    'R_global_fold_change',
    'voltage_fold_change',
    # O – Multi-scale extended (4)
    'R_shell_3_mean_wt',
    'shell_gradient_wt',
    'delta_shell_gradient',
    'R_local_global_ratio_wt',
    # P – Cross-feature products (5)
    'conductance_R_active_product',
    'fiedler_squared_wt',
    'current_R_product_wt',
    'voltage_conductance_product_wt',
    'power_deltaR_product',
]

FEATURE_GROUPS = {
    'resistance_distance': FEATURE_NAMES[0:6],
    'global_circuit': FEATURE_NAMES[6:10],
    'current_flow_centrality': FEATURE_NAMES[10:14],
    'voltage_potential': FEATURE_NAMES[14:18],
    'spectral_mode': FEATURE_NAMES[18:21],
    'current_flow_patterns': FEATURE_NAMES[21:33],
    'voltage_transfer': FEATURE_NAMES[33:38],
    'power_dissipation': FEATURE_NAMES[38:43],
    'multiscale_dynamics': FEATURE_NAMES[43:48],
    'green_function': FEATURE_NAMES[48:53],
    'thevenin_norton': FEATURE_NAMES[53:57],
    'spectral_descriptors': FEATURE_NAMES[57:61],
    'resistance_geometry': FEATURE_NAMES[61:69],
    'fold_change': FEATURE_NAMES[69:74],
    'multiscale_extended': FEATURE_NAMES[74:78],
    'cross_products': FEATURE_NAMES[78:83],
}
