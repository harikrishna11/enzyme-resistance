"""
Enzyme Resistance — Electrical Resistance Model of Mutation Propagation in Enzymes.

This package models mutation propagation through protein structures using
circuit theory. Residues are nodes, non-covalent contacts are resistors,
and mutations perturb resistor values. Effective resistance captures
allosteric communication between residues.
"""

__version__ = "0.1.0"

from enzyme_resistance.contact_graph import build_contact_graph
from enzyme_resistance.resistance import effective_resistance_matrix
from enzyme_resistance.mutation import apply_mutation
from enzyme_resistance.features import extract_resistance_features, compute_wt_properties

__all__ = [
    "build_contact_graph",
    "effective_resistance_matrix",
    "apply_mutation",
    "extract_resistance_features",
    "compute_wt_properties",
]
