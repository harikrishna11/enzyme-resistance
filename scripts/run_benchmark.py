#!/usr/bin/env python3
"""
Run the full enzyme resistance benchmark.

This script:
1. Downloads all requested datasets once and caches PDB structures.
2. Builds 21-dimensional circuit-theoretic features for each mutation.
3. Trains our circuit model on each dataset independently.
4. Compares against 15 published ΔΔG prediction methods (each keeps its
   own CV protocol as reported by the authors).
5. Runs ablation and conductance sensitivity studies.
6. Generates comprehensive performance plots.

Usage:
    # Run all datasets (recommended):
    python scripts/run_benchmark.py --dataset all

    # Quick test:
    python scripts/run_benchmark.py --dataset builtin --max-mutations 30

    # Single dataset:
    python scripts/run_benchmark.py --dataset s2648
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

from enzyme_resistance.benchmark import run_benchmark


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run enzyme resistance benchmark')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['s2648', 'fireprotdb', 'builtin', 'all'],
                        help='Dataset: s2648, fireprotdb, builtin, or "all" '
                             'to run every dataset independently. Default: all')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='benchmark_results')
    parser.add_argument('--max-mutations', type=int, default=None)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--no-ablation', action='store_true')
    parser.add_argument('--no-sensitivity', action='store_true')
    args = parser.parse_args()

    run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        max_mutations=args.max_mutations,
        n_folds=args.folds,
        run_ablation=not args.no_ablation,
        run_sensitivity=not args.no_sensitivity,
    )


if __name__ == '__main__':
    main()
