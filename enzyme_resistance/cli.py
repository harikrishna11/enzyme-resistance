"""
Command-line interface for enzyme-resistance package.

Usage:
    enzyme-resistance benchmark [--data-dir DIR] [--output-dir DIR] [--max-mutations N]
    enzyme-resistance analyze PDB_FILE --mutation SITE:FROM:TO [--chain CHAIN] [--cutoff CUTOFF]
"""

import argparse
import logging
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='enzyme-resistance',
        description='Electrical Resistance Model of Mutation Propagation in Enzymes',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose logging',
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ── benchmark command ──────────────────────────────────────────
    bench_parser = subparsers.add_parser(
        'benchmark',
        help='Run full benchmark: download data, build features, train models',
    )
    bench_parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['s2648', 'fireprotdb', 'builtin', 'all'],
        help='Dataset to benchmark on: s2648, fireprotdb, builtin, '
             'or "all" to run every dataset independently and compare. '
             'Default: all',
    )
    bench_parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Directory for cached data/PDB files',
    )
    bench_parser.add_argument(
        '--output-dir', type=str, default='benchmark_results',
        help='Directory for output plots and reports',
    )
    bench_parser.add_argument(
        '--max-mutations', type=int, default=None,
        help='Maximum number of mutations to process (for quick testing)',
    )
    bench_parser.add_argument(
        '--folds', type=int, default=5,
        help='Number of cross-validation folds',
    )
    bench_parser.add_argument(
        '--no-ablation', action='store_true',
        help='Skip feature ablation study',
    )
    bench_parser.add_argument(
        '--no-sensitivity', action='store_true',
        help='Skip conductance sensitivity study',
    )

    # ── analyze command ────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze a single mutation on a PDB structure',
    )
    analyze_parser.add_argument(
        'pdb_file', type=str,
        help='Path to PDB file',
    )
    analyze_parser.add_argument(
        '--mutation', type=str, required=True,
        help='Mutation in format POSITION:FROM:TO (e.g., 42:V:A)',
    )
    analyze_parser.add_argument(
        '--chain', type=str, default=None,
        help='Chain ID (default: all chains)',
    )
    analyze_parser.add_argument(
        '--cutoff', type=float, default=8.0,
        help='Contact distance cutoff in Angstroms',
    )
    analyze_parser.add_argument(
        '--conductance', type=str, default='exponential',
        choices=['exponential', 'inverse_square', 'binary'],
        help='Conductance weighting method',
    )
    analyze_parser.add_argument(
        '--active-site', type=str, default=None,
        help='Comma-separated active site residue numbers (e.g., 42,87,143)',
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    if args.command == 'benchmark':
        _run_benchmark(args)
    elif args.command == 'analyze':
        _run_analyze(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_benchmark(args):
    """Run the benchmark pipeline."""
    from enzyme_resistance.benchmark import run_benchmark

    results = run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        max_mutations=args.max_mutations,
        n_folds=args.folds,
        run_ablation=not args.no_ablation,
        run_sensitivity=not args.no_sensitivity,
    )


def _run_analyze(args):
    """Analyze a single mutation."""
    from enzyme_resistance.contact_graph import (
        build_contact_graph, get_residue_index, get_active_site_indices,
    )
    from enzyme_resistance.mutation import apply_mutation
    from enzyme_resistance.features import extract_resistance_features

    # Parse mutation string
    parts = args.mutation.split(':')
    if len(parts) != 3:
        print(f"Error: Mutation must be in format POSITION:FROM:TO (got: {args.mutation})")
        sys.exit(1)

    position = int(parts[0])
    aa_from = parts[1]
    aa_to = parts[2]

    print(f"Analyzing mutation: {aa_from}{position}{aa_to}")
    print(f"PDB: {args.pdb_file}")
    print(f"Cutoff: {args.cutoff} Å, Conductance: {args.conductance}")
    print()

    # Build graph
    G_wt, residues = build_contact_graph(
        args.pdb_file,
        cutoff=args.cutoff,
        conductance_method=args.conductance,
        chain_id=args.chain,
    )
    print(f"Contact graph: {G_wt.number_of_nodes()} nodes, {G_wt.number_of_edges()} edges")

    # Find mutation site
    mut_site = get_residue_index(residues, position, args.chain)
    print(f"Mutation site node index: {mut_site}")

    # Parse active site
    active_site = None
    if args.active_site:
        active_resids = [int(x.strip()) for x in args.active_site.split(',')]
        active_site = get_active_site_indices(residues, active_resids, args.chain)

    # Apply mutation
    G_mut = apply_mutation(G_wt, mut_site, aa_from, aa_to)

    # Extract features
    features = extract_resistance_features(G_wt, G_mut, mut_site, active_site)

    print()
    print("Resistance Features:")
    print("-" * 50)
    for name, value in features.items():
        print(f"  {name:>25s}: {value:+.6f}")
    print("-" * 50)


if __name__ == '__main__':
    main()
