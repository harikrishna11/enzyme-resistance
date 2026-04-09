# enzyme-resistance

Electrical resistance model of mutation propagation in enzymes: benchmarking, feature engineering, and CLI tools for analyzing stability-related mutations from structure.

## Install

```bash
pip install -e ".[dev]"
```

Optional Hugging Face dataset support:

```bash
pip install -e ".[hf]"
```

## CLI

After install, the `enzyme-resistance` command is available:

```bash
enzyme-resistance benchmark --help
enzyme-resistance analyze STRUCTURE.pdb --mutation POSITION:FROM:TO
```

Benchmarks download or use cached data under a configurable `--data-dir` (see command help). Large PDB caches and benchmark outputs are not committed to this repository (see `.gitignore`).

## Development

```bash
pytest
```

## License

MIT (see `pyproject.toml`).
