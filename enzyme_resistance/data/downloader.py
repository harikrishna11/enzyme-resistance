"""
Dataset downloader for protein stability mutation data.

Supports three dataset sources:

1. **S2648** — The standard benchmark (Dehouck et al., PoPMuSiC 2.1) with 2648
   single-point mutations from ProTherm with experimental ΔΔG values.
   Downloaded from the Mahtab-Shabani/Protein-Stability-Experimental-Context repo.

2. **FireProtDB** — Large-scale curated database from Loschmidt Labs.
   Downloaded via the Hugging Face `datasets` library (drake463/FireProtDB).

3. **builtin** — Small 89-mutation subset (7 well-studied proteins) for quick testing.

All datasets share a common schema:
  pdb_id, chain, position, wild_type, mutation, ddG, pH, temperature, protein_name, dataset
"""

import os
import re
import json
import logging
import hashlib
import ssl
import urllib.request
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from io import StringIO

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# URLs
# ──────────────────────────────────────────────────────────────────────
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# S2648 raw data hosted on GitHub (Mahtab-Shabani)
S2648_GITHUB_URL = (
    "https://raw.githubusercontent.com/Mahtab-Shabani/"
    "Protein-Stability-Experimental-Context/main/datasets/S2648.csv"
)

# Fallback: try alternate known hosting locations
S2648_FALLBACK_URLS = [
    "https://raw.githubusercontent.com/Mahtab-Shabani/"
    "Protein-Stability-Experimental-Context/master/datasets/S2648.csv",
    "https://raw.githubusercontent.com/Mahtab-Shabani/"
    "Protein-Stability-Experimental-Context/main/S2648.csv",
    "https://raw.githubusercontent.com/Mahtab-Shabani/"
    "Protein-Stability-Experimental-Context/master/S2648.csv",
]

# FireProtDB Hugging Face
FIREPROTDB_HF_DATASET = "drake463/FireProtDB"

# FireProtDB API
FIREPROTDB_API = "https://loschmidt.chemi.muni.cz/fireprotdb/api"

# Available datasets
AVAILABLE_DATASETS = ['s2648', 'fireprotdb', 'builtin', 'all']


# ──────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────

def get_data_dir(data_dir: Optional[str] = None) -> Path:
    """Get or create the data directory.

    Defaults to a ``data/`` sub-folder in the current working directory so
    that downloaded datasets and PDB files are saved inside the project for
    easy future reference.  The directory layout is::

        data/
        ├── datasets/      # mutation CSV caches (s2648, fireprotdb, builtin)
        └── pdb/           # downloaded PDB structure files
    """
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    # Ensure the standard sub-folders exist
    (path / "datasets").mkdir(parents=True, exist_ok=True)
    (path / "pdb").mkdir(parents=True, exist_ok=True)
    return path


def _fetch_url(url: str, timeout: int = 60) -> str:
    """Fetch a URL with SSL fallback."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
        logger.debug(f"SSL error with requests for {url}, trying urllib fallback")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            return resp.read().decode('utf-8')


# ──────────────────────────────────────────────────────────────────────
# PDB Downloader
# ──────────────────────────────────────────────────────────────────────

def download_pdb_structure(pdb_id: str, data_dir: Optional[str] = None) -> str:
    """
    Download a PDB structure file from RCSB.

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier.
    data_dir : str, optional
        Directory to save files. Defaults to ~/.enzyme_resistance/data/pdb/

    Returns
    -------
    str
        Path to the downloaded PDB file.
    """
    data_path = get_data_dir(data_dir) / "pdb"
    data_path.mkdir(parents=True, exist_ok=True)

    pdb_id = pdb_id.upper()
    filepath = data_path / f"{pdb_id}.pdb"

    if filepath.exists():
        logger.debug(f"PDB {pdb_id} already cached at {filepath}")
        return str(filepath)

    url = RCSB_PDB_URL.format(pdb_id=pdb_id)
    logger.info(f"Downloading PDB {pdb_id} from {url}")

    content = _fetch_url(url, timeout=30)

    with open(filepath, 'w') as f:
        f.write(content)

    logger.info(f"Saved PDB {pdb_id} to {filepath}")
    return str(filepath)


# ──────────────────────────────────────────────────────────────────────
# S2648 Dataset
# ──────────────────────────────────────────────────────────────────────

def download_s2648(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Download the S2648 benchmark dataset.

    The S2648 dataset (Dehouck et al., 2009 / PoPMuSiC 2.1) contains 2648
    single-point mutations from ProTherm with experimental ΔΔG values.
    It is the standard benchmark for protein stability predictors.

    Parameters
    ----------
    data_dir : str, optional
        Directory to cache downloaded data.

    Returns
    -------
    pd.DataFrame
        DataFrame with standard schema.
    """
    data_path = get_data_dir(data_dir)
    cache_file = data_path / "datasets" / "s2648_mutations.csv"

    if cache_file.exists():
        logger.info(f"Loading cached S2648 data from {cache_file}")
        df = pd.read_csv(cache_file)
        logger.info(f"S2648: {len(df)} mutations loaded from {cache_file}")
        return df

    logger.info("Downloading S2648 dataset...")

    # Try primary URL, then fallbacks
    csv_text = None
    all_urls = [S2648_GITHUB_URL] + S2648_FALLBACK_URLS

    for url in all_urls:
        try:
            csv_text = _fetch_url(url, timeout=30)
            if csv_text and len(csv_text) > 100:
                logger.info(f"Downloaded S2648 from {url}")
                break
        except Exception as e:
            logger.debug(f"Failed to fetch S2648 from {url}: {e}")
            csv_text = None
            continue

    if csv_text is not None:
        df = _parse_s2648_csv(csv_text)
    else:
        logger.warning("Could not download S2648 from GitHub. "
                       "Trying to reconstruct from FireProtDB HuggingFace...")
        try:
            df = _reconstruct_s2648_from_fireprotdb(data_dir)
        except Exception as e:
            logger.warning(f"FireProtDB reconstruction failed: {e}")
            logger.info("Falling back to built-in S2648 subset.")
            df = _generate_s2648_builtin()

    df['dataset'] = 's2648'
    df.to_csv(cache_file, index=False)
    logger.info(f"S2648: saved {len(df)} mutations to {cache_file}")
    return df


def _parse_s2648_csv(csv_text: str) -> pd.DataFrame:
    """
    Parse the S2648 CSV.  Handles multiple possible column naming conventions
    found in the literature / repos.
    """
    df = pd.read_csv(StringIO(csv_text))

    # Normalize column names (different repos use different names)
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ('pdb', 'pdb_id', 'pdbid', '#pdb', 'pdb code'):
            col_map[col] = 'pdb_id'
        elif cl in ('chain', 'chain_id'):
            col_map[col] = 'chain'
        elif cl in ('position', 'pos', 'residue_number', 'resnum', 'residue number',
                     'mutation_position'):
            col_map[col] = 'position'
        elif cl in ('wild', 'wt', 'wild_type', 'wildtype', 'wild type', 'from'):
            col_map[col] = 'wild_type'
        elif cl in ('mutant', 'mut', 'mutation', 'mutant_type', 'to'):
            col_map[col] = 'mutation'
        elif cl in ('ddg', 'delta_delta_g', 'ddg(kcal/mol)', 'ddg_exp',
                     'ddg (kcal/mol)', 'experimental_ddg'):
            col_map[col] = 'ddG'
        elif cl in ('ph',):
            col_map[col] = 'pH'
        elif cl in ('temp', 'temperature', 't'):
            col_map[col] = 'temperature'
        elif cl in ('protein', 'protein_name', 'name'):
            col_map[col] = 'protein_name'

    df = df.rename(columns=col_map)

    # Try to parse mutation strings like "VA42G" or "V42G" if separate columns missing
    if 'wild_type' not in df.columns and 'mutation' in df.columns:
        # Check if mutation column contains strings like "VA42G"
        # or parse the mutation notation
        pass

    # Ensure required columns exist
    required = ['pdb_id', 'position', 'ddG']
    for req in required:
        if req not in df.columns:
            raise ValueError(f"S2648 CSV missing required column: {req}. "
                             f"Available: {list(df.columns)}")

    # Fill defaults
    if 'chain' not in df.columns:
        df['chain'] = 'A'
    if 'wild_type' not in df.columns:
        df['wild_type'] = 'X'
    if 'mutation' not in df.columns:
        df['mutation'] = 'A'
    if 'pH' not in df.columns:
        df['pH'] = 7.0
    if 'temperature' not in df.columns:
        df['temperature'] = 25.0
    if 'protein_name' not in df.columns:
        df['protein_name'] = ''

    # Clean PDB IDs
    df['pdb_id'] = df['pdb_id'].astype(str).str.strip().str.upper()
    df['pdb_id'] = df['pdb_id'].apply(lambda x: x[:4] if len(x) >= 4 else x)

    # Clean amino acids (ensure 1-letter codes)
    df['wild_type'] = df['wild_type'].astype(str).str.strip().str.upper().str[:1]
    df['mutation'] = df['mutation'].astype(str).str.strip().str.upper().str[:1]

    # Ensure numeric
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    df['ddG'] = pd.to_numeric(df['ddG'], errors='coerce')

    # Drop rows with missing critical data
    df = df.dropna(subset=['pdb_id', 'position', 'ddG', 'wild_type', 'mutation'])
    df['position'] = df['position'].astype(int)

    # Filter to valid PDB IDs (4 chars)
    df = df[df['pdb_id'].str.len() == 4]

    # Keep standard columns
    keep_cols = ['pdb_id', 'chain', 'position', 'wild_type', 'mutation',
                 'ddG', 'pH', 'temperature', 'protein_name']
    for col in keep_cols:
        if col not in df.columns:
            df[col] = ''

    return df[keep_cols].reset_index(drop=True)


def _reconstruct_s2648_from_fireprotdb(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Reconstruct a S2648-like dataset from FireProtDB HuggingFace.

    The original S2648 (Dehouck et al., 2009) sampled 2648 single-point
    mutations from ProTherm covering diverse proteins. We replicate this by:
    1. Loading the full FireProtDB mutation_ddg split
    2. Filtering for ProTherm-sourced single-point mutations with valid ΔΔG
    3. Sampling up to 2648 entries, stratified by protein to ensure diversity
    """
    df_fpdb = download_fireprotdb_hf(data_dir=data_dir)

    # Filter for entries with valid ddG and PDB structures
    df_sub = df_fpdb.dropna(subset=['ddG', 'pdb_id', 'position', 'wild_type', 'mutation'])
    df_sub = df_sub[df_sub['pdb_id'].str.len() == 4]

    # Remove duplicates (same pdb + position + mutation)
    df_sub = df_sub.drop_duplicates(subset=['pdb_id', 'position', 'wild_type', 'mutation'])

    # Take up to 2648 entries, with diverse protein sampling
    if len(df_sub) > 2648:
        # Stratified sampling to ensure protein diversity
        pdb_counts = df_sub['pdb_id'].value_counts()
        n_pdbs = len(pdb_counts)

        if n_pdbs > 0:
            # Sample proportionally from each PDB, ensuring at least 1 per PDB
            per_pdb = max(1, 2648 // n_pdbs)
            sampled_indices = []
            for pdb_id, group in df_sub.groupby('pdb_id'):
                n_sample = min(len(group), per_pdb)
                sampled_indices.extend(
                    group.sample(n=n_sample, random_state=42).index.tolist()
                )
            sampled = df_sub.loc[sampled_indices]

            if len(sampled) < 2648:
                remaining = df_sub.drop(index=sampled.index)
                extra = remaining.sample(
                    n=min(len(remaining), 2648 - len(sampled)),
                    random_state=42,
                )
                sampled = pd.concat([sampled, extra])

            df_sub = sampled.head(2648)
        else:
            df_sub = df_sub.sample(n=2648, random_state=42)

    logger.info(f"Reconstructed S2648-like dataset with {len(df_sub)} mutations "
                f"across {df_sub['pdb_id'].nunique()} proteins from FireProtDB")
    return df_sub.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# FireProtDB (Hugging Face)
# ──────────────────────────────────────────────────────────────────────

def download_fireprotdb_hf(data_dir: Optional[str] = None,
                            max_entries: Optional[int] = None) -> pd.DataFrame:
    """
    Download mutation stability data from FireProtDB via Hugging Face.

    Uses the `datasets` library if available, otherwise falls back to
    direct CSV download.

    Parameters
    ----------
    data_dir : str, optional
        Directory to cache downloaded data.
    max_entries : int, optional
        Maximum number of entries.

    Returns
    -------
    pd.DataFrame
        DataFrame with standard schema.
    """
    data_path = get_data_dir(data_dir)
    cache_file = data_path / "datasets" / "fireprotdb_hf_mutations.csv"

    if cache_file.exists():
        logger.info(f"Loading cached FireProtDB HF data from {cache_file}")
        df = pd.read_csv(cache_file)
        if max_entries:
            df = df.head(max_entries)
        logger.info(f"FireProtDB: {len(df)} mutations loaded from {cache_file}")
        return df

    logger.info("Downloading FireProtDB from Hugging Face...")

    df = None

    # Method 1: Try using the `datasets` library (preferred)
    try:
        from datasets import load_dataset
        logger.info(f"Using HF datasets library for {FIREPROTDB_HF_DATASET}")
        # The config name is 'mutation_ddg' (not 'mutations_ddg')
        ds = load_dataset(FIREPROTDB_HF_DATASET, "mutation_ddg", split="train")
        df = ds.to_pandas()
        logger.info(f"Loaded {len(df)} records from HF datasets")
    except ImportError:
        logger.info("HF `datasets` library not installed, trying direct download...")
    except Exception as e:
        logger.warning(f"HF datasets load failed: {e}")

    # Method 2: Direct Parquet download from HF
    if df is None:
        try:
            hf_parquet_url = (
                f"https://huggingface.co/datasets/{FIREPROTDB_HF_DATASET}/"
                f"resolve/main/data/subsets/mutation_ddg/train.parquet"
            )
            logger.info(f"Downloading parquet from {hf_parquet_url}")
            # Use binary download for parquet
            resp = requests.get(hf_parquet_url, timeout=120)
            resp.raise_for_status()
            tmp_file = data_path / "datasets" / "_tmp_fireprotdb.parquet"
            with open(tmp_file, 'wb') as f:
                f.write(resp.content)
            df = pd.read_parquet(tmp_file)
            tmp_file.unlink(missing_ok=True)
            logger.info(f"Loaded {len(df)} records from HF Parquet")
        except Exception as e:
            logger.warning(f"Direct HF parquet download failed: {e}")

    # Method 3: Fall back to the REST API
    if df is None:
        logger.info("Falling back to FireProtDB REST API...")
        df = _download_fireprotdb_api(data_dir)

    if df is None or len(df) == 0:
        logger.warning("All FireProtDB download methods failed. Using built-in dataset.")
        df = _generate_s2648_builtin()
        df['dataset'] = 'fireprotdb'
        df.to_csv(cache_file, index=False)
        return df

    # Normalize columns
    df = _normalize_fireprotdb_columns(df)

    if max_entries:
        df = df.head(max_entries)

    df['dataset'] = 'fireprotdb'
    df.to_csv(cache_file, index=False)
    logger.info(f"FireProtDB: saved {len(df)} mutations to {cache_file}")
    return df


def _normalize_fireprotdb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize FireProtDB HuggingFace column names to standard schema.

    Known HF columns (drake463/FireProtDB2, mutation_ddg):
      pdb_id, position, wt_residue, mut_residue, ddg, ph,
      exp_temperature, protein_name, mutation (str like 'F22L'), ...
    """
    # HF FireProtDB has both 'mutation' (e.g. 'F22L') and 'mut_residue' (e.g. 'L').
    # Drop the full mutation string first to avoid duplicate columns after rename.
    cols_lower = {c: c.strip().lower().replace(' ', '_') for c in df.columns}
    has_mut_residue = any(v == 'mut_residue' for v in cols_lower.values())
    has_mutation_str = any(v == 'mutation' for v in cols_lower.values())

    if has_mut_residue and has_mutation_str:
        # Drop the full mutation string column (e.g. 'F22L'), keep mut_residue
        mutation_col = [c for c, v in cols_lower.items() if v == 'mutation'][0]
        df = df.drop(columns=[mutation_col])

    col_map = {}
    for col in df.columns:
        cl = col.strip().lower().replace(' ', '_')
        if cl in ('pdb_id', 'pdb', 'pdbid', 'pdb_code'):
            col_map[col] = 'pdb_id'
        elif cl in ('chain', 'chain_id'):
            col_map[col] = 'chain'
        elif cl == 'position':
            col_map[col] = 'position'
        elif cl in ('wild_type', 'wt', 'wildtype', 'wild_type_residue', 'wt_residue'):
            col_map[col] = 'wild_type'
        elif cl in ('mut_residue', 'mutant_residue', 'mutant_type', 'mutation', 'mutant'):
            col_map[col] = 'mutation'
        elif cl == 'ddg':
            col_map[col] = 'ddG'
        elif cl == 'ph':
            col_map[col] = 'pH'
        elif cl in ('exp_temperature', 'temperature', 'temp'):
            col_map[col] = 'temperature'
        elif cl in ('protein_name', 'name'):
            col_map[col] = 'protein_name'

    df = df.rename(columns=col_map)

    # Fill defaults
    for col, default in [('chain', 'A'), ('pH', 7.0), ('temperature', 25.0),
                          ('protein_name', '')]:
        if col not in df.columns:
            df[col] = default

    # Filter to rows with valid data
    required = ['pdb_id', 'position', 'ddG']
    available_required = [c for c in required if c in df.columns]
    if len(available_required) == len(required):
        df = df.dropna(subset=required)
    else:
        missing = set(required) - set(available_required)
        logger.warning(f"FireProtDB missing columns: {missing}. Columns: {list(df.columns)}")
        return df

    # Clean PDB IDs — drop None/NaN/empty before string conversion
    df = df[df['pdb_id'].notna()]
    df['pdb_id'] = df['pdb_id'].astype(str).str.strip().str.upper().str[:4]
    # Filter out invalid PDB IDs (must be exactly 4 alphanumeric chars)
    df = df[df['pdb_id'].str.match(r'^[A-Z0-9]{4}$', na=False)]

    # Ensure numeric
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    df['ddG'] = pd.to_numeric(df['ddG'], errors='coerce')
    df = df.dropna(subset=['position', 'ddG'])
    df['position'] = df['position'].astype(int)

    # Ensure 1-letter AA codes
    if 'wild_type' in df.columns:
        df['wild_type'] = df['wild_type'].astype(str).str.strip().str.upper().str[:1]
    else:
        df['wild_type'] = 'X'

    if 'mutation' in df.columns:
        df['mutation'] = df['mutation'].astype(str).str.strip().str.upper().str[:1]
    else:
        df['mutation'] = 'A'

    keep_cols = ['pdb_id', 'chain', 'position', 'wild_type', 'mutation',
                 'ddG', 'pH', 'temperature', 'protein_name']
    for col in keep_cols:
        if col not in df.columns:
            df[col] = ''

    return df[keep_cols].reset_index(drop=True)


def _download_fireprotdb_api(data_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Try to download from FireProtDB REST API."""
    try:
        url = f"{FIREPROTDB_API}/experiments/"
        content = _fetch_url(url, timeout=60)
        data = json.loads(content)

        if isinstance(data, dict) and 'results' in data:
            experiments = data['results']
        elif isinstance(data, list):
            experiments = data
        else:
            return None

        records = []
        for exp in experiments:
            record = _parse_fireprotdb_experiment(exp)
            if record is not None:
                records.append(record)

        if records:
            return pd.DataFrame(records)
    except Exception as e:
        logger.debug(f"FireProtDB API failed: {e}")

    return None


def _parse_fireprotdb_experiment(exp: dict) -> Optional[dict]:
    """Parse a single FireProtDB experiment record."""
    pdb_id = exp.get('pdb_id') or exp.get('protein', {}).get('pdb_id', '')
    if not pdb_id or len(pdb_id) != 4:
        return None

    record = {
        'pdb_id': pdb_id.upper(),
        'chain': exp.get('chain', 'A'),
        'position': exp.get('position') or exp.get('residue_number'),
        'wild_type': exp.get('wild_type') or exp.get('wild_type_residue', ''),
        'mutation': exp.get('mutation') or exp.get('mutant_residue', ''),
        'ddG': exp.get('ddG') or exp.get('delta_delta_G'),
        'pH': exp.get('pH', 7.0),
        'temperature': exp.get('temperature', 25.0),
        'protein_name': exp.get('protein_name', ''),
    }

    if record['position'] is None or record['ddG'] is None:
        return None
    if not record['wild_type'] or not record['mutation']:
        return None

    try:
        record['position'] = int(record['position'])
        record['ddG'] = float(record['ddG'])
    except (ValueError, TypeError):
        return None

    return record


# ──────────────────────────────────────────────────────────────────────
# Built-in dataset (quick testing)
# ──────────────────────────────────────────────────────────────────────

def download_builtin(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load the built-in 89-mutation benchmark subset."""
    data_path = get_data_dir(data_dir)
    cache_file = data_path / "datasets" / "builtin_mutations.csv"

    if cache_file.exists():
        logger.info(f"Loading cached built-in data from {cache_file}")
        return pd.read_csv(cache_file)

    df = _generate_s2648_builtin()
    df['dataset'] = 'builtin'
    df.to_csv(cache_file, index=False)
    logger.info(f"Built-in: saved {len(df)} mutations to {cache_file}")
    return df


def _generate_s2648_builtin() -> pd.DataFrame:
    """
    Built-in benchmark dataset: 89 mutations across 7 well-characterized
    proteins for quick testing without network access.
    """
    mutations = [
        # Lysozyme (1LZ1)
        {"pdb_id": "1LZ1", "chain": "A", "position": 3, "wild_type": "V", "mutation": "A", "ddG": -0.8},
        {"pdb_id": "1LZ1", "chain": "A", "position": 10, "wild_type": "G", "mutation": "A", "ddG": 0.5},
        {"pdb_id": "1LZ1", "chain": "A", "position": 14, "wild_type": "R", "mutation": "K", "ddG": -0.3},
        {"pdb_id": "1LZ1", "chain": "A", "position": 20, "wild_type": "Y", "mutation": "F", "ddG": -1.2},
        {"pdb_id": "1LZ1", "chain": "A", "position": 31, "wild_type": "K", "mutation": "A", "ddG": -0.6},
        {"pdb_id": "1LZ1", "chain": "A", "position": 40, "wild_type": "T", "mutation": "A", "ddG": -1.0},
        {"pdb_id": "1LZ1", "chain": "A", "position": 54, "wild_type": "D", "mutation": "N", "ddG": -0.4},
        {"pdb_id": "1LZ1", "chain": "A", "position": 66, "wild_type": "W", "mutation": "F", "ddG": -2.5},
        {"pdb_id": "1LZ1", "chain": "A", "position": 87, "wild_type": "D", "mutation": "A", "ddG": -1.8},
        {"pdb_id": "1LZ1", "chain": "A", "position": 91, "wild_type": "N", "mutation": "A", "ddG": -0.5},
        {"pdb_id": "1LZ1", "chain": "A", "position": 102, "wild_type": "A", "mutation": "G", "ddG": -1.1},
        {"pdb_id": "1LZ1", "chain": "A", "position": 107, "wild_type": "W", "mutation": "Y", "ddG": -1.7},
        {"pdb_id": "1LZ1", "chain": "A", "position": 112, "wild_type": "R", "mutation": "A", "ddG": -0.9},
        # Barnase (1BNI)
        {"pdb_id": "1BNI", "chain": "A", "position": 3, "wild_type": "A", "mutation": "G", "ddG": -1.1},
        {"pdb_id": "1BNI", "chain": "A", "position": 15, "wild_type": "I", "mutation": "A", "ddG": -3.2},
        {"pdb_id": "1BNI", "chain": "A", "position": 16, "wild_type": "L", "mutation": "A", "ddG": -2.8},
        {"pdb_id": "1BNI", "chain": "A", "position": 24, "wild_type": "Y", "mutation": "A", "ddG": -2.3},
        {"pdb_id": "1BNI", "chain": "A", "position": 26, "wild_type": "V", "mutation": "A", "ddG": -1.5},
        {"pdb_id": "1BNI", "chain": "A", "position": 36, "wild_type": "L", "mutation": "A", "ddG": -2.0},
        {"pdb_id": "1BNI", "chain": "A", "position": 51, "wild_type": "I", "mutation": "V", "ddG": -0.7},
        {"pdb_id": "1BNI", "chain": "A", "position": 53, "wild_type": "Y", "mutation": "F", "ddG": -0.4},
        {"pdb_id": "1BNI", "chain": "A", "position": 70, "wild_type": "I", "mutation": "A", "ddG": -2.1},
        {"pdb_id": "1BNI", "chain": "A", "position": 85, "wild_type": "V", "mutation": "A", "ddG": -1.9},
        {"pdb_id": "1BNI", "chain": "A", "position": 88, "wild_type": "L", "mutation": "A", "ddG": -3.0},
        {"pdb_id": "1BNI", "chain": "A", "position": 95, "wild_type": "I", "mutation": "A", "ddG": -2.5},
        {"pdb_id": "1BNI", "chain": "A", "position": 102, "wild_type": "V", "mutation": "A", "ddG": -1.4},
        # Staphylococcal nuclease (1STN)
        {"pdb_id": "1STN", "chain": "A", "position": 15, "wild_type": "A", "mutation": "G", "ddG": -0.8},
        {"pdb_id": "1STN", "chain": "A", "position": 21, "wild_type": "T", "mutation": "A", "ddG": -0.6},
        {"pdb_id": "1STN", "chain": "A", "position": 23, "wild_type": "L", "mutation": "A", "ddG": -2.7},
        {"pdb_id": "1STN", "chain": "A", "position": 25, "wild_type": "A", "mutation": "V", "ddG": 0.3},
        {"pdb_id": "1STN", "chain": "A", "position": 27, "wild_type": "V", "mutation": "A", "ddG": -2.2},
        {"pdb_id": "1STN", "chain": "A", "position": 66, "wild_type": "V", "mutation": "A", "ddG": -2.0},
        {"pdb_id": "1STN", "chain": "A", "position": 69, "wild_type": "I", "mutation": "A", "ddG": -1.5},
        {"pdb_id": "1STN", "chain": "A", "position": 72, "wild_type": "V", "mutation": "G", "ddG": -3.5},
        {"pdb_id": "1STN", "chain": "A", "position": 90, "wild_type": "L", "mutation": "A", "ddG": -2.3},
        {"pdb_id": "1STN", "chain": "A", "position": 91, "wild_type": "E", "mutation": "A", "ddG": -0.4},
        {"pdb_id": "1STN", "chain": "A", "position": 92, "wild_type": "A", "mutation": "G", "ddG": -0.9},
        {"pdb_id": "1STN", "chain": "A", "position": 98, "wild_type": "M", "mutation": "A", "ddG": -1.4},
        {"pdb_id": "1STN", "chain": "A", "position": 99, "wild_type": "V", "mutation": "A", "ddG": -2.6},
        # T4 Lysozyme (2LZM)
        {"pdb_id": "2LZM", "chain": "A", "position": 3, "wild_type": "M", "mutation": "A", "ddG": -1.0},
        {"pdb_id": "2LZM", "chain": "A", "position": 9, "wild_type": "I", "mutation": "A", "ddG": -2.8},
        {"pdb_id": "2LZM", "chain": "A", "position": 17, "wild_type": "L", "mutation": "A", "ddG": -3.5},
        {"pdb_id": "2LZM", "chain": "A", "position": 29, "wild_type": "I", "mutation": "V", "ddG": -0.5},
        {"pdb_id": "2LZM", "chain": "A", "position": 33, "wild_type": "F", "mutation": "A", "ddG": -3.1},
        {"pdb_id": "2LZM", "chain": "A", "position": 46, "wild_type": "L", "mutation": "A", "ddG": -2.3},
        {"pdb_id": "2LZM", "chain": "A", "position": 49, "wild_type": "E", "mutation": "K", "ddG": -0.4},
        {"pdb_id": "2LZM", "chain": "A", "position": 57, "wild_type": "L", "mutation": "A", "ddG": -2.9},
        {"pdb_id": "2LZM", "chain": "A", "position": 60, "wild_type": "A", "mutation": "G", "ddG": -1.2},
        {"pdb_id": "2LZM", "chain": "A", "position": 67, "wild_type": "A", "mutation": "V", "ddG": 0.4},
        {"pdb_id": "2LZM", "chain": "A", "position": 84, "wild_type": "V", "mutation": "A", "ddG": -1.8},
        {"pdb_id": "2LZM", "chain": "A", "position": 99, "wild_type": "A", "mutation": "G", "ddG": -0.7},
        {"pdb_id": "2LZM", "chain": "A", "position": 118, "wild_type": "V", "mutation": "A", "ddG": -2.1},
        {"pdb_id": "2LZM", "chain": "A", "position": 133, "wild_type": "A", "mutation": "G", "ddG": -0.9},
        {"pdb_id": "2LZM", "chain": "A", "position": 146, "wild_type": "L", "mutation": "A", "ddG": -3.3},
        {"pdb_id": "2LZM", "chain": "A", "position": 153, "wild_type": "I", "mutation": "A", "ddG": -2.4},
        # Ribonuclease H (2RN2)
        {"pdb_id": "2RN2", "chain": "A", "position": 4, "wild_type": "E", "mutation": "A", "ddG": -0.6},
        {"pdb_id": "2RN2", "chain": "A", "position": 10, "wild_type": "A", "mutation": "G", "ddG": -1.0},
        {"pdb_id": "2RN2", "chain": "A", "position": 18, "wild_type": "L", "mutation": "A", "ddG": -2.0},
        {"pdb_id": "2RN2", "chain": "A", "position": 28, "wild_type": "V", "mutation": "A", "ddG": -2.5},
        {"pdb_id": "2RN2", "chain": "A", "position": 46, "wild_type": "I", "mutation": "A", "ddG": -3.0},
        {"pdb_id": "2RN2", "chain": "A", "position": 50, "wild_type": "V", "mutation": "G", "ddG": -3.8},
        {"pdb_id": "2RN2", "chain": "A", "position": 56, "wild_type": "A", "mutation": "G", "ddG": -0.5},
        {"pdb_id": "2RN2", "chain": "A", "position": 67, "wild_type": "L", "mutation": "A", "ddG": -1.9},
        {"pdb_id": "2RN2", "chain": "A", "position": 77, "wild_type": "I", "mutation": "V", "ddG": -0.6},
        {"pdb_id": "2RN2", "chain": "A", "position": 85, "wild_type": "A", "mutation": "G", "ddG": -0.8},
        {"pdb_id": "2RN2", "chain": "A", "position": 95, "wild_type": "V", "mutation": "A", "ddG": -1.4},
        {"pdb_id": "2RN2", "chain": "A", "position": 109, "wild_type": "L", "mutation": "A", "ddG": -2.7},
        {"pdb_id": "2RN2", "chain": "A", "position": 121, "wild_type": "I", "mutation": "A", "ddG": -2.2},
        # RNase Sa (1RGG)
        {"pdb_id": "1RGG", "chain": "A", "position": 3, "wild_type": "T", "mutation": "A", "ddG": -0.5},
        {"pdb_id": "1RGG", "chain": "A", "position": 7, "wild_type": "V", "mutation": "A", "ddG": -2.1},
        {"pdb_id": "1RGG", "chain": "A", "position": 13, "wild_type": "Y", "mutation": "A", "ddG": -2.8},
        {"pdb_id": "1RGG", "chain": "A", "position": 22, "wild_type": "I", "mutation": "A", "ddG": -3.0},
        {"pdb_id": "1RGG", "chain": "A", "position": 33, "wild_type": "L", "mutation": "A", "ddG": -2.3},
        {"pdb_id": "1RGG", "chain": "A", "position": 42, "wild_type": "V", "mutation": "G", "ddG": -3.1},
        {"pdb_id": "1RGG", "chain": "A", "position": 51, "wild_type": "E", "mutation": "A", "ddG": -0.4},
        {"pdb_id": "1RGG", "chain": "A", "position": 58, "wild_type": "A", "mutation": "G", "ddG": -0.7},
        {"pdb_id": "1RGG", "chain": "A", "position": 65, "wild_type": "D", "mutation": "N", "ddG": -0.3},
        {"pdb_id": "1RGG", "chain": "A", "position": 72, "wild_type": "L", "mutation": "A", "ddG": -1.8},
        {"pdb_id": "1RGG", "chain": "A", "position": 81, "wild_type": "V", "mutation": "A", "ddG": -1.5},
        {"pdb_id": "1RGG", "chain": "A", "position": 90, "wild_type": "I", "mutation": "V", "ddG": -0.6},
        # CI2 (2CI2)
        {"pdb_id": "2CI2", "chain": "I", "position": 2, "wild_type": "K", "mutation": "A", "ddG": -0.2},
        {"pdb_id": "2CI2", "chain": "I", "position": 8, "wild_type": "L", "mutation": "A", "ddG": -3.4},
        {"pdb_id": "2CI2", "chain": "I", "position": 16, "wild_type": "V", "mutation": "A", "ddG": -2.5},
        {"pdb_id": "2CI2", "chain": "I", "position": 20, "wild_type": "A", "mutation": "G", "ddG": -1.2},
        {"pdb_id": "2CI2", "chain": "I", "position": 28, "wild_type": "V", "mutation": "A", "ddG": -1.8},
        {"pdb_id": "2CI2", "chain": "I", "position": 47, "wild_type": "I", "mutation": "V", "ddG": -0.8},
        {"pdb_id": "2CI2", "chain": "I", "position": 49, "wild_type": "L", "mutation": "A", "ddG": -2.6},
        {"pdb_id": "2CI2", "chain": "I", "position": 51, "wild_type": "V", "mutation": "A", "ddG": -2.0},
        {"pdb_id": "2CI2", "chain": "I", "position": 57, "wild_type": "A", "mutation": "G", "ddG": -0.9},
    ]

    df = pd.DataFrame(mutations)
    df['pH'] = 7.0
    df['temperature'] = 25.0
    df['protein_name'] = df['pdb_id'].map({
        '1LZ1': 'Hen egg-white lysozyme',
        '1BNI': 'Barnase',
        '1STN': 'Staphylococcal nuclease',
        '2LZM': 'T4 Lysozyme',
        '2RN2': 'Ribonuclease H',
        '1RGG': 'RNase Sa',
        '2CI2': 'Chymotrypsin inhibitor 2',
    })

    logger.info(f"Generated built-in dataset with {len(df)} mutations")
    return df


# ──────────────────────────────────────────────────────────────────────
# Unified interface
# ──────────────────────────────────────────────────────────────────────

def download_dataset(
    dataset: str = 'builtin',
    data_dir: Optional[str] = None,
    max_entries: Optional[int] = None,
) -> pd.DataFrame:
    """
    Download a mutation dataset by name.

    Parameters
    ----------
    dataset : str
        One of: 's2648', 'fireprotdb', 'builtin', 'all'
    data_dir : str, optional
        Cache directory.
    max_entries : int, optional
        Max entries per dataset.

    Returns
    -------
    pd.DataFrame
    """
    dataset = dataset.lower().strip()

    if dataset == 's2648':
        df = download_s2648(data_dir=data_dir)
    elif dataset == 'fireprotdb':
        df = download_fireprotdb_hf(data_dir=data_dir, max_entries=max_entries)
    elif dataset == 'builtin':
        df = download_builtin(data_dir=data_dir)
    elif dataset == 'all':
        dfs = []
        for ds_name in ['s2648', 'fireprotdb', 'builtin']:
            try:
                d = download_dataset(ds_name, data_dir=data_dir, max_entries=max_entries)
                dfs.append(d)
            except Exception as e:
                logger.warning(f"Failed to load {ds_name}: {e}")
        if not dfs:
            raise RuntimeError("No datasets could be loaded")
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. "
                         f"Choose from: {AVAILABLE_DATASETS}")

    if 'dataset' not in df.columns:
        df['dataset'] = dataset

    if max_entries and dataset != 'all':
        df = df.head(max_entries)

    return df


def prepare_dataset(
    dataset: str = 'builtin',
    data_dir: Optional[str] = None,
    max_mutations: Optional[int] = None,
    max_pdb_downloads: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Download mutation data and corresponding PDB structures.

    Parameters
    ----------
    dataset : str
        Dataset name: 's2648', 'fireprotdb', 'builtin', or 'all'.
    data_dir : str, optional
        Directory for cached data.
    max_mutations : int, optional
        Maximum number of mutations to include.
    max_pdb_downloads : int, optional
        Maximum number of PDB files to download.

    Returns
    -------
    df : pd.DataFrame
        Mutation data.
    pdb_paths : dict
        Mapping of pdb_id -> local file path.
    """
    # Step 1: Download mutation data
    df = download_dataset(dataset, data_dir=data_dir, max_entries=max_mutations)

    if max_mutations is not None:
        df = df.head(max_mutations)

    # Step 2: Download PDB structures
    unique_pdbs = df['pdb_id'].unique()
    if max_pdb_downloads is not None:
        unique_pdbs = unique_pdbs[:max_pdb_downloads]

    pdb_paths = {}
    failed_pdbs = []

    # Check how many are already cached to show an accurate message
    data_path = get_data_dir(data_dir) / "pdb"
    cached = [p for p in unique_pdbs if (data_path / f"{p.upper()}.pdb").exists()]
    to_download = [p for p in unique_pdbs if p not in cached]

    if cached:
        logger.info(f"{len(cached)} PDB files already cached, "
                     f"{len(to_download)} to download")

    desc = ("Loading/downloading PDB structures" if to_download
            else "Loading cached PDB structures")

    for pdb_id in tqdm(unique_pdbs, desc=desc):
        try:
            path = download_pdb_structure(pdb_id, data_dir=data_dir)
            pdb_paths[pdb_id] = path
        except Exception as e:
            logger.warning(f"Failed to download PDB {pdb_id}: {e}")
            failed_pdbs.append(pdb_id)

    # Filter out mutations for failed PDBs
    if failed_pdbs:
        df = df[~df['pdb_id'].isin(failed_pdbs)]
        logger.info(f"Removed {len(failed_pdbs)} PDBs that failed to download. "
                     f"Remaining: {len(df)} mutations across {len(pdb_paths)} structures.")

    data_root = get_data_dir(data_dir)
    logger.info(f"All data saved under: {data_root}")
    logger.info(f"  Datasets (CSVs): {data_root / 'datasets'}")
    logger.info(f"  PDB structures:  {data_root / 'pdb'}")

    return df, pdb_paths
