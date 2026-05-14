"""
load_data.py — CWRU Bearing Dataset Loader

Files were downloaded via src/download_data.py (Zenodo mirrors) and live in
data/raw/ with descriptive names:
  normal_<rpm>.mat   — healthy baseline at four motor loads
  B<diam>_1797.mat   — ball fault at 1797 RPM
  IR<diam>_1797.mat  — inner race fault at 1797 RPM
  OR<diam>_1797.mat  — outer race fault at 1797 RPM (6 o'clock position)

Each .mat file contains a Drive End accelerometer channel keyed as X<id>_DE_time.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Maps each .mat filename to fault metadata.
# rpm variants of the normal baseline are merged under class 0 during loading.
CWRU_FILE_MAP = {
    # Normal baseline — four motor loads (1797 / 1772 / 1750 / 1730 RPM)
    "normal_1797.mat": {"label": "Normal",         "fault_type": "normal",      "diameter": 0.000, "rpm": 1797},
    "normal_1772.mat": {"label": "Normal",         "fault_type": "normal",      "diameter": 0.000, "rpm": 1772},
    "normal_1750.mat": {"label": "Normal",         "fault_type": "normal",      "diameter": 0.000, "rpm": 1750},
    "normal_1730.mat": {"label": "Normal",         "fault_type": "normal",      "diameter": 0.000, "rpm": 1730},
    # Ball fault — 12 kHz Drive End, 1797 RPM
    "B007_1797.mat":   {"label": "Ball_007",       "fault_type": "ball",        "diameter": 0.007, "rpm": 1797},
    "B014_1797.mat":   {"label": "Ball_014",       "fault_type": "ball",        "diameter": 0.014, "rpm": 1797},
    "B021_1797.mat":   {"label": "Ball_021",       "fault_type": "ball",        "diameter": 0.021, "rpm": 1797},
    # Inner race fault — 12 kHz Drive End, 1797 RPM
    "IR007_1797.mat":  {"label": "InnerRace_007",  "fault_type": "inner_race",  "diameter": 0.007, "rpm": 1797},
    "IR014_1797.mat":  {"label": "InnerRace_014",  "fault_type": "inner_race",  "diameter": 0.014, "rpm": 1797},
    "IR021_1797.mat":  {"label": "InnerRace_021",  "fault_type": "inner_race",  "diameter": 0.021, "rpm": 1797},
    # Outer race fault (6 o'clock position) — 12 kHz Drive End, 1797 RPM
    "OR007_1797.mat":  {"label": "OuterRace_007",  "fault_type": "outer_race",  "diameter": 0.007, "rpm": 1797},
    "OR014_1797.mat":  {"label": "OuterRace_014",  "fault_type": "outer_race",  "diameter": 0.014, "rpm": 1797},
    "OR021_1797.mat":  {"label": "OuterRace_021",  "fault_type": "outer_race",  "diameter": 0.021, "rpm": 1797},
}

# Integer label encoding used throughout the project
LABEL_ENCODING = {
    "normal":      0,
    "ball":        1,
    "inner_race":  2,
    "outer_race":  3,
}


# ---------------------------------------------------------------------------
# Low-level .mat reader
# ---------------------------------------------------------------------------

def _read_mat(filepath: Path) -> dict:
    """Return the contents of a .mat file as a plain Python dict."""
    try:
        data = sio.loadmat(str(filepath), squeeze_me=True)
        # Remove scipy metadata keys that start with '__'
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception as e:
        raise IOError(f"Cannot read {filepath}: {e}") from e


def _extract_de_signal(mat_dict: dict) -> np.ndarray:
    """
    Pull the Drive End (DE) accelerometer channel from a mat dict.
    CWRU keys look like 'X097_DE_time', 'X098_DE_time', etc.
    """
    for key, value in mat_dict.items():
        if "DE_time" in key:
            signal = np.asarray(value, dtype=np.float64).ravel()
            return signal
    raise KeyError(f"No DE_time channel found. Available keys: {list(mat_dict.keys())}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_single_file(filename: str, data_dir: Path = DATA_DIR) -> dict:
    """
    Load one CWRU .mat file and return a dict with signal + metadata.

    Returns
    -------
    {
        "signal"     : np.ndarray  – raw DE accelerometer time series,
        "label"      : str         – human-readable fault label,
        "fault_type" : str         – 'normal' | 'ball' | 'inner_race' | 'outer_race',
        "class_id"   : int         – integer class index,
        "diameter"   : float       – fault diameter in inches,
        "filename"   : str,
    }
    """
    if filename not in CWRU_FILE_MAP:
        raise ValueError(f"Unknown file '{filename}'. Add it to CWRU_FILE_MAP first.")

    filepath = data_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    meta = CWRU_FILE_MAP[filename]
    mat  = _read_mat(filepath)
    sig  = _extract_de_signal(mat)

    return {
        "signal":     sig,
        "label":      meta["label"],
        "fault_type": meta["fault_type"],
        "class_id":   LABEL_ENCODING[meta["fault_type"]],
        "diameter":   meta["diameter"],
        "rpm":        meta["rpm"],
        "filename":   filename,
    }


def load_all_files(data_dir: Path = DATA_DIR, verbose: bool = True) -> list[dict]:
    """
    Load every .mat file listed in CWRU_FILE_MAP that exists in data_dir.

    Returns a list of record dicts (same schema as load_single_file).
    Missing files are skipped with a warning rather than raising an error,
    so a partially-downloaded dataset still works.
    """
    records = []
    for filename in CWRU_FILE_MAP:
        filepath = data_dir / filename
        if not filepath.exists():
            if verbose:
                print(f"[SKIP] {filename} not found in {data_dir}")
            continue
        try:
            record = load_single_file(filename, data_dir)
            records.append(record)
            if verbose:
                print(f"[OK]   {filename:30s} | {len(record['signal']):>9,} samples | class={record['class_id']}")
        except Exception as exc:
            print(f"[ERR]  {filename}: {exc}")

    if not records:
        raise RuntimeError(
            f"No data files loaded from '{data_dir}'.\n"
            "Download CWRU .mat files and place them in data/raw/."
        )
    return records


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Convert a list of record dicts (without the raw signal array) into a
    tidy DataFrame suitable for logging, EDA, and reporting.
    """
    rows = [
        {k: v for k, v in r.items() if k != "signal"}
        for r in records
    ]
    df = pd.DataFrame(rows)
    df["n_samples"] = [len(r["signal"]) for r in records]
    return df


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Looking for data in: {DATA_DIR}\n")
    records = load_all_files(verbose=True)
    df = records_to_dataframe(records)
    print("\nDataset summary:")
    print(df.to_string(index=False))
