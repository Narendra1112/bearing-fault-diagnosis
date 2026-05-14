"""
preprocess.py — Signal segmentation and preprocessing pipeline

Steps:
  1. Segment each raw signal into overlapping 1024-sample windows
  2. Z-score normalise each window independently (zero mean, unit variance)
  3. Assign a fine-grained label per window: fault_type + severity (10 classes)
  4. Save full X / y arrays + stratified train / val / test splits to data/processed/
"""

import sys
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
import joblib

ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Fine-grained label encoding  (fault_type + severity → integer class)
# ---------------------------------------------------------------------------
# Normal files span four RPM variants but all map to class 0.
# Fault files each have a unique (type, severity) pair → classes 1-9.

LABEL_ENCODING = {
    "normal":          0,
    "ball_0.007":      1,
    "ball_0.014":      2,
    "ball_0.021":      3,
    "inner_race_0.007": 4,
    "inner_race_0.014": 5,
    "inner_race_0.021": 6,
    "outer_race_0.007": 7,
    "outer_race_0.014": 8,
    "outer_race_0.021": 9,
}

CLASS_NAMES = {v: k for k, v in LABEL_ENCODING.items()}   # int → string


def _record_label_key(record: dict) -> str:
    """Build the LABEL_ENCODING key from a loaded record's metadata."""
    ft   = record["fault_type"]    # e.g. 'ball', 'inner_race', 'normal'
    diam = record["diameter"]      # e.g. 0.007
    if ft == "normal":
        return "normal"
    return f"{ft}_{diam:.3f}"


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_signal(
    signal: np.ndarray,
    window_size: int = 1024,
    overlap: float   = 0.5,
) -> np.ndarray:
    """
    Slice a 1-D signal into overlapping windows.

    Returns (n_windows, window_size) float32 array.
    """
    step = int(window_size * (1.0 - overlap))
    starts  = range(0, len(signal) - window_size + 1, step)
    windows = np.stack([signal[s: s + window_size] for s in starts])
    return windows.astype(np.float32)


def segment_records(
    records: list[dict],
    window_size: int = 1024,
    overlap: float   = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment all records into windows and build aligned label array.

    Returns
    -------
    X : (N, window_size) float32 — raw signal windows
    y : (N,)             int64  — fine-grained class ids
    """
    X_parts, y_parts = [], []

    for rec in records:
        key     = _record_label_key(rec)
        cls_id  = LABEL_ENCODING[key]
        windows = segment_signal(rec["signal"], window_size, overlap)
        X_parts.append(windows)
        y_parts.append(np.full(len(windows), cls_id, dtype=np.int64))

    return np.concatenate(X_parts), np.concatenate(y_parts)


# ---------------------------------------------------------------------------
# Per-window Z-score normalisation
# ---------------------------------------------------------------------------

def normalise_per_window(X: np.ndarray) -> np.ndarray:
    """
    Normalise each window independently to zero mean and unit variance.

    This removes amplitude differences between files (different motor loads,
    sensor gains) and focuses the model on signal *shape* rather than magnitude.
    A small epsilon guards against near-constant windows.
    """
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    return ((X - mean) / (std + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float  = 0.15,
    val_size: float   = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Stratified split → train / val / test.

    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=relative_val, stratify=y_train, random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_all(
    X: np.ndarray,
    y: np.ndarray,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full dataset (used for exploration / feature extraction later)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    # Train / val / test splits (compressed)
    np.savez_compressed(out_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(out_dir / "val.npz",   X=X_val,   y=y_val)
    np.savez_compressed(out_dir / "test.npz",  X=X_test,  y=y_test)

    # Class name lookup
    (out_dir / "class_names.txt").write_text(
        "\n".join(f"{cls_id}\t{name}" for cls_id, name in sorted(CLASS_NAMES.items()))
    )
    print(f"\nSaved to {out_dir}:")
    for fname in ["X.npy", "y.npy", "train.npz", "val.npz", "test.npz", "class_names.txt"]:
        size_kb = (out_dir / fname).stat().st_size // 1024
        print(f"  {fname:<20s}  {size_kb:>7,} KB")


def load_processed(split: str = "train", data_dir: Path = PROCESSED_DIR):
    path = data_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run preprocess.py first — {path} not found.")
    d = np.load(path)
    return d["X"], d["y"]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_class_distribution(y: np.ndarray, title: str = "Class distribution") -> None:
    counts = Counter(y.tolist())
    total  = len(y)
    print(f"\n{title}  (total windows: {total:,})")
    print(f"  {'Class':>5}  {'Label':<25}  {'Count':>7}  {'%':>6}")
    print("  " + "-" * 50)
    for cls_id in sorted(counts):
        name  = CLASS_NAMES.get(cls_id, str(cls_id))
        cnt   = counts[cls_id]
        pct   = 100.0 * cnt / total
        print(f"  {cls_id:>5}  {name:<25}  {cnt:>7,}  {pct:>5.1f}%")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    records: list[dict],
    window_size: int = 1024,
    overlap: float   = 0.5,
    test_size: float = 0.15,
    val_size: float  = 0.15,
) -> dict:
    """Segment → normalise → split → save → report."""

    # 1. Segment
    print(f"\n[1/4] Segmenting signals  (window={window_size}, overlap={overlap})...")
    X_raw, y = segment_records(records, window_size, overlap)
    print(f"      Raw windows shape : {X_raw.shape}  dtype={X_raw.dtype}")

    # 2. Normalise per window
    print("\n[2/4] Normalising (per-window z-score)...")
    X = normalise_per_window(X_raw)
    print(f"      Normalised shape  : {X.shape}  dtype={X.dtype}")
    print(f"      Sample mean  (should ~0) : {X.mean():.6f}")
    print(f"      Sample std   (should ~1) : {X.std():.6f}")

    # 3. Split
    print("\n[3/4] Splitting dataset (stratified)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, test_size=test_size, val_size=val_size
    )
    print(f"      Train : X={X_train.shape}  y={y_train.shape}")
    print(f"      Val   : X={X_val.shape}  y={y_val.shape}")
    print(f"      Test  : X={X_test.shape}  y={y_test.shape}")

    # 4. Save
    print("\n[4/4] Saving to data/processed/...")
    save_all(X, y, X_train, X_val, X_test, y_train, y_val, y_test)

    # Report
    print_class_distribution(y,       "Full dataset")
    print_class_distribution(y_train, "Train split ")
    print_class_distribution(y_test,  "Test split  ")

    return {
        "X": X, "y": y,
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from load_data import load_all_files

    print("Loading raw signals from data/raw/...")
    records = load_all_files(verbose=True)

    run_pipeline(records, window_size=1024, overlap=0.5)
