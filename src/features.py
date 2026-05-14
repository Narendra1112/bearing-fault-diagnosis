"""
features.py — Hand-crafted feature extraction for bearing fault diagnosis

Features extracted per window (11 total):
  Time-domain  : RMS, Kurtosis, Crest Factor, Skewness, Peak-to-Peak
  Frequency    : FFT top-5 magnitudes (sorted descending)
  Envelope     : Envelope Spectrum RMS

Usage (CLI):
    python src/features.py
Loads data/processed/{train,val,test}.npz, extracts features, saves
data/processed/{train,val,test}_features.npz, prints shapes.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert
from tqdm import tqdm

ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Feature names (fixed order — must match extract_window exactly)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "rms",
    "kurtosis",
    "crest_factor",
    "skewness",
    "peak_to_peak",
    "fft_mag_1",      # highest FFT magnitude
    "fft_mag_2",
    "fft_mag_3",
    "fft_mag_4",
    "fft_mag_5",      # 5th highest FFT magnitude
    "envelope_rms",
]

N_FEATURES = len(FEATURE_NAMES)   # 11


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    """Root Mean Square — reflects overall vibration energy level."""
    return float(np.sqrt(np.mean(x ** 2)))


def _kurtosis(x: np.ndarray) -> float:
    """
    Fisher kurtosis (excess kurtosis, 0 for Gaussian).
    Rises sharply when impulsive spikes appear — a key fault indicator.
    """
    return float(kurtosis(x, fisher=True))


def _crest_factor(x: np.ndarray) -> float:
    """
    Peak / RMS.  Healthy bearings ~1.4; faulty ones can exceed 6.
    """
    rms = np.sqrt(np.mean(x ** 2))
    return float(np.max(np.abs(x)) / (rms + 1e-12))


def _skewness(x: np.ndarray) -> float:
    """Asymmetry of the amplitude distribution."""
    return float(skew(x))


def _peak_to_peak(x: np.ndarray) -> float:
    """Max minus min — overall dynamic range."""
    return float(np.max(x) - np.min(x))


def _fft_top5(x: np.ndarray) -> np.ndarray:
    """
    Single-sided FFT amplitude spectrum; return top-5 magnitudes (descending).
    Hann window is applied first to reduce spectral leakage.
    """
    n    = len(x)
    win  = np.hanning(n)
    amps = np.abs(np.fft.rfft(x * win)) * (2.0 / n)   # two-sided correction
    top5 = np.sort(amps)[::-1][:5]
    # Pad with zeros if spectrum has fewer than 5 bins (won't happen at 1024)
    if len(top5) < 5:
        top5 = np.pad(top5, (0, 5 - len(top5)))
    return top5.astype(np.float64)


def _envelope_rms(x: np.ndarray) -> float:
    """
    RMS of the analytic envelope (magnitude of Hilbert transform).
    The envelope captures amplitude modulation caused by fault impacts.
    """
    envelope = np.abs(hilbert(x))
    return float(np.sqrt(np.mean(envelope ** 2)))


# ---------------------------------------------------------------------------
# Single-window extraction
# ---------------------------------------------------------------------------

def extract_window(window: np.ndarray) -> np.ndarray:
    """
    Extract all 11 features from one 1-D signal window.

    Parameters
    ----------
    window : 1-D float array, length = window_size (1024)

    Returns
    -------
    feat : (11,) float64 array in FEATURE_NAMES order
    """
    x = window.astype(np.float64)
    feat = np.empty(N_FEATURES, dtype=np.float64)

    feat[0]    = _rms(x)
    feat[1]    = _kurtosis(x)
    feat[2]    = _crest_factor(x)
    feat[3]    = _skewness(x)
    feat[4]    = _peak_to_peak(x)
    feat[5:10] = _fft_top5(x)
    feat[10]   = _envelope_rms(x)

    return feat


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_features(
    X: np.ndarray,
    desc: str = "Extracting features",
) -> np.ndarray:
    """
    Extract features from every row of X.

    Parameters
    ----------
    X    : (N, window_size) array of signal windows
    desc : tqdm progress-bar label

    Returns
    -------
    X_feat : (N, 11) float64 feature matrix
    """
    N      = len(X)
    X_feat = np.empty((N, N_FEATURES), dtype=np.float64)
    for i, window in enumerate(tqdm(X, desc=desc, unit="win")):
        X_feat[i] = extract_window(window)
    return X_feat


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_features(
    X_feat: np.ndarray,
    y: np.ndarray,
    split: str,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}_features.npz"
    np.savez_compressed(path, X=X_feat, y=y)
    size_kb = path.stat().st_size // 1024
    print(f"  Saved {path.name:<30s}  shape={X_feat.shape}  ({size_kb:,} KB)")


def load_features(
    split: str,
    data_dir: Path = PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / f"{split}_features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run features.py first — {path} not found.")
    d = np.load(path)
    return d["X"], d["y"]


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(split: str, X_feat: np.ndarray, y: np.ndarray) -> None:
    print(f"\n  {split.upper()} feature matrix : {X_feat.shape}")
    print(f"  {'Feature':<20s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}")
    print("  " + "-" * 58)
    for i, name in enumerate(FEATURE_NAMES):
        col = X_feat[:, i]
        print(
            f"  {name:<20s}  {col.mean():>10.4f}  {col.std():>10.4f}"
            f"  {col.min():>10.4f}  {col.max():>10.4f}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from preprocess import load_processed

    print("=" * 60)
    print("CWRU Bearing Feature Extraction")
    print(f"Features ({N_FEATURES}): {', '.join(FEATURE_NAMES)}")
    print("=" * 60)

    results = {}
    for split in ("train", "val", "test"):
        print(f"\nLoading data/processed/{split}.npz ...")
        X_win, y = load_processed(split)
        print(f"  Windows loaded : {X_win.shape}")

        X_feat = extract_features(X_win, desc=f"  {split:>5s}")
        results[split] = (X_feat, y)

    print("\n\nSaving feature matrices to data/processed/ ...")
    for split, (X_feat, y) in results.items():
        save_features(X_feat, y, split)

    print("\n" + "=" * 60)
    print("Final feature matrix shapes")
    print("=" * 60)
    for split, (X_feat, y) in results.items():
        print(f"  {split:<6s}  X={str(X_feat.shape):<16s}  y={str(y.shape):<12s}")

    print("\nPer-feature statistics (train set):")
    X_train_feat, y_train = results["train"]
    _print_summary("train", X_train_feat, y_train)
