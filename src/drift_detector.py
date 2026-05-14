"""
drift_detector.py — Statistical data-drift detection for incoming signals

At startup, training-set statistics (mean and std) are computed for three
signal descriptors that remain meaningful even after per-window z-score
normalisation:

  kurtosis     — impulsiveness; spikes with faulty bearings
  peak_to_peak — dynamic range
  crest_factor — peak / RMS ratio

An incoming signal is flagged as drifting if any descriptor's z-score
exceeds `threshold` standard deviations from the training mean.
"""

import threading
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis as _kurtosis

ROOT = Path(__file__).resolve().parent.parent


def _signal_stats(x: np.ndarray) -> dict[str, float]:
    """Compute the three drift-sensitive descriptors for one window."""
    x    = x.astype(np.float64)
    rms  = float(np.sqrt(np.mean(x ** 2)))
    return {
        "kurtosis":     float(_kurtosis(x, fisher=True)),
        "peak_to_peak": float(x.max() - x.min()),
        "crest_factor": float(np.max(np.abs(x)) / (rms + 1e-12)),
    }


class DriftDetector:
    """
    Computes per-feature z-scores for incoming signals against the training
    distribution and raises a WARNING flag when any z-score exceeds the
    threshold (default 2.0 std deviations).
    """

    def __init__(
        self,
        train_npz:  Path | None = None,
        threshold:  float = 2.0,
    ):
        self.threshold = threshold
        self._lock     = threading.Lock()

        # Mutable drift state (updated on every check() call)
        self._state: dict = {
            "drifting":  False,
            "flags":     {},
            "n_checked": 0,
        }

        # ---- Compute training reference statistics -----------------------
        if train_npz is None:
            train_npz = ROOT / "data" / "processed" / "train.npz"

        print(f"[DriftDetector] Computing training reference stats from {train_npz} ...")
        data    = np.load(train_npz)
        X_train = data["X"]                       # (N, 1024) float32

        all_stats = [_signal_stats(w) for w in X_train]
        features  = list(all_stats[0].keys())

        self.reference: dict[str, dict[str, float]] = {}
        for feat in features:
            vals = np.array([s[feat] for s in all_stats])
            self.reference[feat] = {
                "mean": float(vals.mean()),
                "std":  float(vals.std()),
                "p5":   float(np.percentile(vals, 5)),
                "p95":  float(np.percentile(vals, 95)),
            }

        print(f"[DriftDetector] Reference built on {len(X_train):,} windows.")
        for feat, ref in self.reference.items():
            print(
                f"  {feat:<15s}  mean={ref['mean']:.4f}  "
                f"std={ref['std']:.4f}  "
                f"[p5={ref['p5']:.4f}, p95={ref['p95']:.4f}]"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, signal: np.ndarray) -> dict:
        """
        Evaluate one incoming signal window for drift.

        Returns a dict with:
          drifting  — bool, True if any feature exceeds the threshold
          flags     — per-feature breakdown with value, z-score, drift bool
          n_checked — running count of windows checked
        """
        incoming = _signal_stats(np.asarray(signal, dtype=np.float64))
        flags    = {}

        for feat, val in incoming.items():
            ref    = self.reference[feat]
            z      = abs(val - ref["mean"]) / (ref["std"] + 1e-12)
            flags[feat] = {
                "value":      round(val,          4),
                "train_mean": round(ref["mean"],  4),
                "train_std":  round(ref["std"],   4),
                "z_score":    round(float(z),     3),
                "drifting":   bool(z > self.threshold),
            }

        any_drift = any(f["drifting"] for f in flags.values())

        with self._lock:
            self._state = {
                "drifting":  any_drift,
                "flags":     flags,
                "n_checked": self._state["n_checked"] + 1,
                "threshold_std": self.threshold,
            }

        return dict(self._state)

    def status(self) -> dict:
        """Return the drift result from the most recent check() call."""
        with self._lock:
            return dict(self._state)

    def training_reference(self) -> dict:
        """Expose the reference statistics (used by /drift endpoint)."""
        return {
            feat: {k: round(v, 4) for k, v in ref.items()}
            for feat, ref in self.reference.items()
        }
