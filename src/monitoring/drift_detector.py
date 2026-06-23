"""
monitoring/drift_detector.py — Multi-method data drift detection.

Three complementary detection methods:

1. Z-SCORE (per-sample, fast)
   Flag when a single window's feature value is > threshold std deviations
   from the training mean. Catches sudden, large distributional shifts.

2. CUSUM (Cumulative Sum, sequential)
   Accumulates deviations from the training mean. Sensitive to small
   persistent drifts that z-score misses. Standard threshold h=5.
   Based on: Page, E.S. (1954). "Continuous Inspection Schemes."

3. MMD (Maximum Mean Discrepancy, batch)
   Compares the distribution of the last N predictions vs training using
   kernel two-sample test. Most statistically robust — uses RBF kernel.
   Only runs when drift_window has >= MMD_WINDOW samples.

Drift severity levels:
  NONE     — all methods clean
  WARNING  — z-score OR CUSUM triggered (single method)
  CRITICAL — multiple methods triggered simultaneously

Usage:
    from src.monitoring.drift_detector import DriftDetector
    detector = DriftDetector(train_npz=Path("data/processed/train.npz"))
    result = detector.check(signal)
    print(result.severity)   # "NONE" | "WARNING" | "CRITICAL"
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

DriftSeverity = Literal["NONE", "WARNING", "CRITICAL"]
DetectionMethod = Literal["zscore", "cusum", "mmd", "none"]

FEATURES = ["kurtosis", "peak_to_peak", "crest_factor"]


# ── Signal statistics ─────────────────────────────────────────────────────────

def _signal_stats(x: np.ndarray) -> dict[str, float]:
    """
    Compute the three drift-sensitive signal descriptors.

    All three remain meaningful after per-window z-score normalisation:
      kurtosis     — impulsiveness; spikes with faulty bearings
      peak_to_peak — dynamic range
      crest_factor — peak / RMS ratio
    """
    x   = x.astype(np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)))
    return {
        "kurtosis":     float(scipy_kurtosis(x, fisher=True)),
        "peak_to_peak": float(x.max() - x.min()),
        "crest_factor": float(np.max(np.abs(x)) / (rms + 1e-12)),
    }


# ── MMD (RBF kernel) ──────────────────────────────────────────────────────────

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Compute the RBF (Gaussian) kernel between two sample sets.
    Used for MMD calculation.
    """
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    XY = X @ Y.T
    dist_sq = XX + YY.T - 2 * XY
    return float(np.exp(-dist_sq / (2 * sigma ** 2)).mean())


def _mmd_squared(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Unbiased estimator of MMD² between sample sets X and Y.

    MMD² = E[k(X,X)] - 2*E[k(X,Y)] + E[k(Y,Y)]
    where k is the RBF kernel.

    A value near 0 means X and Y come from the same distribution.
    Values > 0.1 typically indicate meaningful distribution shift.

    Args:
        X: Reference samples, shape (n, d).
        Y: Test samples, shape (m, d).
        sigma: RBF kernel bandwidth (auto-set to median pairwise distance
               if not provided — the "median heuristic").

    Returns:
        MMD² value (float >= 0).
    """
    if sigma <= 0:
        # Median heuristic for bandwidth selection
        pairwise = np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
        sigma = float(np.sqrt(np.median(pairwise) / 2.0)) + 1e-8

    kxx = _rbf_kernel(X, X, sigma)
    kyy = _rbf_kernel(Y, Y, sigma)
    kxy = _rbf_kernel(X, Y, sigma)
    return max(0.0, kxx - 2 * kxy + kyy)


# ── CUSUM state ───────────────────────────────────────────────────────────────

@dataclass
class _CUSUMState:
    """Per-feature CUSUM accumulator."""
    pos: float = 0.0   # upper CUSUM (detects upward drift)
    neg: float = 0.0   # lower CUSUM (detects downward drift)
    n:   int   = 0     # samples processed since last reset

    def update(self, value: float, mean: float, std: float, k: float = 0.5) -> float:
        """
        Update CUSUM statistics.

        k: allowance parameter (typically 0.5 * sigma). Controls sensitivity.

        Returns:
            Max of pos/neg (the current CUSUM statistic).
        """
        z = (value - mean) / (std + 1e-12)
        self.pos = max(0.0, self.pos + z - k)
        self.neg = max(0.0, self.neg - z - k)
        self.n  += 1
        return max(self.pos, self.neg)

    def reset(self) -> None:
        self.pos = 0.0
        self.neg = 0.0
        self.n   = 0


# ── Drift result ──────────────────────────────────────────────────────────────

@dataclass
class DriftResult:
    """Result of a single drift check call."""
    drifting:         bool
    severity:         DriftSeverity
    detection_method: DetectionMethod
    n_checked:        int
    flags:            dict[str, dict[str, Any]] = field(default_factory=dict)
    mmd_score:        float | None = None
    mmd_drifting:     bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "drifting":         self.drifting,
            "severity":         self.severity,
            "detection_method": self.detection_method,
            "n_checked":        self.n_checked,
            "flags":            self.flags,
            "mmd_score":        round(self.mmd_score, 6) if self.mmd_score is not None else None,
            "mmd_drifting":     self.mmd_drifting,
        }


# ── Drift detector ────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Multi-method drift detector for bearing vibration signals.

    Computes training reference statistics at startup and compares
    incoming signals against them using three methods:
      - z-score (per-sample)
      - CUSUM   (sequential, accumulates)
      - MMD     (batch, last N predictions vs training)

    Args:
        train_npz:  Path to train.npz (must contain 'X' array).
        threshold:  Z-score threshold for per-sample detection.
        cusum_h:    CUSUM decision threshold (default 5).
        mmd_window: Minimum samples before running MMD (default 50).
        mmd_threshold: MMD² value above which batch drift is flagged.
    """

    MMD_THRESHOLD = 0.05   # empirically tuned for CWRU feature space

    def __init__(
        self,
        train_npz: Path | None = None,
        threshold: float | None = None,
        cusum_h: float | None = None,
        mmd_window: int | None = None,
    ) -> None:
        self.threshold  = threshold  or settings.DRIFT_ZSCORE_THRESHOLD
        self.cusum_h    = cusum_h    or settings.DRIFT_CUSUM_H
        self.mmd_window = mmd_window or settings.DRIFT_MMD_WINDOW
        self._lock      = threading.Lock()

        # CUSUM accumulators, one per feature
        self._cusum: dict[str, _CUSUMState] = {f: _CUSUMState() for f in FEATURES}

        # Sliding window for MMD (stores feature vectors of recent predictions)
        self._recent: deque[dict[str, float]] = deque(maxlen=self.mmd_window)

        # Running count
        self._n_checked = 0

        # Latest drift result (for /drift endpoint)
        self._last_result: DriftResult = DriftResult(
            drifting=False, severity="NONE",
            detection_method="none", n_checked=0
        )

        # ── Build training reference ──────────────────────────────────────
        if train_npz is None:
            train_npz = settings.TRAIN_DATA_PATH

        if not train_npz.exists():
            log.warning(
                "Training data not found — drift detection disabled",
                path=str(train_npz),
            )
            self.reference: dict[str, dict[str, float]] = {}
            self._train_features: np.ndarray | None = None
            return

        log.info("Building drift reference from training data", path=str(train_npz))
        data    = np.load(train_npz)
        X_train = data["X"]   # (N, 1024)

        all_stats = [_signal_stats(w) for w in X_train]

        self.reference = {}
        for feat in FEATURES:
            vals = np.array([s[feat] for s in all_stats])
            self.reference[feat] = {
                "mean": float(vals.mean()),
                "std":  float(vals.std()),
                "p5":   float(np.percentile(vals, 5)),
                "p95":  float(np.percentile(vals, 95)),
            }

        # Store normalised training features for MMD
        self._train_features = np.array([
            [s[f] for f in FEATURES] for s in all_stats
        ], dtype=np.float32)
        # Sample 200 for MMD efficiency (full set is slow)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(self._train_features),
                         size=min(200, len(self._train_features)), replace=False)
        self._train_sample = self._train_features[idx]

        log.info(
            "Drift reference built",
            n_windows=len(X_train),
            features=FEATURES,
            reference={k: {m: round(v, 4) for m, v in ref.items()}
                       for k, ref in self.reference.items()},
        )

    @property
    def enabled(self) -> bool:
        """Return True if training reference is available."""
        return bool(self.reference)

    # ── Main check ────────────────────────────────────────────────────────

    def check(self, signal: np.ndarray) -> DriftResult:
        """
        Evaluate one incoming signal for drift using all three methods.

        Args:
            signal: 1-D float array of length SIGNAL_LENGTH.

        Returns:
            DriftResult with severity, flags, CUSUM values, and MMD score.
        """
        if not self.enabled:
            return DriftResult(
                drifting=False, severity="NONE",
                detection_method="none", n_checked=self._n_checked
            )

        incoming = _signal_stats(np.asarray(signal, dtype=np.float64))
        flags: dict[str, dict[str, Any]] = {}
        methods_triggered: list[str] = []

        with self._lock:
            self._n_checked += 1
            self._recent.append(incoming)

            # ── Method 1: Z-score ─────────────────────────────────────────
            zscore_drift = False
            for feat, val in incoming.items():
                ref   = self.reference[feat]
                z     = abs(val - ref["mean"]) / (ref["std"] + 1e-12)
                drifting = bool(z > self.threshold)
                if drifting:
                    zscore_drift = True

                # ── Method 2: CUSUM ───────────────────────────────────────
                cusum_val     = self._cusum[feat].update(val, ref["mean"], ref["std"])
                cusum_drift   = bool(cusum_val > self.cusum_h)
                if cusum_drift:
                    methods_triggered.append("cusum")

                flags[feat] = {
                    "value":        round(val, 4),
                    "train_mean":   round(ref["mean"], 4),
                    "train_std":    round(ref["std"],  4),
                    "z_score":      round(float(z), 3),
                    "zscore_drift": drifting,
                    "cusum_value":  round(cusum_val, 3),
                    "cusum_drift":  cusum_drift,
                }

            if zscore_drift:
                methods_triggered.append("zscore")

            # ── Method 3: MMD (only when window is full) ──────────────────
            mmd_score:    float | None = None
            mmd_drifting: bool         = False

            if len(self._recent) >= self.mmd_window:
                recent_arr = np.array(
                    [[s[f] for f in FEATURES] for s in self._recent],
                    dtype=np.float32,
                )
                mmd_score    = _mmd_squared(self._train_sample, recent_arr)
                mmd_drifting = mmd_score > self.MMD_THRESHOLD
                if mmd_drifting:
                    methods_triggered.append("mmd")

            # ── Severity ──────────────────────────────────────────────────
            n_methods = len(set(methods_triggered))
            if n_methods == 0:
                severity: DriftSeverity = "NONE"
                method:   DetectionMethod = "none"
            elif n_methods == 1:
                severity = "WARNING"
                method   = methods_triggered[0]   # type: ignore[assignment]
            else:
                severity = "CRITICAL"
                method   = "zscore+cusum" if "mmd" not in methods_triggered else "zscore+cusum+mmd"  # type: ignore[assignment]

            result = DriftResult(
                drifting         = severity != "NONE",
                severity         = severity,
                detection_method = method,
                n_checked        = self._n_checked,
                flags            = flags,
                mmd_score        = mmd_score,
                mmd_drifting     = mmd_drifting,
            )
            self._last_result = result

        log.debug(
            "Drift check",
            severity=severity,
            method=method,
            n_checked=self._n_checked,
            mmd_score=round(mmd_score, 4) if mmd_score else None,
        )
        return result

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset CUSUM accumulators and sliding window.
        Call after investigating and resolving a confirmed drift event.
        """
        with self._lock:
            for state in self._cusum.values():
                state.reset()
            self._recent.clear()
        log.info("Drift detector reset", n_cusum_reset=len(self._cusum))

    # ── Status endpoints ──────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return the result of the most recent check() call."""
        with self._lock:
            return self._last_result.to_dict()

    def cusum_status(self) -> dict[str, dict[str, float]]:
        """Return current CUSUM accumulator values per feature."""
        with self._lock:
            return {
                feat: {
                    "pos": round(state.pos, 4),
                    "neg": round(state.neg, 4),
                    "n":   state.n,
                }
                for feat, state in self._cusum.items()
            }

    def training_reference(self) -> dict[str, dict[str, float]]:
        """Return the training reference statistics (for /drift endpoint)."""
        return {
            feat: {k: round(v, 4) for k, v in ref.items()}
            for feat, ref in self.reference.items()
        }
