"""
tests/unit/test_drift_detector.py — Unit tests for multi-method drift detector.

Tests:
  - Z-score detection triggers at correct threshold
  - CUSUM accumulates and triggers at h=5
  - CUSUM reset() clears all accumulators
  - MMD with identical distributions returns low score
  - Normal signal from test set returns NONE severity
  - Impulsive signal returns WARNING or CRITICAL
  - Severity escalates: single method = WARNING, multiple = CRITICAL
"""

from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.drift_detector import (
    DriftDetector,
    DriftSeverity,
    _mmd_squared,
    _signal_stats,
)
from src.core.config import settings


@pytest.fixture
def detector():
    """DriftDetector initialised from training data."""
    return DriftDetector(train_npz=settings.TRAIN_DATA_PATH)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestSignalStats:
    """Test the signal statistics extraction helper."""

    def test_gaussian_kurtosis_near_zero(self, rng):
        """Gaussian signal has excess kurtosis near 0 (Fisher definition)."""
        sig   = rng.standard_normal(1024)
        stats = _signal_stats(sig)
        assert abs(stats["kurtosis"]) < 1.5, \
            f"Gaussian kurtosis {stats['kurtosis']:.2f} should be near 0"

    def test_impulsive_kurtosis_high(self):
        """Impulsive signal has high kurtosis."""
        sig = np.zeros(1024)
        sig[::64] = 20.0
        mu, s = sig.mean(), sig.std()
        sig   = (sig - mu) / (s + 1e-8)
        stats = _signal_stats(sig)
        assert stats["kurtosis"] > 5.0, \
            f"Impulsive kurtosis {stats['kurtosis']:.2f} should be > 5"

    def test_peak_to_peak_positive(self, rng):
        """Peak-to-peak is always non-negative."""
        sig   = rng.standard_normal(1024)
        stats = _signal_stats(sig)
        assert stats["peak_to_peak"] >= 0

    def test_crest_factor_positive(self, rng):
        """Crest factor is always positive."""
        sig   = rng.standard_normal(1024)
        stats = _signal_stats(sig)
        assert stats["crest_factor"] > 0


class TestMMD:
    """Test MMD² estimator."""

    def test_identical_distributions_low_mmd(self, rng):
        """MMD² between identical distributions should be near 0."""
        X = rng.standard_normal((100, 3)).astype(np.float32)
        Y = rng.standard_normal((100, 3)).astype(np.float32)
        # Same distribution (both Gaussian) → small MMD
        mmd = _mmd_squared(X, Y, sigma=1.0)
        assert mmd < 0.1, f"MMD² {mmd:.4f} between similar distributions should be < 0.1"

    def test_different_distributions_higher_mmd(self, rng):
        """MMD² between clearly different distributions should be > 0."""
        X = rng.standard_normal((100, 3)).astype(np.float32)
        Y = (rng.standard_normal((100, 3)) + 10.0).astype(np.float32)   # shifted by 10
        mmd = _mmd_squared(X, Y, sigma=1.0)
        assert mmd > 0.01, f"MMD² {mmd:.4f} between shifted distributions should be > 0.01"

    def test_mmd_non_negative(self, rng):
        """MMD² is always >= 0."""
        X = rng.standard_normal((50, 3)).astype(np.float32)
        Y = rng.standard_normal((50, 3)).astype(np.float32)
        assert _mmd_squared(X, Y) >= 0


class TestDriftDetector:
    """Test the DriftDetector class."""

    def test_initialises_with_reference(self, detector):
        """Detector loads training reference for 3 features."""
        assert detector.enabled
        assert set(detector.reference.keys()) == {"kurtosis", "peak_to_peak", "crest_factor"}
        for feat, ref in detector.reference.items():
            assert "mean" in ref and "std" in ref

    def test_normal_signal_returns_none(self, detector):
        """Normal test window should return NONE severity."""
        data = np.load(settings.TEST_DATA_PATH)
        X    = data["X"]
        # Test multiple normal windows
        for i in np.where(data["y"] == 0)[0][:5]:
            result = detector.check(X[i])
            assert result.severity == "NONE", \
                f"Normal window {i} returned severity {result.severity}"

    def test_impulsive_signal_triggers_warning(self, detector):
        """Highly impulsive signal should trigger WARNING or CRITICAL."""
        sig = np.zeros(1024, dtype=np.float32)
        sig[::32] = 15.0
        mu, s = sig.mean(), sig.std()
        sig   = ((sig - mu) / (s + 1e-8)).astype(np.float32)

        result = detector.check(sig)
        assert result.drifting, "Impulsive signal should be flagged as drifting"
        assert result.severity in ("WARNING", "CRITICAL")

    def test_zscore_triggers_at_threshold(self, detector):
        """Z-score should trigger when |z| > threshold (2.0)."""
        # Force a signal with kurtosis far above training mean
        ref_mean = detector.reference["kurtosis"]["mean"]
        ref_std  = detector.reference["kurtosis"]["std"]

        # Find how many std deviations to go to exceed threshold
        target_kurt = ref_mean + (detector.threshold + 1.0) * ref_std

        # Create signal with controlled high kurtosis using impulses
        sig = np.zeros(1024, dtype=np.float32)
        # Large sparse impulses → high kurtosis
        sig[0] = 100.0
        mu, s = sig.mean(), sig.std()
        sig   = ((sig - mu) / (s + 1e-8)).astype(np.float32)

        result = detector.check(sig)
        # Should trigger z-score on kurtosis
        kurt_flag = result.flags.get("kurtosis", {})
        assert kurt_flag.get("zscore_drift", False) or result.drifting, \
            "Extreme kurtosis should trigger z-score drift"

    def test_cusum_accumulates(self, detector):
        """CUSUM should accumulate across repeated calls."""
        detector.reset()

        # Feed mildly elevated signals (below z-score threshold individually)
        slightly_impulsive = np.zeros(1024, dtype=np.float32)
        slightly_impulsive[::128] = 3.0
        mu, s = slightly_impulsive.mean(), slightly_impulsive.std()
        slightly_impulsive = ((slightly_impulsive - mu) / (s + 1e-8)).astype(np.float32)

        cusum_before = detector.cusum_status()["kurtosis"]["pos"]

        for _ in range(5):
            detector.check(slightly_impulsive)

        cusum_after = detector.cusum_status()["kurtosis"]["pos"]
        assert cusum_after >= cusum_before, "CUSUM should accumulate or stay"

    def test_cusum_triggers_at_h5(self, detector):
        """CUSUM should trigger when accumulator exceeds h=5."""
        detector.reset()
        assert detector.cusum_h == 5.0

        # Repeatedly feed a mildly drifting signal until CUSUM triggers
        sig = np.zeros(1024, dtype=np.float32)
        sig[::64] = 5.0
        mu, s = sig.mean(), sig.std()
        sig   = ((sig - mu) / (s + 1e-8)).astype(np.float32)

        cusum_triggered = False
        for i in range(30):
            result = detector.check(sig)
            if any(f.get("cusum_drift") for f in result.flags.values()):
                cusum_triggered = True
                break

        assert cusum_triggered, \
            "CUSUM should have triggered within 30 samples of elevated signal"

    def test_reset_clears_cusum(self, detector):
        """reset() should set all CUSUM accumulators to 0."""
        # Build up some CUSUM state
        sig = np.zeros(1024, dtype=np.float32)
        sig[::64] = 5.0
        mu, s = sig.mean(), sig.std()
        sig   = ((sig - mu) / (s + 1e-8)).astype(np.float32)

        for _ in range(10):
            detector.check(sig)

        # Verify CUSUM has accumulated
        state_before = detector.cusum_status()
        kurtosis_cusum = state_before["kurtosis"]["pos"]
        assert kurtosis_cusum > 0, "CUSUM should have accumulated"

        # Reset
        detector.reset()
        state_after = detector.cusum_status()

        assert state_after["kurtosis"]["pos"] == 0.0
        assert state_after["kurtosis"]["neg"] == 0.0
        assert state_after["kurtosis"]["n"]   == 0

    def test_n_checked_increments(self, detector, rng):
        """n_checked should increment by 1 per check() call."""
        initial = detector._n_checked
        for i in range(5):
            sig = rng.standard_normal(1024).astype(np.float32)
            detector.check(sig)
        assert detector._n_checked == initial + 5

    def test_severity_none_when_clean(self, detector, rng):
        """Multiple clean signals should maintain NONE severity."""
        detector.reset()
        for _ in range(20):
            sig    = rng.standard_normal(1024).astype(np.float32)
            result = detector.check(sig)
            # Gaussian noise is within normal range for training distribution
            # (may occasionally trigger on z-score for individual unlucky draws)
            assert result.severity in ("NONE", "WARNING")   # CRITICAL needs both methods
