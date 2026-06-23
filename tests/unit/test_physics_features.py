"""
tests/unit/test_physics_features.py — Unit tests for physics feature extraction.

Tests:
  - BPFO/BPFI/BSF/FTF formula correctness (against known CWRU values)
  - Band energy extraction on synthetic signals with known frequencies
  - Bearing health index == 1.0 for normal (Gaussian) signals
  - Feature vector length is always N_FEATURES (19)
  - Batch extraction shape is (N, 19)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.ml.physics_features import (
    BearingGeometry,
    N_FEATURES,
    PhysicsFeatureExtractor,
    _band_energy,
)


# ── Known CWRU defect frequencies at 1797 RPM ────────────────────────────────
# Verified against CWRU documentation:
# https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures

KNOWN_BPFO = 107.36   # Hz (outer race)
KNOWN_BPFI = 162.19   # Hz (inner race)
KNOWN_BSF  =  70.58   # Hz (ball spin)
KNOWN_FTF  =  11.93   # Hz (cage)
TOLERANCE  = 0.5      # Hz — allow 0.5 Hz rounding tolerance


class TestBearingGeometry:
    """Verify defect frequency formulas against published CWRU values."""

    def setup_method(self):
        self.bearing = BearingGeometry(
            nb=9, bd=0.3126, pd=1.537, contact_angle_deg=0.0, fs=12_000.0
        )

    def test_bpfo(self):
        fd = self.bearing.defect_frequencies(rpm=1797.0)
        assert abs(fd.bpfo - KNOWN_BPFO) < TOLERANCE, \
            f"BPFO {fd.bpfo:.2f} Hz != {KNOWN_BPFO} Hz"

    def test_bpfi(self):
        fd = self.bearing.defect_frequencies(rpm=1797.0)
        assert abs(fd.bpfi - KNOWN_BPFI) < TOLERANCE, \
            f"BPFI {fd.bpfi:.2f} Hz != {KNOWN_BPFI} Hz"

    def test_bsf(self):
        fd = self.bearing.defect_frequencies(rpm=1797.0)
        assert abs(fd.bsf - KNOWN_BSF) < TOLERANCE, \
            f"BSF {fd.bsf:.2f} Hz != {KNOWN_BSF} Hz"

    def test_ftf(self):
        fd = self.bearing.defect_frequencies(rpm=1797.0)
        assert abs(fd.ftf - KNOWN_FTF) < TOLERANCE, \
            f"FTF {fd.ftf:.2f} Hz != {KNOWN_FTF} Hz"

    def test_bpfo_greater_than_ftf(self):
        """BPFO is always much larger than FTF."""
        fd = self.bearing.defect_frequencies(rpm=1797.0)
        assert fd.bpfo > fd.ftf * 5

    def test_bpfi_greater_than_bpfo(self):
        """Inner race frequency is always higher than outer race."""
        fd = self.bearing.defect_frequencies(rpm=1797.0)
        assert fd.bpfi > fd.bpfo

    def test_scales_with_rpm(self):
        """Frequencies scale linearly with RPM."""
        fd_1797 = self.bearing.defect_frequencies(1797.0)
        fd_1730 = self.bearing.defect_frequencies(1730.0)
        ratio   = 1730.0 / 1797.0
        assert abs(fd_1730.bpfo / fd_1797.bpfo - ratio) < 0.01


class TestBandEnergy:
    """Verify band energy extraction on synthetic signals."""

    def test_pure_tone_at_center(self):
        """A pure tone at center frequency should have maximum band energy."""
        fs    = 12_000.0
        n     = 1024
        freq  = 107.36   # BPFO
        t     = np.arange(n) / fs
        sig   = np.sin(2 * np.pi * freq * t).astype(np.float32)

        mag   = np.abs(np.fft.rfft(sig)) / n
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        energy_at_bpfo    = _band_energy(mag, freqs, center=freq,    half_bw=30.0)
        energy_at_far_off = _band_energy(mag, freqs, center=3000.0,  half_bw=30.0)

        assert energy_at_bpfo > energy_at_far_off * 10, \
            "Tone energy should be much higher at BPFO than at unrelated frequency"

    def test_empty_band(self):
        """Band outside signal bandwidth should return near-zero energy."""
        fs    = 12_000.0
        n     = 1024
        sig   = np.zeros(n, dtype=np.float32)
        mag   = np.abs(np.fft.rfft(sig)) / n
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        assert _band_energy(mag, freqs, center=100.0) < 1e-10


class TestPhysicsFeatureExtractor:
    """End-to-end feature extraction tests."""

    @pytest.fixture
    def extractor(self):
        return PhysicsFeatureExtractor(rpm=1797.0)

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_feature_vector_length(self, extractor, rng):
        """Feature vector must always have exactly N_FEATURES (19) elements."""
        sig = rng.standard_normal(1024).astype(np.float32)
        features = extractor.extract(sig)
        assert len(features) == N_FEATURES == 19

    def test_health_index_normal_signal(self, extractor, rng):
        """Gaussian noise (healthy signal) should give health index near 1.0."""
        # Average over multiple samples for stability
        scores = []
        for _ in range(10):
            sig    = rng.standard_normal(1024).astype(np.float32)
            feats  = extractor.extract(sig)
            scores.append(feats[15])   # health index

        mean_health = float(np.mean(scores))
        assert mean_health > 0.85, \
            f"Health index {mean_health:.3f} too low for normal signal"

    def test_health_index_impulsive_signal(self, extractor):
        """Highly impulsive signal (simulated fault) should give low health index."""
        sig = np.zeros(1024, dtype=np.float32)
        sig[::64] = 20.0   # strong impulses every 64 samples
        mu, s = sig.mean(), sig.std()
        sig   = ((sig - mu) / (s + 1e-8)).astype(np.float32)

        feats = extractor.extract(sig)
        health = feats[15]
        assert health < 0.5, \
            f"Health index {health:.3f} should be < 0.5 for impulsive signal"

    def test_envelope_kurtosis_normal_is_near_3(self, extractor, rng):
        """Gaussian signal has kurtosis ≈ 3 (excess kurtosis ≈ 0)."""
        scores = []
        for _ in range(20):
            sig   = rng.standard_normal(1024).astype(np.float32)
            feats = extractor.extract(sig)
            scores.append(feats[9])   # envelope kurtosis

        mean_kurt = float(np.mean(scores))
        assert 2.0 < mean_kurt < 6.0, \
            f"Normal envelope kurtosis {mean_kurt:.2f} should be near 3"

    def test_batch_shape(self, extractor, rng):
        """Batch extraction must return shape (N, 19)."""
        N      = 50
        X      = rng.standard_normal((N, 1024)).astype(np.float32)
        feats  = extractor.extract_batch(X)
        assert feats.shape == (N, N_FEATURES)

    def test_features_finite(self, extractor, rng):
        """All extracted features must be finite (no NaN/Inf)."""
        for _ in range(20):
            sig   = rng.standard_normal(1024).astype(np.float32)
            feats = extractor.extract(sig)
            assert np.all(np.isfinite(feats)), "Feature vector contains NaN or Inf"

    def test_bpfo_energy_higher_for_outer_race(self, extractor):
        """
        Synthetic outer race fault (impulse at BPFO period) should have
        higher BPFO energy than a normal Gaussian signal on average.
        Note: This test uses the average over many windows, not individual windows,
        because single-window energy varies widely.
        """
        fs     = 12_000.0
        n      = 1024
        n_wins = 50
        rng    = np.random.default_rng(99)

        # Normal: pure Gaussian
        normal_bpfo = []
        for _ in range(n_wins):
            sig   = rng.standard_normal(n).astype(np.float32)
            feats = extractor.extract(sig)
            normal_bpfo.append(feats[0])

        # Outer race fault: periodic impulses at BPFO = 107.36 Hz
        bpfo_hz    = extractor.freqs_def.bpfo
        period_s   = 1.0 / bpfo_hz
        period_smp = int(period_s * fs)
        fault_bpfo = []
        for _ in range(n_wins):
            sig = rng.standard_normal(n).astype(np.float64) * 0.1
            for k in range(0, n, period_smp):
                sig[k] += 5.0   # impulse at each BPFO period
            mu, s = sig.mean(), sig.std()
            sig   = ((sig - mu) / (s + 1e-8)).astype(np.float32)
            feats = extractor.extract(sig)
            fault_bpfo.append(feats[0])

        assert np.mean(fault_bpfo) > np.mean(normal_bpfo), \
            "Outer race fault should show higher mean BPFO energy than normal"
