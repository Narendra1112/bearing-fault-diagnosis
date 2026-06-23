"""
monitoring/metrics.py — Prometheus metrics for the bearing inference API.

Metrics exposed at GET /metrics (Prometheus text format):

  inference_requests_total      Counter   — total requests by class + status
  inference_latency_seconds     Histogram — per-request latency distribution
  drift_detections_total        Counter   — drift events by feature + method
  model_confidence              Histogram — distribution of prediction confidence
  bearing_health_index          Gauge     — most recent bearing health score
  active_model_version          Info      — currently loaded model version

Prometheus scrapes /metrics every 15s (configured in prometheus.yml).

Usage:
    from src.monitoring.metrics import metrics
    metrics.record_inference("normal", "success", latency_s=0.004, confidence=0.99)
    metrics.record_drift("kurtosis", "cusum")
    metrics.set_health_index(0.92)
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

# ── Latency buckets (seconds) — fine-grained for low-latency inference ────────
_LATENCY_BUCKETS = (0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0)


class BearingMetrics:
    """
    Centralised Prometheus metrics registry for the bearing API.

    Using a dedicated registry (not the default global) so that tests can
    instantiate isolated registries without metric name conflicts.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry or CollectorRegistry()

        # ── Inference request counter ──────────────────────────────────────
        self.inference_requests_total = Counter(
            name="inference_requests_total",
            documentation=(
                "Total number of inference requests processed. "
                "Labels: predicted_class, status (success|error|timeout)."
            ),
            labelnames=["predicted_class", "status"],
            registry=self._registry,
        )

        # ── Latency histogram ──────────────────────────────────────────────
        self.inference_latency_seconds = Histogram(
            name="inference_latency_seconds",
            documentation="End-to-end inference latency in seconds.",
            buckets=_LATENCY_BUCKETS,
            registry=self._registry,
        )

        # ── Drift detection counter ────────────────────────────────────────
        self.drift_detections_total = Counter(
            name="drift_detections_total",
            documentation=(
                "Total drift detection events. "
                "Labels: feature (kurtosis|peak_to_peak|crest_factor), "
                "method (zscore|cusum|mmd)."
            ),
            labelnames=["feature", "method"],
            registry=self._registry,
        )

        # ── Confidence histogram ───────────────────────────────────────────
        self.model_confidence = Histogram(
            name="model_confidence",
            documentation="Distribution of model prediction confidence scores.",
            buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
            registry=self._registry,
        )

        # ── Bearing health gauge ───────────────────────────────────────────
        self.bearing_health_index = Gauge(
            name="bearing_health_index",
            documentation=(
                "Most recent bearing health index [0=severe fault, 1=healthy]. "
                "Derived from physics feature extraction."
            ),
            registry=self._registry,
        )

        # ── Active model info ──────────────────────────────────────────────
        self.active_model_version = Info(
            name="active_model",
            documentation="Currently active model version and backend.",
            registry=self._registry,
        )

        # ── Batch size histogram ───────────────────────────────────────────
        self.batch_size = Histogram(
            name="batch_predict_size",
            documentation="Number of signals per batch_predict request.",
            buckets=(1, 2, 4, 8, 16, 32),
            registry=self._registry,
        )

        # ── Uncertainty gauge ──────────────────────────────────────────────
        self.uncertainty_std = Histogram(
            name="prediction_uncertainty_std",
            documentation="MC Dropout uncertainty std per prediction.",
            buckets=(0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
            registry=self._registry,
        )

        log.info("Prometheus metrics registry initialised")

    # ── Recording helpers ─────────────────────────────────────────────────

    def record_inference(
        self,
        predicted_class: str,
        status: str,
        latency_s: float,
        confidence: float,
        health_index: float | None = None,
        uncertainty: float | None = None,
    ) -> None:
        """
        Record a completed inference call.

        Args:
            predicted_class: Predicted fault class name.
            status:          "success", "error", or "timeout".
            latency_s:       End-to-end latency in seconds.
            confidence:      Softmax confidence of the top prediction.
            health_index:    Optional bearing health index [0, 1].
            uncertainty:     Optional MC Dropout uncertainty std.
        """
        self.inference_requests_total.labels(
            predicted_class=predicted_class,
            status=status,
        ).inc()

        self.inference_latency_seconds.observe(latency_s)
        self.model_confidence.observe(confidence)

        if health_index is not None:
            self.bearing_health_index.set(health_index)

        if uncertainty is not None:
            self.uncertainty_std.observe(uncertainty)

    def record_drift(
        self,
        feature: str,
        method: str,
    ) -> None:
        """
        Increment the drift detection counter.

        Args:
            feature: Feature name (kurtosis, peak_to_peak, crest_factor).
            method:  Detection method (zscore, cusum, mmd).
        """
        self.drift_detections_total.labels(
            feature=feature,
            method=method,
        ).inc()

    def record_batch(self, batch_size: int) -> None:
        """Record the size of a batch_predict request."""
        self.batch_size.observe(batch_size)

    def set_model_version(self, version: str, backend: str = "pytorch") -> None:
        """Update the active model info metric."""
        self.active_model_version.info({
            "version": version,
            "backend": backend,
            "environment": settings.ENVIRONMENT,
        })

    # ── Prometheus output ─────────────────────────────────────────────────

    def generate_text(self) -> tuple[bytes, str]:
        """
        Generate Prometheus text format output.

        Returns:
            Tuple of (content_bytes, content_type_string).
            Pass directly to a FastAPI Response.
        """
        return generate_latest(self._registry), CONTENT_TYPE_LATEST


# ── Module-level singleton ────────────────────────────────────────────────────

metrics: BearingMetrics = BearingMetrics()
