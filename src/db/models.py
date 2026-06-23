"""
db/models.py — SQLAlchemy ORM models for bearing fault diagnosis persistence.

Tables:
  predictions    — every inference call with physics features + uncertainty
  drift_events   — each detected drift event with method and severity
  model_registry — registered model versions with accuracy metrics

All tables use integer primary keys for compatibility with both SQLite (dev)
and PostgreSQL (prod). Timestamps are stored in UTC.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, Index
)
from sqlalchemy.orm import DeclarativeBase


def _utcnow() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


# ── Predictions ───────────────────────────────────────────────────────────────

class PredictionRecord(Base):
    """
    Persists every /predict call for monitoring, retraining, and audit.

    signal_hash allows deduplication and linking back to raw inputs stored
    externally (e.g., S3).
    """

    __tablename__ = "predictions"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    timestamp            = Column(DateTime(timezone=True), nullable=False,
                                  default=_utcnow, index=True)
    signal_hash          = Column(String(64), nullable=False, index=True)
    predicted_class      = Column(String(32), nullable=False, index=True)
    class_id             = Column(Integer, nullable=False)
    confidence           = Column(Float, nullable=False)
    uncertainty_std      = Column(Float, nullable=True)   # None if MC Dropout disabled
    is_uncertain         = Column(Boolean, nullable=True)
    latency_ms           = Column(Float, nullable=False)
    drift_warning        = Column(Boolean, nullable=False, default=False)
    drift_severity       = Column(String(16), nullable=True)   # NONE/WARNING/CRITICAL
    model_version        = Column(String(64), nullable=False, default="unknown")
    backend              = Column(String(16), nullable=False, default="pytorch")

    # Physics features (key discriminators stored for retraining analysis)
    bpfo_energy          = Column(Float, nullable=True)
    bpfi_energy          = Column(Float, nullable=True)
    bsf_energy           = Column(Float, nullable=True)
    envelope_kurtosis    = Column(Float, nullable=True)
    bearing_health_index = Column(Float, nullable=True)

    created_at           = Column(DateTime(timezone=True), nullable=False,
                                  default=_utcnow)

    # Composite index for time-range + class queries
    __table_args__ = (
        Index("ix_predictions_ts_class", "timestamp", "predicted_class"),
    )

    @staticmethod
    def hash_signal(signal: list[float] | "np.ndarray") -> str:  # noqa: F821
        """SHA-256 hash of the signal bytes — used for deduplication."""
        import numpy as np
        arr = np.asarray(signal, dtype=np.float32)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def __repr__(self) -> str:
        return (
            f"<PredictionRecord id={self.id} class={self.predicted_class} "
            f"conf={self.confidence:.3f} ts={self.timestamp}>"
        )


# ── Drift events ──────────────────────────────────────────────────────────────

class DriftEvent(Base):
    """
    Records each drift detection event — one row per triggered feature per check.

    Multiple features can drift simultaneously; each gets its own row.
    resolved_at is populated when an operator calls POST /drift/reset.
    """

    __tablename__ = "drift_events"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    timestamp        = Column(DateTime(timezone=True), nullable=False,
                              default=_utcnow, index=True)
    feature_name     = Column(String(64), nullable=False)
    z_score          = Column(Float, nullable=True)
    cusum_value      = Column(Float, nullable=True)
    threshold        = Column(Float, nullable=False)
    severity         = Column(String(16), nullable=False, default="WARNING")
    detection_method = Column(String(16), nullable=False)  # zscore | cusum | mmd
    resolved_at      = Column(DateTime(timezone=True), nullable=True)
    created_at       = Column(DateTime(timezone=True), nullable=False,
                              default=_utcnow)

    __table_args__ = (
        Index("ix_drift_events_ts", "timestamp"),
        Index("ix_drift_events_feature", "feature_name"),
    )

    def __repr__(self) -> str:
        return (
            f"<DriftEvent id={self.id} feature={self.feature_name} "
            f"method={self.detection_method} severity={self.severity}>"
        )


# ── Model registry ────────────────────────────────────────────────────────────

class ModelRecord(Base):
    """
    Tracks all registered model versions with their validation metrics.

    Only one record has is_active=True at any time.
    Switching versions is done via POST /models/{version}/activate which
    updates is_active and triggers a model reload.
    """

    __tablename__ = "model_registry"

    id                        = Column(Integer, primary_key=True, autoincrement=True)
    version                   = Column(String(64), nullable=False, unique=True)
    path                      = Column(Text, nullable=False)
    is_active                 = Column(Boolean, nullable=False, default=False,
                                       index=True)

    # Validation metrics (populated by model_validator.py)
    accuracy                  = Column(Float, nullable=True)   # same-condition test
    per_class_min_accuracy    = Column(Float, nullable=True)   # worst class accuracy
    cross_condition_accuracy  = Column(Float, nullable=True)   # unseen RPM
    noise_10db_accuracy       = Column(Float, nullable=True)   # accuracy at 10dB SNR
    inference_p95_ms          = Column(Float, nullable=True)   # p95 latency

    # Validation gate result
    validation_passed         = Column(Boolean, nullable=True)
    validation_notes          = Column(Text, nullable=True)

    created_at                = Column(DateTime(timezone=True), nullable=False,
                                       default=_utcnow)

    def __repr__(self) -> str:
        return (
            f"<ModelRecord version={self.version} active={self.is_active} "
            f"acc={self.accuracy}>"
        )
