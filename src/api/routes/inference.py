"""
api/routes/inference.py — Single and batch inference endpoints.

POST /predict
  - Accepts one 1024-sample signal
  - Runs InferenceEngine (PyTorch/ONNX)
  - MC Dropout uncertainty estimation
  - Physics feature extraction (19 features)
  - Drift detection (z-score + CUSUM + MMD)
  - Persists to DB asynchronously
  - Records Prometheus metrics
  - Sends health alert if bearing_health_index < threshold

POST /batch_predict
  - Accepts up to MAX_BATCH_SIZE signals in one request
  - Processes in parallel via asyncio
  - Returns per-item results + overall batch latency header
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator

from src.api.middleware.auth import require_api_key
from src.core.config import settings
from src.core.exceptions import (
    BatchSizeLimitError,
    ModelNotLoadedError,
    NonFiniteSignalError,
    SignalLengthError,
)
from src.core.logger import get_logger
from src.db.crud import save_prediction
from src.db.models import PredictionRecord
from src.db.session import db_session
from src.ml.inference_engine import get_engine
from src.ml.physics_features import PhysicsFeatureExtractor
from src.ml.uncertainty import MCDropoutEstimator
from src.monitoring.alerting import alerter
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.metrics import metrics

log = get_logger(__name__)
router = APIRouter(tags=["Inference"])

# ── Module-level helpers (initialised in app lifespan) ────────────────────────
# These are set by src/api/main.py during startup
_physics_extractor: PhysicsFeatureExtractor | None = None
_mc_estimator:      MCDropoutEstimator      | None = None
_drift_detector:    DriftDetector           | None = None


def set_inference_helpers(
    extractor: PhysicsFeatureExtractor,
    mc:        MCDropoutEstimator,
    drift:     DriftDetector,
) -> None:
    """Called from main.py lifespan to inject shared helpers."""
    global _physics_extractor, _mc_estimator, _drift_detector
    _physics_extractor = extractor
    _mc_estimator      = mc
    _drift_detector    = drift


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    signal: list[float] = Field(
        ...,
        min_length=settings.SIGNAL_LENGTH,
        max_length=settings.SIGNAL_LENGTH,
        description=f"Exactly {settings.SIGNAL_LENGTH} z-score normalised samples.",
    )

    @field_validator("signal")
    @classmethod
    def check_finite(cls, v: list[float]) -> list[float]:
        arr = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise NonFiniteSignalError(
                "Signal contains NaN or infinite values."
            )
        return v


class BatchPredictRequest(BaseModel):
    signals: list[list[float]] = Field(
        ...,
        min_length=1,
        description=f"List of signals, each {settings.SIGNAL_LENGTH} samples. "
                    f"Max {settings.MAX_BATCH_SIZE} signals per request.",
    )

    @field_validator("signals")
    @classmethod
    def check_batch_size(cls, v: list) -> list:
        if len(v) > settings.MAX_BATCH_SIZE:
            raise BatchSizeLimitError(
                f"Batch size {len(v)} exceeds maximum {settings.MAX_BATCH_SIZE}.",
                details={"received": len(v), "max": settings.MAX_BATCH_SIZE},
            )
        return v


class Top3Entry(BaseModel):
    rank:        int
    class_id:    int
    class_name:  str
    probability: float


class PhysicsFeatures(BaseModel):
    bpfo_energy:          float
    bpfi_energy:          float
    bsf_energy:           float
    envelope_kurtosis:    float
    bearing_health_index: float


class UncertaintyInfo(BaseModel):
    uncertainty_std: float
    ci_lower:        float
    ci_upper:        float
    is_uncertain:    bool


class PredictResponse(BaseModel):
    predicted_class:   str
    class_id:          int
    confidence:        float
    top3:              list[Top3Entry]
    inference_ms:      float
    backend:           str
    uncertainty:       UncertaintyInfo | None
    physics:           PhysicsFeatures | None
    drift_warning:     bool
    drift_severity:    str
    drift_detail:      dict[str, Any] | None


class BatchPredictResponse(BaseModel):
    results:      list[PredictResponse]
    batch_size:   int
    total_ms:     float


# ── Shared inference logic ────────────────────────────────────────────────────

async def _run_single_inference(
    sig: np.ndarray,
    request: Request | None = None,
) -> PredictResponse:
    """
    Core inference pipeline for a single signal.
    Shared between /predict and /batch_predict.

    Persistence opens its OWN short-lived session so that concurrent
    batch_predict tasks never share a single AsyncSession (which is not
    safe for concurrent use).
    """
    engine = get_engine()
    if not engine.is_loaded():
        raise ModelNotLoadedError()

    t0 = time.perf_counter()

    # ── 1. CNN inference ──────────────────────────────────────────────────
    result = engine.predict(sig)

    # ── 2. MC Dropout uncertainty ─────────────────────────────────────────
    uncertainty_info = None
    if _mc_estimator is not None:
        unc = _mc_estimator.estimate(sig, n_samples=20)
        uncertainty_info = UncertaintyInfo(
            uncertainty_std=round(unc.uncertainty_std, 6),
            ci_lower=round(unc.ci_lower, 6),
            ci_upper=round(unc.ci_upper, 6),
            is_uncertain=unc.is_uncertain,
        )
        if unc.is_uncertain:
            log.warning(
                "Uncertain prediction",
                predicted=result.predicted_class,
                std=round(unc.uncertainty_std, 4),
            )

    # ── 3. Physics features ───────────────────────────────────────────────
    physics_info = None
    health_index = None
    if _physics_extractor is not None:
        feats        = _physics_extractor.extract(sig)
        health_index = float(feats[15])
        physics_info = PhysicsFeatures(
            bpfo_energy          = round(float(feats[0]), 6),
            bpfi_energy          = round(float(feats[1]), 6),
            bsf_energy           = round(float(feats[2]), 6),
            envelope_kurtosis    = round(float(feats[9]), 4),
            bearing_health_index = round(health_index, 4),
        )
        # Health alert (fires async if below threshold)
        alerter.send_health_alert(health_index)

    # ── 4. Drift detection ────────────────────────────────────────────────
    drift_result   = None
    drift_warning  = False
    drift_severity = "NONE"

    if _drift_detector is not None and _drift_detector.enabled:
        dr             = _drift_detector.check(sig)
        drift_warning  = dr.drifting
        drift_severity = dr.severity
        drift_result   = dr.to_dict()

        if dr.drifting:
            alerter.send_drift_alert(drift_result)
            metrics.record_drift(
                feature=next(
                    (f for f, info in dr.flags.items() if info.get("zscore_drift") or info.get("cusum_drift")),
                    "unknown",
                ),
                method=dr.detection_method,
            )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # ── 5. Prometheus metrics ─────────────────────────────────────────────
    metrics.record_inference(
        predicted_class=result.predicted_class,
        status="success",
        latency_s=latency_ms / 1000.0,
        confidence=result.confidence,
        health_index=health_index,
        uncertainty=uncertainty_info.uncertainty_std if uncertainty_info else None,
    )

    # ── 6. Persist to DB (fire and forget) ───────────────────────────────
    try:
        pred_data = {
            "signal_hash":          PredictionRecord.hash_signal(sig),
            "predicted_class":      result.predicted_class,
            "class_id":             result.class_id,
            "confidence":           result.confidence,
            "uncertainty_std":      uncertainty_info.uncertainty_std if uncertainty_info else None,
            "is_uncertain":         uncertainty_info.is_uncertain if uncertainty_info else None,
            "latency_ms":           latency_ms,
            "drift_warning":        drift_warning,
            "drift_severity":       drift_severity,
            "model_version":        settings.MODEL_VERSION,
            "backend":              result.backend,
            "bpfo_energy":          physics_info.bpfo_energy if physics_info else None,
            "bpfi_energy":          physics_info.bpfi_energy if physics_info else None,
            "bsf_energy":           physics_info.bsf_energy if physics_info else None,
            "envelope_kurtosis":    physics_info.envelope_kurtosis if physics_info else None,
            "bearing_health_index": physics_info.bearing_health_index if physics_info else None,
        }
        # Own session per call — safe for concurrent batch tasks.
        async with db_session() as s:
            await save_prediction(s, pred_data)
    except Exception as exc:
        log.error("Failed to persist prediction", error=str(exc))
        # Never fail inference because of a DB error

    log.info(
        "Inference complete",
        predicted=result.predicted_class,
        confidence=round(result.confidence, 4),
        latency_ms=round(latency_ms, 2),
        drift=drift_severity,
        health=round(health_index, 3) if health_index else None,
    )

    return PredictResponse(
        predicted_class = result.predicted_class,
        class_id        = result.class_id,
        confidence      = round(result.confidence, 6),
        top3            = [Top3Entry(**t) for t in result.top3],
        inference_ms    = round(latency_ms, 3),
        backend         = result.backend,
        uncertainty     = uncertainty_info,
        physics         = physics_info,
        drift_warning   = drift_warning,
        drift_severity  = drift_severity,
        drift_detail    = drift_result,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Single signal inference",
    description=(
        "Run CNN inference on a single 1024-sample vibration window. "
        "Returns fault class, confidence, MC Dropout uncertainty, "
        "physics features (BPFO/BPFI energy, health index), and drift status."
    ),
)
async def predict(
    req:  PredictRequest,
    _auth: str = Depends(require_api_key),
) -> PredictResponse:
    sig = np.asarray(req.signal, dtype=np.float32)
    return await _run_single_inference(sig)


@router.post(
    "/batch_predict",
    response_model=BatchPredictResponse,
    summary="Batch signal inference",
    description=(
        f"Run inference on up to {settings.MAX_BATCH_SIZE} signals in parallel. "
        "Total batch latency is returned in the X-Batch-Latency-Ms response header."
    ),
)
async def batch_predict(
    req:   BatchPredictRequest,
    _auth: str = Depends(require_api_key),
) -> BatchPredictResponse:
    metrics.record_batch(len(req.signals))
    t0 = time.perf_counter()

    # Each task persists via its own session (see _run_single_inference),
    # so concurrent gather() never shares a single AsyncSession.
    tasks = [
        _run_single_inference(np.asarray(sig, dtype=np.float32))
        for sig in req.signals
    ]
    results = list(await asyncio.gather(*tasks))
    total_ms = (time.perf_counter() - t0) * 1000.0

    log.info(
        "Batch inference complete",
        batch_size=len(results),
        total_ms=round(total_ms, 2),
    )

    return BatchPredictResponse(
        results    = results,
        batch_size = len(results),
        total_ms   = round(total_ms, 3),
    )
