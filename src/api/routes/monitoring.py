"""
api/routes/monitoring.py — Monitoring, drift, and metrics endpoints.

GET  /metrics/summary     — rolling stats from DB (last 100 predictions)
GET  /drift               — current drift status with CUSUM values
GET  /drift/history       — last 24h drift events from DB
GET  /health/bearing      — bearing health trend
POST /drift/reset         — reset CUSUM accumulators (admin)
GET  /alerts/recent       — alerter delivery stats
GET  /metrics             — Prometheus text format (public)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.auth import require_api_key
import src.api.routes.inference as _inference_module   # access _drift_detector dynamically
from src.core.logger import get_logger
from src.db.crud import (
    get_drift_history,
    get_prediction_summary,
    get_recent_predictions,
    resolve_drift_events,
)
from src.db.session import get_db
from src.monitoring.alerting import alerter
from src.monitoring.metrics import metrics

log = get_logger(__name__)
router = APIRouter(tags=["Monitoring"])


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus text-format metrics. Public endpoint — no auth required.",
    include_in_schema=True,
)
async def prometheus_metrics() -> Response:
    """Return Prometheus metrics in text exposition format."""
    content, content_type = metrics.generate_text()
    return Response(content=content, media_type=content_type)


@router.get(
    "/metrics/summary",
    summary="Rolling prediction summary",
    description="Summary of the last 100 predictions: class distribution, "
                "avg confidence, p95 latency, drift rate.",
)
async def metrics_summary(
    db:    AsyncSession = Depends(get_db),
    _auth: str = Depends(require_api_key),
) -> dict:
    summary = await get_prediction_summary(db, n=100)
    log.debug("Metrics summary fetched", n_predictions=summary.get("n_predictions"))
    return summary


@router.get(
    "/drift",
    summary="Current drift status",
    description="Current drift detection status including z-scores, "
                "CUSUM accumulator values, and MMD score.",
)
async def drift_status(
    _auth: str = Depends(require_api_key),
) -> dict:
    dd = _inference_module._drift_detector
    if dd is None or not dd.enabled:
        return {
            "enabled": False,
            "reason":  "Drift detector not initialised or training data not found.",
        }

    return {
        "enabled":            True,
        "threshold_zscore":   dd.threshold,
        "threshold_cusum":    dd.cusum_h,
        "current_status":     dd.status(),
        "cusum_state":        dd.cusum_status(),
        "training_reference": dd.training_reference(),
    }


@router.get(
    "/drift/history",
    summary="Drift event history",
    description="All drift events detected in the last 24 hours.",
)
async def drift_history(
    hours: int = 24,
    db:    AsyncSession = Depends(get_db),
    _auth: str = Depends(require_api_key),
) -> dict:
    events = await get_drift_history(db, hours=hours)
    return {
        "hours":  hours,
        "count":  len(events),
        "events": [
            {
                "id":               e.id,
                "timestamp":        e.timestamp.isoformat() if e.timestamp else None,
                "feature_name":     e.feature_name,
                "z_score":          e.z_score,
                "cusum_value":      e.cusum_value,
                "severity":         e.severity,
                "detection_method": e.detection_method,
                "resolved_at":      e.resolved_at.isoformat() if e.resolved_at else None,
            }
            for e in events
        ],
    }


@router.get(
    "/health/bearing",
    summary="Bearing health trend",
    description="Health index from the last N predictions, ordered newest first.",
)
async def bearing_health(
    n:     int = 50,
    db:    AsyncSession = Depends(get_db),
    _auth: str = Depends(require_api_key),
) -> dict:
    records = await get_recent_predictions(db, n=n)
    health_series = [
        {
            "timestamp":        r.timestamp.isoformat() if r.timestamp else None,
            "predicted_class":  r.predicted_class,
            "bearing_health":   r.bearing_health_index,
            "envelope_kurtosis": r.envelope_kurtosis,
        }
        for r in records
        if r.bearing_health_index is not None
    ]

    avg_health = (
        sum(h["bearing_health"] for h in health_series) / len(health_series)
        if health_series else None
    )

    return {
        "n_records":   len(health_series),
        "avg_health":  round(avg_health, 4) if avg_health is not None else None,
        "trend":       health_series,
    }


@router.post(
    "/drift/reset",
    summary="Reset CUSUM accumulators",
    description="Reset CUSUM drift accumulators and sliding MMD window. "
                "Call after investigating and resolving a confirmed drift event.",
)
async def drift_reset(
    db:    AsyncSession = Depends(get_db),
    _auth: str = Depends(require_api_key),
) -> dict:
    dd = _inference_module._drift_detector
    if dd is not None:
        dd.reset()

    resolved_count = await resolve_drift_events(db)
    log.info("Drift reset via API", resolved_events=resolved_count)

    return {
        "status":           "reset",
        "cusum_cleared":    dd is not None,
        "events_resolved":  resolved_count,
    }


@router.get(
    "/alerts/recent",
    summary="Recent alert delivery stats",
    description="Stats from the webhook alerter: sent, failed, queue size.",
)
async def alerts_recent(
    _auth: str = Depends(require_api_key),
) -> dict:
    return alerter.get_stats()
