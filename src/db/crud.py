"""
db/crud.py — Database CRUD operations for predictions, drift events, and models.

All functions are async and accept an AsyncSession from the session factory.
They do NOT commit — the session context manager in session.py handles that.

Usage:
    from src.db.crud import save_prediction
    async with db_session() as db:
        record = await save_prediction(db, prediction_data)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import DatabaseError
from src.core.logger import get_logger
from src.db.models import DriftEvent, ModelRecord, PredictionRecord

log = get_logger(__name__)


# ── Predictions ───────────────────────────────────────────────────────────────

async def save_prediction(
    db: AsyncSession,
    data: dict[str, Any],
) -> PredictionRecord:
    """
    Persist a prediction result to the database.

    Args:
        db:   Active AsyncSession.
        data: Dict with keys matching PredictionRecord columns.

    Returns:
        The created PredictionRecord (id populated after flush).

    Raises:
        DatabaseError: On any SQLAlchemy error.
    """
    try:
        record = PredictionRecord(**data)
        db.add(record)
        await db.flush()   # populate id without committing
        log.debug(
            "Prediction saved",
            id=record.id,
            predicted_class=record.predicted_class,
        )
        return record
    except Exception as exc:
        log.error("Failed to save prediction", error=str(exc))
        raise DatabaseError(f"Failed to save prediction: {exc}") from exc


async def get_recent_predictions(
    db: AsyncSession,
    n: int = 100,
) -> list[PredictionRecord]:
    """
    Retrieve the N most recent prediction records, newest first.

    Args:
        db: Active AsyncSession.
        n:  Number of records to return (default 100).

    Returns:
        List of PredictionRecord ordered by timestamp descending.
    """
    try:
        result = await db.execute(
            select(PredictionRecord)
            .order_by(PredictionRecord.timestamp.desc())
            .limit(n)
        )
        return list(result.scalars().all())
    except Exception as exc:
        log.error("Failed to fetch predictions", error=str(exc))
        raise DatabaseError(f"Failed to fetch predictions: {exc}") from exc


async def get_prediction_summary(
    db: AsyncSession,
    n: int = 100,
) -> dict[str, Any]:
    """
    Compute a rolling summary of the last N predictions:
    class distribution, avg confidence, avg/p95 latency, drift rate.
    """
    records = await get_recent_predictions(db, n)
    if not records:
        return {"n_predictions": 0}

    latencies   = [r.latency_ms for r in records]
    confidences = [r.confidence for r in records]
    drift_count = sum(1 for r in records if r.drift_warning)

    class_dist: dict[str, int] = {}
    for r in records:
        class_dist[r.predicted_class] = class_dist.get(r.predicted_class, 0) + 1

    import numpy as np
    return {
        "n_predictions":      len(records),
        "class_distribution": class_dist,
        "avg_confidence":     round(float(sum(confidences) / len(confidences)), 4),
        "latency_ms": {
            "avg": round(float(sum(latencies) / len(latencies)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3),
            "p99": round(float(np.percentile(latencies, 99)), 3),
        },
        "drift_rate": round(drift_count / len(records), 4),
        "drift_count": drift_count,
    }


# ── Drift events ──────────────────────────────────────────────────────────────

async def save_drift_event(
    db: AsyncSession,
    data: dict[str, Any],
) -> DriftEvent:
    """
    Persist a drift detection event.

    Args:
        db:   Active AsyncSession.
        data: Dict with keys matching DriftEvent columns.

    Returns:
        Created DriftEvent record.
    """
    try:
        event = DriftEvent(**data)
        db.add(event)
        await db.flush()
        log.info(
            "Drift event saved",
            id=event.id,
            feature=event.feature_name,
            method=event.detection_method,
            severity=event.severity,
        )
        return event
    except Exception as exc:
        log.error("Failed to save drift event", error=str(exc))
        raise DatabaseError(f"Failed to save drift event: {exc}") from exc


async def get_drift_history(
    db: AsyncSession,
    hours: int = 24,
) -> list[DriftEvent]:
    """
    Retrieve drift events from the last N hours.

    Args:
        db:    Active AsyncSession.
        hours: Lookback window in hours (default 24).

    Returns:
        List of DriftEvent records ordered by timestamp descending.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    try:
        result = await db.execute(
            select(DriftEvent)
            .where(DriftEvent.timestamp >= cutoff)
            .order_by(DriftEvent.timestamp.desc())
        )
        return list(result.scalars().all())
    except Exception as exc:
        log.error("Failed to fetch drift history", error=str(exc))
        raise DatabaseError(f"Failed to fetch drift history: {exc}") from exc


async def resolve_drift_events(
    db: AsyncSession,
) -> int:
    """
    Mark all unresolved drift events as resolved (sets resolved_at = now).
    Called when POST /drift/reset is invoked.

    Returns:
        Number of events resolved.
    """
    now = datetime.now(timezone.utc)
    try:
        result = await db.execute(
            update(DriftEvent)
            .where(DriftEvent.resolved_at.is_(None))
            .values(resolved_at=now)
        )
        count = result.rowcount
        log.info("Drift events resolved", count=count)
        return count
    except Exception as exc:
        log.error("Failed to resolve drift events", error=str(exc))
        raise DatabaseError(f"Failed to resolve drift events: {exc}") from exc


# ── Model registry ────────────────────────────────────────────────────────────

async def register_model(
    db: AsyncSession,
    data: dict[str, Any],
) -> ModelRecord:
    """
    Register a new model version.

    Args:
        db:   Active AsyncSession.
        data: Dict with keys matching ModelRecord columns.

    Returns:
        Created ModelRecord.
    """
    try:
        record = ModelRecord(**data)
        db.add(record)
        await db.flush()
        log.info(
            "Model registered",
            version=record.version,
            accuracy=record.accuracy,
        )
        return record
    except Exception as exc:
        log.error("Failed to register model", error=str(exc))
        raise DatabaseError(f"Failed to register model: {exc}") from exc


async def get_active_model(db: AsyncSession) -> ModelRecord | None:
    """
    Return the currently active model record, or None if no model is registered.
    """
    try:
        result = await db.execute(
            select(ModelRecord).where(ModelRecord.is_active.is_(True)).limit(1)
        )
        return result.scalar_one_or_none()
    except Exception as exc:
        log.error("Failed to fetch active model", error=str(exc))
        raise DatabaseError(f"Failed to fetch active model: {exc}") from exc


async def get_all_models(db: AsyncSession) -> list[ModelRecord]:
    """Return all registered model versions ordered by creation date."""
    try:
        result = await db.execute(
            select(ModelRecord).order_by(ModelRecord.created_at.desc())
        )
        return list(result.scalars().all())
    except Exception as exc:
        log.error("Failed to fetch models", error=str(exc))
        raise DatabaseError(f"Failed to fetch models: {exc}") from exc


async def activate_model(
    db: AsyncSession,
    version: str,
) -> ModelRecord:
    """
    Activate a model version — deactivates all others first.

    Args:
        db:      Active AsyncSession.
        version: Version string to activate.

    Returns:
        The activated ModelRecord.

    Raises:
        DatabaseError: If version is not found.
    """
    try:
        # Deactivate all
        await db.execute(
            update(ModelRecord).values(is_active=False)
        )
        # Activate target
        result = await db.execute(
            select(ModelRecord).where(ModelRecord.version == version)
        )
        record = result.scalar_one_or_none()
        if record is None:
            raise DatabaseError(f"Model version '{version}' not found in registry.")
        record.is_active = True
        await db.flush()
        log.info("Model activated", version=version)
        return record
    except DatabaseError:
        raise
    except Exception as exc:
        log.error("Failed to activate model", version=version, error=str(exc))
        raise DatabaseError(f"Failed to activate model: {exc}") from exc
