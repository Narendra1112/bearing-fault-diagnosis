"""
api/routes/models.py — Model registry endpoints.

GET  /models                      — list all registered models
GET  /models/active               — currently active model info + stats
POST /models/{version}/activate   — switch active model (admin)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.auth import require_api_key
from src.core.exceptions import DatabaseError
from src.core.logger import get_logger
from src.db.crud import activate_model, get_active_model, get_all_models
from src.db.session import get_db
from src.ml.inference_engine import get_engine

log = get_logger(__name__)
router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    summary="List all model versions",
    description="Returns all models registered in the model registry with "
                "their accuracy metrics and validation status.",
)
async def list_models(
    db:    AsyncSession = Depends(get_db),
    _auth: str = Depends(require_api_key),
) -> dict:
    records = await get_all_models(db)
    return {
        "count":  len(records),
        "models": [
            {
                "id":                       r.id,
                "version":                  r.version,
                "path":                     r.path,
                "is_active":                r.is_active,
                "accuracy":                 r.accuracy,
                "per_class_min_accuracy":   r.per_class_min_accuracy,
                "cross_condition_accuracy": r.cross_condition_accuracy,
                "noise_10db_accuracy":      r.noise_10db_accuracy,
                "inference_p95_ms":         r.inference_p95_ms,
                "validation_passed":        r.validation_passed,
                "created_at":               r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
    }


@router.get(
    "/active",
    summary="Active model info",
    description="Returns the currently loaded model version, backend, and "
                "live inference statistics.",
)
async def active_model(
    db:    AsyncSession = Depends(get_db),
    _auth: str = Depends(require_api_key),
) -> dict:
    engine = get_engine()
    record = await get_active_model(db)

    return {
        "backend":       engine.get_backend(),
        "model_loaded":  engine.is_loaded(),
        "inference_stats": engine.get_stats(),
        "registry": {
            "version":                  record.version if record else None,
            "accuracy":                 record.accuracy if record else None,
            "cross_condition_accuracy": record.cross_condition_accuracy if record else None,
            "validation_passed":        record.validation_passed if record else None,
        } if record else None,
    }


@router.post(
    "/{version}/activate",
    summary="Activate a model version",
    description="Switch the active model version in the registry. "
                "Does NOT reload the in-memory model — restart the service to apply.",
)
async def activate_model_version(
    version: str,
    db:      AsyncSession = Depends(get_db),
    _auth:   str = Depends(require_api_key),
) -> dict:
    try:
        record = await activate_model(db, version)
        log.info("Model version activated via API", version=version)
        return {
            "status":  "activated",
            "version": record.version,
            "note":    "Restart the service to load the new model into memory.",
        }
    except DatabaseError as exc:
        # Log the specific reason internally; return a generic message so the
        # endpoint cannot be used to enumerate which versions exist.
        log.warning("Model activation failed", version=version, error=str(exc))
        return {"status": "error", "message": "Model activation failed."}
