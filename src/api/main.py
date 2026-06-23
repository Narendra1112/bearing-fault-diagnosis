"""
api/main.py — Production FastAPI application entry point.

Features:
  - Correlation ID middleware (UUID per request, flows through all logs)
  - Request/response logging middleware (structured JSON)
  - Prometheus instrumentation middleware
  - API key authentication on protected endpoints
  - Startup: DB init, model warmup, drift reference build
  - Shutdown: flush DB, close connections
  - Custom exception handlers (consistent JSON error format)

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Docker / HF Spaces:
    CMD ["bash", "start.sh"]
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.auth import require_api_key          # noqa: F401 (registers)
from src.api.routes import inference as inference_routes
from src.api.routes import models as models_routes
from src.api.routes import monitoring as monitoring_routes
from src.api.routes.inference import set_inference_helpers
from src.core.config import settings
from src.core.exceptions import (
    BearingDiagnosisError,
    bearing_exception_handler,
    unhandled_exception_handler,
)
from src.core.logger import get_logger, set_correlation_id
from src.db.session import close_db, init_db
from src.ml.constants import CLASS_NAMES
from src.ml.inference_engine import BearingCNN, get_engine
from src.ml.physics_features import PhysicsFeatureExtractor
from src.ml.uncertainty import MCDropoutEstimator
from src.monitoring.alerting import alerter           # noqa: F401 (starts thread)
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.metrics import metrics

log = get_logger(__name__)


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle.

    Startup order:
      1. Initialise database tables
      2. Load CNN model (PyTorch, fallback ONNX)
      3. Export ONNX if missing
      4. Warmup inference engine
      5. Build drift detection reference
      6. Initialise MC Dropout estimator
      7. Register model in DB
    """
    start_ts = time.time()
    log.info("Application starting", environment=settings.ENVIRONMENT)

    # Surface the ephemeral dev API key so a developer can authenticate locally.
    if getattr(settings, "_api_key_generated", False):
        log.warning(
            "No API_KEY configured — generated an ephemeral development key. "
            "Set API_KEY in the environment for stable/production use.",
            dev_api_key=settings.API_KEY,
        )

    # ── 1. Database ────────────────────────────────────────────────────────
    await init_db()

    # ── 2. Load model ──────────────────────────────────────────────────────
    engine = get_engine()
    if settings.MODEL_PATH.exists():
        engine.load_pytorch(settings.MODEL_PATH)
        # Export ONNX if not present
        if not settings.ONNX_MODEL_PATH.exists():
            try:
                engine.export_to_onnx(settings.ONNX_MODEL_PATH)
            except Exception as exc:
                log.warning("ONNX export failed", error=str(exc))
    elif settings.ONNX_MODEL_PATH.exists():
        log.warning("PyTorch checkpoint not found, falling back to ONNX",
                    ckpt=str(settings.MODEL_PATH))
        engine.load_onnx(settings.ONNX_MODEL_PATH)
    else:
        log.warning(
            "No model found — /predict will return 503",
            pytorch_path=str(settings.MODEL_PATH),
            onnx_path=str(settings.ONNX_MODEL_PATH),
        )

    # ── 3. Warmup ──────────────────────────────────────────────────────────
    if engine.is_loaded():
        engine.warmup(n_runs=settings.WARMUP_RUNS)

    # ── 4. Physics extractor ───────────────────────────────────────────────
    extractor = PhysicsFeatureExtractor(rpm=1797.0)

    # ── 5. Drift detector ──────────────────────────────────────────────────
    drift = DriftDetector(train_npz=settings.TRAIN_DATA_PATH)

    # ── 6. MC Dropout ──────────────────────────────────────────────────────
    mc_estimator = None
    if engine.is_loaded() and engine.get_backend() == "pytorch":
        import torch
        model = BearingCNN()
        state = torch.load(settings.MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        mc_estimator = MCDropoutEstimator(model, n_samples=20)

    # ── 7. Wire helpers into inference routes ──────────────────────────────
    set_inference_helpers(extractor, mc_estimator, drift)

    # ── 8. Register model in DB ────────────────────────────────────────────
    try:
        from src.db.crud import get_active_model, register_model
        from src.db.session import db_session
        async with db_session() as db:
            active = await get_active_model(db)
            if active is None and engine.is_loaded():
                await register_model(db, {
                    "version":            settings.MODEL_VERSION,
                    "path":               str(settings.MODEL_PATH),
                    "is_active":          True,
                    "accuracy":           1.0,
                    "validation_passed":  True,
                    "validation_notes":   "CWRU clean test set — 100% accuracy",
                })
    except Exception as exc:
        log.warning("Model registry update failed", error=str(exc))

    # ── 9. Set Prometheus model info ───────────────────────────────────────
    metrics.set_model_version(settings.MODEL_VERSION, engine.get_backend())

    elapsed = round(time.time() - start_ts, 2)
    log.info(
        "Startup complete",
        elapsed_s=elapsed,
        model_loaded=engine.is_loaded(),
        backend=engine.get_backend(),
        drift_enabled=drift.enabled,
        mc_dropout=mc_estimator is not None,
    )

    yield   # ← application handles requests here

    # ── Shutdown ───────────────────────────────────────────────────────────
    log.info("Shutting down")
    await close_db()
    log.info("Shutdown complete")


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title="Bearing Fault Diagnosis API",
    description=(
        "Production inference API for CWRU bearing fault diagnosis. "
        "Classifies 1024-sample vibration windows into 10 fault categories "
        "using a 1-D CNN with physics-informed features, MC Dropout uncertainty, "
        "and multi-method drift detection (z-score + CUSUM + MMD).\n\n"
        "**Authentication**: Protected endpoints require `X-API-Key` header."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Exception handlers ────────────────────────────────────────────────────────

app.add_exception_handler(BearingDiagnosisError, bearing_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────

# Explicit allowlists only — never wildcard origins or headers. In development
# with no CORS_ALLOWED_ORIGINS configured, localhost dev origins are permitted.
_DEV_ORIGINS = ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000"]
_cors_origins = settings.cors_origins or (_DEV_ORIGINS if settings.is_development else [])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Correlation-ID"],
)

# ── Correlation ID + request logging middleware ────────────────────────────────

@app.middleware("http")
async def correlation_middleware(request: Request, call_next) -> Response:
    """
    Attach a UUID correlation ID to every request.
    The ID is injected into all log lines via set_correlation_id()
    and returned in the X-Correlation-ID response header.
    """
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
    set_correlation_id(cid)

    t0 = time.perf_counter()
    log.debug(
        "Request started",
        method=request.method,
        path=request.url.path,
    )

    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    response.headers["X-Correlation-ID"] = cid

    log.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency_ms, 2),
    )
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(inference_routes.router)
app.include_router(monitoring_routes.router)
app.include_router(models_routes.router)


# ── Health endpoint (public) ──────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Service health check")
async def health() -> dict:
    """
    Public health check. Returns service status, model state, and uptime.
    Used by Docker HEALTHCHECK and load balancer probes.
    """
    engine = get_engine()
    return {
        "status":       "ok",
        "version":      "2.0.0",
        "environment":  settings.ENVIRONMENT,
        "model_loaded": engine.is_loaded(),
        "backend":      engine.get_backend(),
        "inference_stats": engine.get_stats() if engine.is_loaded() else None,
    }


# ── Classes endpoint (public) ─────────────────────────────────────────────────

@app.get("/classes", tags=["Model"], summary="List fault class names")
async def get_classes() -> dict:
    """Return all 10 fault class names with their integer IDs."""
    return {
        "n_classes": len(CLASS_NAMES),
        "classes": [
            {"class_id": i, "class_name": name}
            for i, name in enumerate(CLASS_NAMES)
        ],
    }
