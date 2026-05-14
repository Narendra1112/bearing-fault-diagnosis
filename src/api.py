"""
api.py — Production FastAPI layer for bearing fault diagnosis

Endpoints
---------
  GET  /health   — service health, uptime, model info
  GET  /classes  — all 10 fault class names
  POST /predict  — CNN inference on a 1024-float signal window
  GET  /metrics  — rolling summary of last 100 predictions (MLflow-backed)
  GET  /drift    — current data-drift status vs training distribution

Run locally:
    uvicorn src.api:app --reload --port 8000

Docker:
    see docker/docker-compose.yml
"""

import sys
import time
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from monitor        import PredictionMonitor
from drift_detector import DriftDetector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "normal",
    "ball_0.007", "ball_0.014", "ball_0.021",
    "ir_0.007",   "ir_0.014",   "ir_0.021",
    "or_0.007",   "or_0.014",   "or_0.021",
]

CKPT_PATH    = ROOT / "models" / "best_cnn.pth"
TRAIN_NPZ    = ROOT / "data"   / "processed" / "train.npz"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))

SIGNAL_LENGTH = 1024
N_CLASSES     = len(CLASS_NAMES)   # 10


# ---------------------------------------------------------------------------
# CNN model definition  (must stay in sync with train_cnn.py)
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class BearingCNN(nn.Module):
    def __init__(self, n_classes: int = 10, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(1,   32,  kernel=7, dropout=0.1),
            _ConvBlock(32,  64,  kernel=5, dropout=0.1),
            _ConvBlock(64,  128, kernel=3, dropout=0.2),
            _ConvBlock(128, 256, kernel=3, dropout=0.2),
        )
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


# ---------------------------------------------------------------------------
# Application state  (populated during lifespan startup)
# ---------------------------------------------------------------------------

class _State:
    model:   BearingCNN      | None = None
    monitor: PredictionMonitor | None = None
    drift:   DriftDetector    | None = None
    start_time: float = 0.0
    model_loaded: bool = False


state = _State()


# ---------------------------------------------------------------------------
# Lifespan — load model + helpers once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.start_time = time.time()

    # CNN model
    if not CKPT_PATH.exists():
        print(f"[WARN] Checkpoint not found: {CKPT_PATH}. /predict will return 503.")
    else:
        m = BearingCNN(n_classes=N_CLASSES)
        m.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        m.eval()
        state.model        = m
        state.model_loaded = True
        total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[API] BearingCNN loaded — {total_params:,} parameters.")

    # Prediction monitor (MLflow)
    state.monitor = PredictionMonitor(tracking_uri=TRACKING_URI)
    print(f"[API] PredictionMonitor ready (MLflow URI: {TRACKING_URI})")

    # Drift detector (precomputes training stats — takes ~2 s)
    if TRAIN_NPZ.exists():
        state.drift = DriftDetector(train_npz=TRAIN_NPZ, threshold=2.0)
    else:
        print(f"[WARN] {TRAIN_NPZ} not found — drift detection disabled.")

    print("[API] Startup complete. Listening...")
    yield   # ← application runs here

    print("[API] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Bearing Fault Diagnosis API",
    description=(
        "Production inference layer for the CWRU bearing fault diagnosis project. "
        "Classifies 1024-sample vibration windows into 10 fault categories using a "
        "trained 1-D CNN."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    signal: list[float] = Field(
        ...,
        min_length=SIGNAL_LENGTH,
        max_length=SIGNAL_LENGTH,
        description=f"Exactly {SIGNAL_LENGTH} z-score normalised accelerometer samples.",
    )

    @field_validator("signal")
    @classmethod
    def check_finite(cls, v):
        arr = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Signal must contain only finite (non-NaN, non-Inf) values.")
        return v


class Top3Entry(BaseModel):
    rank:       int
    class_id:   int
    class_name: str
    probability: float


class PredictResponse(BaseModel):
    predicted_class:   str
    class_id:          int
    confidence:        float
    top3:              list[Top3Entry]
    inference_ms:      float
    drift_warning:     bool
    drift_detail:      dict[str, Any] | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Returns service status, uptime, and model load state."""
    return {
        "status":       "ok",
        "model_loaded": state.model_loaded,
        "uptime_s":     round(time.time() - state.start_time, 2),
        "checkpoint":   str(CKPT_PATH.relative_to(ROOT)),
        "n_classes":    N_CLASSES,
        "signal_length": SIGNAL_LENGTH,
    }


@app.get("/classes", tags=["Model"])
def get_classes():
    """Returns all 10 fault class names with their integer IDs."""
    return {
        "n_classes": N_CLASSES,
        "classes": [
            {"class_id": i, "class_name": name}
            for i, name in enumerate(CLASS_NAMES)
        ],
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """
    Run CNN inference on a 1024-sample vibration window.

    - Accepts a JSON body with a `signal` array of exactly 1024 floats.
    - Returns the predicted fault class, confidence, top-3 probabilities,
      inference latency, and a drift warning if the signal statistics are
      unusual relative to training data.
    """
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    sig = np.asarray(req.signal, dtype=np.float32)

    # Inference
    t0      = time.perf_counter()
    x       = torch.from_numpy(sig[np.newaxis, np.newaxis, :])   # (1, 1, 1024)
    with torch.no_grad():
        logits = state.model(x)                                   # (1, 10)
        proba  = torch.softmax(logits, dim=1).numpy()[0]          # (10,)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_id   = int(np.argmax(proba))
    pred_name = CLASS_NAMES[pred_id]
    confidence = float(proba[pred_id])

    # Top-3
    top3_ids = np.argsort(proba)[::-1][:3]
    top3 = [
        Top3Entry(
            rank        = r + 1,
            class_id    = int(i),
            class_name  = CLASS_NAMES[i],
            probability = round(float(proba[i]), 6),
        )
        for r, i in enumerate(top3_ids)
    ]

    # Drift check
    drift_result  = None
    drift_warning = False
    if state.drift is not None:
        drift_result  = state.drift.check(sig)
        drift_warning = bool(drift_result["drifting"])

    # Log to monitor
    state.monitor.log_prediction(
        signal          = sig,
        predicted_class = pred_id,
        class_name      = pred_name,
        confidence      = confidence,
        top3            = [t.model_dump() for t in top3],
        latency_ms      = latency_ms,
    )

    return PredictResponse(
        predicted_class  = pred_name,
        class_id         = pred_id,
        confidence       = round(confidence, 6),
        top3             = top3,
        inference_ms     = round(latency_ms, 3),
        drift_warning    = drift_warning,
        drift_detail     = drift_result,
    )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Returns a rolling summary of the last 100 predictions:
    class distribution, average confidence, average and p95 latency.
    """
    if state.monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialised.")
    return state.monitor.get_summary(n=100)


@app.get("/drift", tags=["Monitoring"])
def drift_status():
    """
    Returns the current drift status from the most recent /predict call,
    plus the training reference statistics used for comparison.
    """
    if state.drift is None:
        return {
            "enabled":  False,
            "reason":   "Training data not found — drift detection disabled.",
        }
    return {
        "enabled":            True,
        "threshold_std":      state.drift.threshold,
        "current_status":     state.drift.status(),
        "training_reference": state.drift.training_reference(),
    }
