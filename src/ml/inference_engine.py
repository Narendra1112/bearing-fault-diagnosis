"""
ml/inference_engine.py — Dual-mode inference: PyTorch primary + ONNX fallback.

The InferenceEngine wraps the BearingCNN and exposes a clean predict() API
that is backend-agnostic. On first load it tries PyTorch; if the checkpoint
is missing it can fall back to ONNX for edge deployment.

Features:
  - Warmup run at startup (prevents cold-start latency spikes)
  - Thread-safe inference via threading.Lock
  - Per-call latency tracking with p50/p95/p99 reporting
  - ONNX export utility (for edge / Triton deployment)
  - Async batch inference with asyncio

Usage:
    from src.ml.inference_engine import InferenceEngine
    engine = InferenceEngine()
    engine.load_pytorch(settings.MODEL_PATH)
    engine.warmup()
    result = engine.predict(signal_array)
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from src.core.config import settings
from src.core.exceptions import InferenceTimeoutError, ModelNotLoadedError
from src.core.logger import get_logger
from src.ml.constants import CLASS_NAMES, N_CLASSES

log = get_logger(__name__)


# ── BearingCNN definition (must stay in sync with train_cnn.py) ───────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BearingCNN(nn.Module):
    """
    1-D CNN for bearing fault classification.
    Architecture: 4 ConvBlocks -> GlobalAveragePool -> 2-layer FC head.
    """

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.4):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.gap(self.features(x)))


# ── Prediction result dataclass ───────────────────────────────────────────────

class PredictionResult:
    """Structured result from a single-signal inference call."""

    __slots__ = (
        "predicted_class", "class_id", "confidence",
        "probabilities", "top3", "latency_ms", "backend",
    )

    def __init__(
        self,
        predicted_class: str,
        class_id: int,
        confidence: float,
        probabilities: np.ndarray,
        latency_ms: float,
        backend: str,
    ) -> None:
        self.predicted_class = predicted_class
        self.class_id        = class_id
        self.confidence      = confidence
        self.probabilities   = probabilities
        self.latency_ms      = latency_ms
        self.backend         = backend

        top3_ids = np.argsort(probabilities)[::-1][:3]
        self.top3 = [
            {
                "rank":        r + 1,
                "class_id":    int(i),
                "class_name":  CLASS_NAMES[i],
                "probability": round(float(probabilities[i]), 6),
            }
            for r, i in enumerate(top3_ids)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "class_id":        self.class_id,
            "confidence":      round(self.confidence, 6),
            "top3":            self.top3,
            "latency_ms":      round(self.latency_ms, 3),
            "backend":         self.backend,
        }


# ── Latency tracker ───────────────────────────────────────────────────────────

class _LatencyTracker:
    """Thread-safe rolling window for latency percentile computation."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._window: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, ms: float) -> None:
        with self._lock:
            self._window.append(ms)

    def percentiles(self) -> dict[str, float]:
        # All reads + computation under the lock so the percentile values and
        # the reported count come from a single consistent window snapshot.
        with self._lock:
            if not self._window:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}
            arr = np.array(self._window)
            return {
                "p50":   round(float(np.percentile(arr, 50)), 3),
                "p95":   round(float(np.percentile(arr, 95)), 3),
                "p99":   round(float(np.percentile(arr, 99)), 3),
                "count": len(arr),
            }


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Backend-agnostic inference engine for BearingCNN.

    Supports two backends:
      - "pytorch"  — native PyTorch (fp32, CPU)
      - "onnx"     — ONNX Runtime (faster on CPU, edge-deployable)

    Thread-safe: all predict() calls are serialised with a lock.
    """

    def __init__(self) -> None:
        self._model: BearingCNN | None = None
        self._ort_session: Any | None = None   # onnxruntime.InferenceSession
        self._backend: Literal["pytorch", "onnx", "none"] = "none"
        self._lock = threading.Lock()
        self._latency = _LatencyTracker()
        self._n_predictions: int = 0
        self.device = torch.device("cpu")

    # ── Loading ───────────────────────────────────────────────────────────

    def load_pytorch(self, path: Path | str) -> None:
        """
        Load a BearingCNN checkpoint from disk.

        Args:
            path: Path to the .pth state-dict file.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        model = BearingCNN(n_classes=N_CLASSES)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        with self._lock:
            self._model   = model
            self._backend = "pytorch"

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(
            "PyTorch model loaded",
            path=str(path),
            n_params=n_params,
            backend="pytorch",
        )

    def load_onnx(self, path: Path | str) -> None:
        """
        Load an ONNX model for CPU inference.

        Args:
            path: Path to the .onnx file.

        Raises:
            ImportError: If onnxruntime is not installed.
            FileNotFoundError: If the ONNX file does not exist.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            ) from exc

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2

        session = ort.InferenceSession(str(path), sess_options=opts)

        with self._lock:
            self._ort_session = session
            self._backend     = "onnx"

        log.info("ONNX model loaded", path=str(path), backend="onnx")

    # ── ONNX export ───────────────────────────────────────────────────────

    def export_to_onnx(
        self,
        output_path: Path | str,
        signal_length: int | None = None,
    ) -> Path:
        """
        Export the loaded PyTorch model to ONNX format.

        Args:
            output_path: Destination .onnx file path.
            signal_length: Expected input length (default from settings).

        Returns:
            Path to the saved ONNX file.

        Raises:
            ModelNotLoadedError: If no PyTorch model is loaded.
        """
        if self._model is None:
            raise ModelNotLoadedError("PyTorch model must be loaded before ONNX export.")

        sig_len = signal_length or settings.SIGNAL_LENGTH
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.zeros(1, 1, sig_len)
        with torch.no_grad():
            torch.onnx.export(
                self._model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=17,
                input_names=["signal"],
                output_names=["logits"],
                dynamic_axes={
                    "signal": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                },
                verbose=False,
            )

        log.info("ONNX export complete", path=str(output_path), opset=17)
        return output_path

    # ── Warmup ────────────────────────────────────────────────────────────

    def warmup(self, n_runs: int | None = None) -> None:
        """
        Run dummy inferences to JIT-warm the model and stabilise latency.

        Args:
            n_runs: Number of warmup runs (default from settings).
        """
        n = n_runs or settings.WARMUP_RUNS
        if self._backend == "none":
            log.warning("Warmup called but no model is loaded")
            return

        dummy = np.zeros(settings.SIGNAL_LENGTH, dtype=np.float32)
        log.info("Starting inference warmup", n_runs=n, backend=self._backend)
        for _ in range(n):
            self.predict(dummy)

        # Clear warmup latencies from tracker so they don't pollute real stats
        self._latency = _LatencyTracker()
        log.info("Warmup complete", n_runs=n)

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, signal: np.ndarray) -> PredictionResult:
        """
        Run inference on a single 1024-sample signal.

        Args:
            signal: Float array of shape (SIGNAL_LENGTH,).

        Returns:
            PredictionResult with class, confidence, top3, latency.

        Raises:
            ModelNotLoadedError: If no model is loaded.
            InferenceTimeoutError: If inference exceeds configured timeout.
        """
        if self._backend == "none":
            raise ModelNotLoadedError()

        sig = np.asarray(signal, dtype=np.float32)
        t0  = time.perf_counter()

        if self._backend == "pytorch":
            proba = self._predict_pytorch(sig)
        else:
            proba = self._predict_onnx(sig)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if elapsed_ms > settings.INFERENCE_TIMEOUT_S * 1000:
            raise InferenceTimeoutError(
                f"Inference took {elapsed_ms:.1f}ms, limit is "
                f"{settings.INFERENCE_TIMEOUT_S * 1000:.0f}ms"
            )

        self._latency.record(elapsed_ms)
        self._n_predictions += 1

        pred_id   = int(np.argmax(proba))
        pred_name = CLASS_NAMES[pred_id]

        log.debug(
            "Inference complete",
            predicted=pred_name,
            confidence=round(float(proba[pred_id]), 4),
            latency_ms=round(elapsed_ms, 2),
            backend=self._backend,
        )

        return PredictionResult(
            predicted_class=pred_name,
            class_id=pred_id,
            confidence=float(proba[pred_id]),
            probabilities=proba,
            latency_ms=elapsed_ms,
            backend=self._backend,
        )

    def _predict_pytorch(self, sig: np.ndarray) -> np.ndarray:
        """Run PyTorch inference. Called with lock from predict()."""
        x = torch.from_numpy(sig[np.newaxis, np.newaxis, :])   # (1, 1, 1024)
        with self._lock:
            # Defensive: ensure eval() mode (BatchNorm/Dropout deterministic)
            # in case any other code path left the model in train() mode.
            self._model.eval()
            with torch.no_grad():
                logits = self._model(x)                         # (1, 10)
        return torch.softmax(logits, dim=1).numpy()[0]          # (10,)

    def _predict_onnx(self, sig: np.ndarray) -> np.ndarray:
        """Run ONNX Runtime inference."""
        x = sig[np.newaxis, np.newaxis, :]                      # (1, 1, 1024)
        with self._lock:
            outputs = self._ort_session.run(
                None, {"signal": x}
            )
        logits = torch.from_numpy(outputs[0])
        return torch.softmax(logits, dim=1).numpy()[0]

    # ── Batch inference ───────────────────────────────────────────────────

    async def predict_batch(
        self,
        signals: list[np.ndarray],
    ) -> list[PredictionResult]:
        """
        Async batch inference — runs each signal in executor to avoid
        blocking the event loop.

        Args:
            signals: List of 1-D signal arrays.

        Returns:
            List of PredictionResult, one per signal.
        """
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self.predict, sig)
            for sig in signals
        ]
        return list(await asyncio.gather(*tasks))

    # ── Status ────────────────────────────────────────────────────────────

    def get_backend(self) -> str:
        """Return which backend is currently active: 'pytorch', 'onnx', or 'none'."""
        return self._backend

    def is_loaded(self) -> bool:
        """Return True if any model backend is ready."""
        return self._backend != "none"

    def get_stats(self) -> dict[str, Any]:
        """Return inference statistics including latency percentiles."""
        return {
            "backend":       self._backend,
            "n_predictions": self._n_predictions,
            "latency":       self._latency.percentiles(),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    """
    Return the global InferenceEngine singleton.
    The engine is loaded during application startup (lifespan).
    """
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
