"""
monitor.py — Prediction monitoring with MLflow and in-memory rolling store

Every call to log_prediction():
  - appends a record to a thread-safe deque (maxlen=100)
  - fires an async background MLflow run (non-blocking; silently skipped if
    the tracking server is unavailable)

get_summary(n) returns a JSON-serialisable dict over the last n records.
"""

import os
import threading
import numpy as np
from collections import Counter, deque
from datetime import datetime, timezone
from scipy.stats import kurtosis as _kurtosis

import mlflow

EXPERIMENT_NAME = "bearing_fault_predictions"
TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


class PredictionMonitor:
    """Thread-safe rolling store + MLflow logger for inference events."""

    def __init__(self, tracking_uri: str = TRACKING_URI, maxlen: int = 100):
        self._lock   = threading.Lock()
        self._recent: deque[dict] = deque(maxlen=maxlen)

        # MLflow setup — failure here must not crash the API
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)
            self._mlflow_ok = True
        except Exception:
            self._mlflow_ok = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        signal:          np.ndarray,
        predicted_class: int,
        class_name:      str,
        confidence:      float,
        top3:            list[dict],
        latency_ms:      float,
    ) -> None:
        """Record one inference event; MLflow logging happens in a daemon thread."""
        sig    = np.asarray(signal, dtype=np.float64)
        record = {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "predicted_class":  predicted_class,
            "class_name":       class_name,
            "confidence":       round(float(confidence), 6),
            "latency_ms":       round(float(latency_ms), 3),
            # Signal statistics used for drift monitoring
            "signal_mean":      round(float(sig.mean()), 6),
            "signal_std":       round(float(sig.std()),  6),
            "signal_kurtosis":  round(float(_kurtosis(sig, fisher=True)), 6),
            "signal_rms":       round(float(np.sqrt(np.mean(sig ** 2))), 6),
            "signal_peak2peak": round(float(np.ptp(sig)), 6),
        }

        with self._lock:
            self._recent.append(record)

        # Fire-and-forget MLflow logging
        if self._mlflow_ok:
            t = threading.Thread(target=self._mlflow_log, args=(record,), daemon=True)
            t.start()

    def get_summary(self, n: int = 100) -> dict:
        """Summarise the last *n* prediction records."""
        with self._lock:
            recent = list(self._recent)[-n:]

        if not recent:
            return {"n_predictions": 0, "message": "No predictions recorded yet."}

        confidences = [r["confidence"]  for r in recent]
        latencies   = [r["latency_ms"]  for r in recent]
        class_dist  = Counter(r["class_name"] for r in recent)

        return {
            "n_predictions":       len(recent),
            "class_distribution":  dict(class_dist),
            "average_confidence":  round(float(np.mean(confidences)), 4),
            "min_confidence":      round(float(np.min(confidences)),  4),
            "max_confidence":      round(float(np.max(confidences)),  4),
            "average_latency_ms":  round(float(np.mean(latencies)),   3),
            "p95_latency_ms":      round(float(np.percentile(latencies, 95)), 3),
            "last_prediction_at":  recent[-1]["timestamp"],
            "first_prediction_at": recent[0]["timestamp"],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mlflow_log(self, record: dict) -> None:
        """Run in a daemon thread; silently swallow any MLflow errors."""
        try:
            with mlflow.start_run(run_name=f"pred_{record['timestamp']}"):
                mlflow.log_params({
                    "predicted_class": record["predicted_class"],
                    "class_name":      record["class_name"],
                })
                mlflow.log_metrics({
                    k: record[k]
                    for k in (
                        "confidence", "latency_ms",
                        "signal_mean", "signal_std",
                        "signal_kurtosis", "signal_rms", "signal_peak2peak",
                    )
                })
        except Exception:
            pass
