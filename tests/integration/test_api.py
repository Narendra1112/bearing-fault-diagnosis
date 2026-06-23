"""
tests/integration/test_api.py — Integration tests for the full API.

Uses FastAPI TestClient (ASGI, no real HTTP server needed).
All tests run against the real application lifespan (model loaded, DB initialised).

Tests:
  - /health returns 200 (public)
  - /classes returns 10 classes (public)
  - /predict without API key returns 401
  - /predict with valid signal returns correct shape
  - /predict with wrong length returns 422
  - /predict with NaN signal returns 422
  - /batch_predict with 32 signals works
  - /batch_predict exceeding MAX_BATCH_SIZE returns 422
  - /drift returns enabled=True after predict calls
  - /drift/reset returns status=reset (auth required)
  - /drift/reset without auth returns 401
  - /metrics/summary returns n_predictions after predict calls
  - /models returns list with at least 1 model
  - X-Correlation-ID header present on all responses
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import settings

API_KEY  = settings.API_KEY
AUTH_HDR = {"X-API-Key": API_KEY}


@pytest.fixture(scope="module")
def client():
    """
    Module-scoped TestClient — runs lifespan once for all tests in this module.
    This loads the model and initialises the DB once, then runs all tests.
    """
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="module")
def real_signal() -> list[float]:
    """Load the first window from the test set."""
    data = np.load(settings.TEST_DATA_PATH)
    return data["X"][0].tolist()


@pytest.fixture(scope="module")
def real_batch(real_signal) -> list[list[float]]:
    """Load 32 windows from the test set for batch testing."""
    data = np.load(settings.TEST_DATA_PATH)
    return [data["X"][i].tolist() for i in range(32)]


# ── Public endpoints ──────────────────────────────────────────────────────────

class TestPublicEndpoints:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["backend"] in ("pytorch", "onnx")

    def test_health_no_auth_required(self, client):
        """Health is public — must work without API key."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_classes_returns_10(self, client):
        r = client.get("/classes")
        assert r.status_code == 200
        body = r.json()
        assert body["n_classes"] == 10
        assert len(body["classes"]) == 10
        assert body["classes"][0]["class_name"] == "normal"

    def test_prometheus_metrics_public(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]
        assert "inference_requests_total" in r.text

    def test_correlation_id_in_response(self, client):
        """All responses must include X-Correlation-ID header."""
        r = client.get("/health")
        assert "x-correlation-id" in r.headers


# ── Authentication ────────────────────────────────────────────────────────────

class TestAuthentication:

    def test_predict_without_key_returns_401(self, client, real_signal):
        r = client.post("/predict", json={"signal": real_signal})
        assert r.status_code == 401
        body = r.json()
        assert body["error"]["code"] == "AUTHENTICATION_FAILED"

    def test_predict_with_wrong_key_returns_401(self, client, real_signal):
        r = client.post(
            "/predict",
            json={"signal": real_signal},
            headers={"X-API-Key": "totally-wrong-key"},
        )
        assert r.status_code == 401

    def test_drift_without_key_returns_401(self, client):
        r = client.get("/drift")
        assert r.status_code == 401

    def test_drift_reset_without_key_returns_401(self, client):
        r = client.post("/drift/reset")
        assert r.status_code == 401

    def test_metrics_summary_without_key_returns_401(self, client):
        r = client.get("/metrics/summary")
        assert r.status_code == 401


# ── Predict ───────────────────────────────────────────────────────────────────

class TestPredict:

    def test_predict_valid_signal(self, client, real_signal):
        r = client.post(
            "/predict",
            json={"signal": real_signal},
            headers=AUTH_HDR,
        )
        assert r.status_code == 200
        body = r.json()
        assert "predicted_class" in body
        assert "confidence" in body
        assert 0.0 <= body["confidence"] <= 1.0
        assert len(body["top3"]) == 3
        assert body["inference_ms"] > 0
        assert body["backend"] in ("pytorch", "onnx")

    def test_predict_includes_physics(self, client, real_signal):
        r = client.post("/predict", json={"signal": real_signal}, headers=AUTH_HDR)
        assert r.status_code == 200
        body = r.json()
        physics = body.get("physics")
        assert physics is not None
        assert "bearing_health_index" in physics
        assert 0.0 <= physics["bearing_health_index"] <= 1.0

    def test_predict_includes_uncertainty(self, client, real_signal):
        r = client.post("/predict", json={"signal": real_signal}, headers=AUTH_HDR)
        assert r.status_code == 200
        unc = r.json().get("uncertainty")
        assert unc is not None
        assert "uncertainty_std" in unc
        assert "is_uncertain" in unc
        assert isinstance(unc["is_uncertain"], bool)

    def test_predict_includes_drift(self, client, real_signal):
        r = client.post("/predict", json={"signal": real_signal}, headers=AUTH_HDR)
        assert r.status_code == 200
        body = r.json()
        assert "drift_warning" in body
        assert "drift_severity" in body
        assert body["drift_severity"] in ("NONE", "WARNING", "CRITICAL")

    def test_predict_wrong_length_returns_422(self, client):
        r = client.post(
            "/predict",
            json={"signal": [0.0] * 512},
            headers=AUTH_HDR,
        )
        assert r.status_code == 422

    def test_predict_nan_signal_returns_422(self, client):
        # JSON spec doesn't support NaN natively, but Python's json.loads does.
        # Send raw bytes with literal NaN so the validator can reject it.
        values = ["0.0"] * 512 + ["NaN"] + ["0.0"] * 511
        body   = ('{"signal": [' + ",".join(values) + "]}").encode()
        r = client.post(
            "/predict",
            content=body,
            headers={**AUTH_HDR, "Content-Type": "application/json"},
        )
        assert r.status_code == 422

    def test_predict_real_window_correct_class(self, client):
        """Verify model predicts correctly on one window per class."""
        data = np.load(settings.TEST_DATA_PATH)
        X, y = data["X"], data["y"]

        CLASS_NAMES = [
            "normal",
            "ball_0.007", "ball_0.014", "ball_0.021",
            "ir_0.007",   "ir_0.014",   "ir_0.021",
            "or_0.007",   "or_0.014",   "or_0.021",
        ]

        for cls_id in range(10):
            idx = int(np.where(y == cls_id)[0][0])
            r   = client.post(
                "/predict",
                json={"signal": X[idx].tolist()},
                headers=AUTH_HDR,
            )
            assert r.status_code == 200
            assert r.json()["predicted_class"] == CLASS_NAMES[cls_id], \
                f"Class {cls_id}: expected {CLASS_NAMES[cls_id]}, got {r.json()['predicted_class']}"


# ── Batch predict ─────────────────────────────────────────────────────────────

class TestBatchPredict:

    def test_batch_predict_32_signals(self, client, real_batch):
        r = client.post(
            "/batch_predict",
            json={"signals": real_batch},
            headers=AUTH_HDR,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["batch_size"] == 32
        assert len(body["results"]) == 32
        assert body["total_ms"] > 0

    def test_batch_predict_exceeds_limit(self, client, real_signal):
        """33 signals (> MAX_BATCH_SIZE=32) should return 422."""
        signals = [real_signal] * 33
        r = client.post(
            "/batch_predict",
            json={"signals": signals},
            headers=AUTH_HDR,
        )
        assert r.status_code == 422

    def test_batch_predict_without_auth(self, client, real_batch):
        r = client.post("/batch_predict", json={"signals": real_batch[:2]})
        assert r.status_code == 401


# ── Monitoring ────────────────────────────────────────────────────────────────

class TestMonitoring:

    def test_drift_status_enabled(self, client):
        r = client.get("/drift", headers=AUTH_HDR)
        assert r.status_code == 200
        body = r.json()
        assert body["enabled"] is True
        assert "training_reference" in body

    def test_drift_history(self, client):
        r = client.get("/drift/history", headers=AUTH_HDR)
        assert r.status_code == 200
        body = r.json()
        assert "events" in body
        assert "count" in body

    def test_metrics_summary_after_predict(self, client, real_signal):
        # Make a prediction first
        client.post("/predict", json={"signal": real_signal}, headers=AUTH_HDR)

        r = client.get("/metrics/summary", headers=AUTH_HDR)
        assert r.status_code == 200
        body = r.json()
        assert body["n_predictions"] >= 1
        assert "class_distribution" in body
        assert "latency_ms" in body

    def test_drift_reset_requires_auth(self, client):
        r = client.post("/drift/reset")
        assert r.status_code == 401

    def test_drift_reset_success(self, client):
        r = client.post("/drift/reset", headers=AUTH_HDR)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "reset"

    def test_bearing_health_endpoint(self, client, real_signal):
        client.post("/predict", json={"signal": real_signal}, headers=AUTH_HDR)
        r = client.get("/health/bearing", headers=AUTH_HDR)
        assert r.status_code == 200
        body = r.json()
        assert "trend" in body

    def test_alerts_endpoint(self, client):
        r = client.get("/alerts/recent", headers=AUTH_HDR)
        assert r.status_code == 200
        body = r.json()
        assert "enabled" in body

    def test_prometheus_contains_inference_counter(self, client, real_signal):
        client.post("/predict", json={"signal": real_signal}, headers=AUTH_HDR)
        r = client.get("/metrics")
        assert "inference_requests_total" in r.text
        assert "inference_latency_seconds" in r.text


# ── Models ────────────────────────────────────────────────────────────────────

class TestModels:

    def test_list_models_returns_registry(self, client):
        r = client.get("/models", headers=AUTH_HDR)
        assert r.status_code == 200
        body = r.json()
        assert "models" in body
        assert body["count"] >= 1

    def test_active_model_info(self, client):
        r = client.get("/models/active", headers=AUTH_HDR)
        assert r.status_code == 200
        body = r.json()
        assert body["model_loaded"] is True
        assert body["backend"] in ("pytorch", "onnx")
