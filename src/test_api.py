"""
test_api.py — Smoke-test every endpoint of the FastAPI inference server.

Run AFTER starting the server:
    uvicorn src.api:app --port 8000

Then:
    python src/test_api.py
"""

import sys
import time
import json
import numpy as np
import requests
from pathlib import Path

BASE = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SEP  = "=" * 60
SEP2 = "-" * 60


def _print(label: str, resp: requests.Response) -> dict:
    data = resp.json()
    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  Status : {resp.status_code}  ({resp.elapsed.total_seconds()*1000:.1f} ms)")
    print(SEP2)
    print(json.dumps(data, indent=2))
    return data


def wait_for_server(retries: int = 15, delay: float = 2.0) -> bool:
    print(f"Waiting for server at {BASE} ...")
    for i in range(retries):
        try:
            r = requests.get(f"{BASE}/health", timeout=3)
            if r.status_code == 200:
                print(f"  Server ready after {i * delay:.0f}s\n")
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def test_health():
    r = requests.get(f"{BASE}/health")
    d = _print("GET /health", r)
    assert r.status_code == 200
    assert d["status"] == "ok"
    assert d["model_loaded"] is True
    print("  [PASS]")


def test_classes():
    r = requests.get(f"{BASE}/classes")
    d = _print("GET /classes", r)
    assert r.status_code == 200
    assert d["n_classes"] == 10
    assert len(d["classes"]) == 10
    print("  [PASS]")


def test_predict_real_window():
    """Load an actual test window from data/processed/test.npz."""
    data   = np.load(ROOT / "data" / "processed" / "test.npz")
    X_test = data["X"]
    y_test = data["y"]

    CLASS_NAMES = [
        "normal",
        "ball_0.007", "ball_0.014", "ball_0.021",
        "ir_0.007",   "ir_0.014",   "ir_0.021",
        "or_0.007",   "or_0.014",   "or_0.021",
    ]

    # Pick one window of each class to exercise all code paths
    for cls_id in range(10):
        idx    = int(np.where(y_test == cls_id)[0][0])
        window = X_test[idx].tolist()

        r = requests.post(f"{BASE}/predict", json={"signal": window})
        d = _print(f"POST /predict  [true={CLASS_NAMES[cls_id]}]", r)
        assert r.status_code == 200
        assert d["predicted_class"] == CLASS_NAMES[cls_id], (
            f"Wrong prediction: got {d['predicted_class']}, expected {CLASS_NAMES[cls_id]}"
        )
        assert len(d["top3"]) == 3
        assert d["inference_ms"] > 0
        print(f"  predicted={d['predicted_class']}  "
              f"confidence={d['confidence']:.4f}  "
              f"latency={d['inference_ms']:.2f}ms  "
              f"drift={d['drift_warning']}")
        print("  [PASS]")


def test_predict_noisy_signal():
    """Send pure white noise — should trigger drift warning on kurtosis."""
    rng    = np.random.default_rng(99)
    # Gaussian noise has kurtosis ~0, vs training mean ~2.1 — may or may not drift
    signal = rng.standard_normal(1024).tolist()

    r = requests.post(f"{BASE}/predict", json={"signal": signal})
    d = _print("POST /predict  [synthetic Gaussian noise]", r)
    assert r.status_code == 200
    print(f"  drift_warning={d['drift_warning']}")
    if d["drift_detail"]:
        for feat, info in d["drift_detail"]["flags"].items():
            print(f"    {feat}: z={info['z_score']}  drifting={info['drifting']}")
    print("  [PASS]")


def test_predict_impulsive_signal():
    """Send a highly impulsive signal — expect drift warning (kurtosis spike)."""
    sig = np.zeros(1024)
    # Place sharp impulses every 64 samples — mimics severe fault
    sig[::64] = 10.0
    # Z-score normalise (as the API expects)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)

    r = requests.post(f"{BASE}/predict", json={"signal": sig.tolist()})
    d = _print("POST /predict  [synthetic impulsive signal]", r)
    assert r.status_code == 200
    print(f"  drift_warning={d['drift_warning']}")
    print("  [PASS]")


def test_predict_validation_error():
    """Wrong signal length must return HTTP 422."""
    r = requests.post(f"{BASE}/predict", json={"signal": [0.0] * 512})
    print(f"\n{SEP}")
    print("  POST /predict  [wrong length — expect 422]")
    print(SEP2)
    print(f"  Status: {r.status_code}")
    assert r.status_code == 422
    print("  [PASS]")


def test_metrics():
    r = requests.get(f"{BASE}/metrics")
    d = _print("GET /metrics", r)
    assert r.status_code == 200
    assert d["n_predictions"] >= 10    # we called /predict 10+ times above
    print("  [PASS]")


def test_drift():
    r = requests.get(f"{BASE}/drift")
    d = _print("GET /drift", r)
    assert r.status_code == 200
    assert d["enabled"] is True
    assert "training_reference" in d
    print("  [PASS]")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not wait_for_server():
        print("ERROR: Server did not start within timeout. Aborting.")
        sys.exit(1)

    tests = [
        ("Health check",               test_health),
        ("Class list",                 test_classes),
        ("Predict — real test windows", test_predict_real_window),
        ("Predict — Gaussian noise",   test_predict_noisy_signal),
        ("Predict — impulsive signal", test_predict_impulsive_signal),
        ("Predict — validation error", test_predict_validation_error),
        ("Metrics summary",            test_metrics),
        ("Drift status",               test_drift),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            failed += 1

    print(f"\n{SEP}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(SEP)
    sys.exit(0 if failed == 0 else 1)
