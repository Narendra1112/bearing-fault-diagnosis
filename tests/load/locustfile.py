"""
tests/load/locustfile.py — Locust load test for the bearing inference API.

Target: p95 latency < 200ms at 50 concurrent users.

Run:
    pip install locust
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 1 --run-time 60s --headless \
           --csv outputs/reports/load_test

Or with the Locust web UI:
    locust -f tests/load/locustfile.py --host http://localhost:8000

User behaviour:
  - 70% single /predict requests  (primary inference path)
  - 20% GET /health               (monitoring / load balancer probes)
  - 10% GET /drift                (monitoring checks)

Ramp: 1 → 50 users over 30 seconds (spawn-rate=1.67/s).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from locust import HttpUser, between, events, task

ROOT    = Path(__file__).resolve().parent.parent.parent
API_KEY = os.getenv("API_KEY", "dev-insecure-key-change-in-production")
HEADERS = {
    "X-API-Key":    API_KEY,
    "Content-Type": "application/json",
}


# ── Load test signals once at module level ────────────────────────────────────

def _load_test_signals(n: int = 100) -> list[list[float]]:
    """
    Load N random test windows from test.npz.
    Falls back to synthetic Gaussian noise if file is unavailable.
    """
    test_path = ROOT / "data" / "processed" / "test.npz"
    if test_path.exists():
        data    = np.load(test_path)
        X       = data["X"]
        indices = np.random.choice(len(X), size=min(n, len(X)), replace=False)
        return [X[i].tolist() for i in indices]
    else:
        rng = np.random.default_rng(42)
        return [rng.standard_normal(1024).tolist() for _ in range(n)]


_TEST_SIGNALS: list[list[float]] = _load_test_signals(100)


# ── Locust user ───────────────────────────────────────────────────────────────

class BearingAPIUser(HttpUser):
    """
    Simulated API user hitting the bearing fault inference endpoints.

    wait_time: 0.1 to 0.5 seconds between tasks.
    This gives roughly 2-10 requests/second per user.
    At 50 users → ~100-500 req/s total throughput.
    """

    wait_time = between(0.1, 0.5)

    def on_start(self):
        """Called once per user when they start. Verify server is reachable."""
        r = self.client.get("/health")
        if r.status_code != 200:
            print(f"[WARN] /health returned {r.status_code}")

    @task(7)
    def predict_single(self):
        """
        POST /predict — single signal inference.
        Uses a randomly selected real test window for realistic latency.
        """
        signal = random.choice(_TEST_SIGNALS)
        payload = json.dumps({"signal": signal})
        with self.client.post(
            "/predict",
            data=payload,
            headers=HEADERS,
            catch_response=True,
            name="/predict",
        ) as r:
            if r.status_code == 200:
                body = r.json()
                if "predicted_class" not in body:
                    r.failure(f"Missing predicted_class in response: {body}")
                elif body.get("inference_ms", 0) <= 0:
                    r.failure("inference_ms is zero or negative")
                else:
                    r.success()
            elif r.status_code == 503:
                r.failure("Model not loaded (503)")
            else:
                r.failure(f"Unexpected status {r.status_code}: {r.text[:200]}")

    @task(2)
    def health_check(self):
        """GET /health — public, used by load balancer probes."""
        with self.client.get("/health", catch_response=True, name="/health") as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"Health check failed: {r.status_code}")

    @task(1)
    def drift_check(self):
        """GET /drift — monitoring check (authenticated)."""
        with self.client.get(
            "/drift",
            headers=HEADERS,
            catch_response=True,
            name="/drift",
        ) as r:
            if r.status_code == 200:
                r.success()
            elif r.status_code == 401:
                r.failure("Auth failed on /drift")
            else:
                r.failure(f"Drift check failed: {r.status_code}")


# ── Summary event hook ────────────────────────────────────────────────────────

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """
    Print a pass/fail summary when the test finishes.
    Checks p95 latency against the 200ms target.
    """
    stats = environment.stats.total

    p95_ms        = stats.get_response_time_percentile(0.95)
    failure_rate  = stats.fail_ratio
    rps           = stats.current_rps

    print("\n" + "=" * 60)
    print("  LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"  Total requests  : {stats.num_requests}")
    print(f"  Failure rate    : {failure_rate:.1%}")
    print(f"  Req/sec (peak)  : {rps:.1f}")
    print(f"  p50 latency     : {stats.get_response_time_percentile(0.50):.0f}ms")
    print(f"  p95 latency     : {p95_ms:.0f}ms")
    print(f"  p99 latency     : {stats.get_response_time_percentile(0.99):.0f}ms")
    print()

    target_p95_ms    = 200.0
    target_fail_rate = 0.01   # < 1%

    p95_pass  = p95_ms < target_p95_ms
    fail_pass = failure_rate < target_fail_rate

    print(f"  [{'PASS' if p95_pass else 'FAIL'}] p95 < {target_p95_ms:.0f}ms  "
          f"(actual: {p95_ms:.0f}ms)")
    print(f"  [{'PASS' if fail_pass else 'FAIL'}] failure rate < {target_fail_rate:.0%}  "
          f"(actual: {failure_rate:.1%})")
    print("=" * 60)

    # Non-zero exit code if targets not met
    if not (p95_pass and fail_pass):
        environment.process_exit_code = 1
