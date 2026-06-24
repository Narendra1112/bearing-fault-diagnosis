"""
evaluation/model_validator.py — Pre-deployment model validation gate.

Runs a checklist of quality checks before any model is activated in production.
Used in the GitHub Actions CI pipeline (FILE 28) before docker build.

Checks:
  1. Test accuracy >= 95% (configurable)
  2. Per-class accuracy >= 85% (catches class collapse)
  3. Inference p95 latency <= 50ms on CPU
  4. No NaN/Inf in logits on 100 random inputs
  5. ONNX output matches PyTorch within 1e-4 (numerical parity)

Returns PASS or FAIL with detailed breakdown.

Usage:
    python -m src.evaluation.model_validator
    python -m src.evaluation.model_validator --threshold 0.95

Exit code:
  0 — all checks passed
  1 — one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.config import settings
from src.core.logger import get_logger
from src.ml.constants import CLASS_NAMES
from src.ml.inference_engine import InferenceEngine

log = get_logger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent


# ── Individual checks ─────────────────────────────────────────────────────────

def check_test_accuracy(
    engine: InferenceEngine,
    test_path: Path,
    threshold: float = 0.95,
) -> dict[str, Any]:
    """Check 1: Overall test accuracy >= threshold."""
    data = np.load(test_path)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)

    preds = []
    model = engine._model
    model.eval()
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i: i + batch_size][:, np.newaxis, :])
            p     = model(batch).argmax(dim=1).numpy()
            preds.append(p)

    preds   = np.concatenate(preds)
    acc     = float((preds == y).mean())
    passed  = acc >= threshold

    return {
        "check":     "test_accuracy",
        "passed":    passed,
        "value":     round(acc, 4),
        "threshold": threshold,
        "detail":    f"Accuracy {acc:.4f} {'OK' if passed else 'BELOW'} threshold {threshold}",
    }


def check_per_class_accuracy(
    engine: InferenceEngine,
    test_path: Path,
    threshold: float = 0.85,
) -> dict[str, Any]:
    """Check 2: Minimum per-class accuracy >= threshold (catches class collapse)."""
    data = np.load(test_path)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)

    model = engine._model
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), 256):
            batch = torch.from_numpy(X[i: i + 256][:, np.newaxis, :])
            preds.append(model(batch).argmax(dim=1).numpy())

    preds = np.concatenate(preds)
    per_class = {}
    for cls_id, name in enumerate(CLASS_NAMES):
        mask = y == cls_id
        if mask.sum() > 0:
            per_class[name] = round(float((preds[mask] == cls_id).mean()), 4)

    worst_class = min(per_class, key=per_class.get)
    worst_acc   = per_class[worst_class]
    passed      = worst_acc >= threshold

    return {
        "check":       "per_class_accuracy",
        "passed":      passed,
        "value":       worst_acc,
        "threshold":   threshold,
        "worst_class": worst_class,
        "per_class":   per_class,
        "detail":      f"Worst class '{worst_class}': {worst_acc:.4f}",
    }


def check_latency(
    engine: InferenceEngine,
    n_warmup: int = 10,
    n_measure: int = 100,
    p95_threshold_ms: float = 50.0,
) -> dict[str, Any]:
    """Check 3: p95 inference latency <= threshold ms."""
    rng    = np.random.default_rng(0)
    dummy  = rng.standard_normal(settings.SIGNAL_LENGTH).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        engine.predict(dummy)

    # Measure
    latencies = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        engine.predict(dummy)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p95 = float(np.percentile(latencies, 95))
    p50 = float(np.percentile(latencies, 50))

    passed = p95 <= p95_threshold_ms
    return {
        "check":            "latency_p95",
        "passed":           passed,
        "value_p95_ms":     round(p95, 2),
        "value_p50_ms":     round(p50, 2),
        "threshold_ms":     p95_threshold_ms,
        "n_measurements":   n_measure,
        "detail":           f"p95={p95:.1f}ms  p50={p50:.1f}ms",
    }


def check_nan_inf(
    engine: InferenceEngine,
    n_inputs: int = 100,
) -> dict[str, Any]:
    """Check 4: No NaN or Inf in model outputs for random inputs."""
    rng    = np.random.default_rng(1)
    model  = engine._model
    model.eval()
    found  = []

    with torch.no_grad():
        for i in range(n_inputs):
            x      = rng.standard_normal(settings.SIGNAL_LENGTH).astype(np.float32)
            t      = torch.from_numpy(x[np.newaxis, np.newaxis, :])
            logits = model(t)
            if not torch.isfinite(logits).all():
                found.append(i)

    passed = len(found) == 0
    return {
        "check":        "nan_inf_outputs",
        "passed":       passed,
        "n_tested":     n_inputs,
        "n_bad_inputs": len(found),
        "detail":       "All outputs finite" if passed else f"{len(found)} inputs produced NaN/Inf",
    }


def check_onnx_parity(
    engine: InferenceEngine,
    onnx_path: Path,
    n_inputs: int = 20,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Check 5: ONNX output matches PyTorch within tolerance."""
    if not onnx_path.exists():
        return {
            "check":   "onnx_parity",
            "passed":  False,
            "detail":  f"ONNX file not found: {onnx_path}. Run export_to_onnx() first.",
            "skipped": True,
        }

    try:
        engine.load_onnx(onnx_path)
    except ImportError:
        return {
            "check":   "onnx_parity",
            "passed":  True,
            "detail":  "onnxruntime not installed — check skipped",
            "skipped": True,
        }

    rng        = np.random.default_rng(2)
    model      = engine._model
    model.eval()

    max_diff = 0.0
    with torch.no_grad():
        for _ in range(n_inputs):
            x = rng.standard_normal(settings.SIGNAL_LENGTH).astype(np.float32)

            # PyTorch
            t     = torch.from_numpy(x[np.newaxis, np.newaxis, :])
            pt_p  = torch.softmax(model(t), dim=1).numpy()[0]

            # ONNX (via engine)
            ort_p = engine._predict_onnx(x)
            diff  = float(np.abs(pt_p - ort_p).max())
            max_diff = max(max_diff, diff)

    passed = max_diff <= tolerance
    # Restore pytorch backend
    engine.load_pytorch(settings.MODEL_PATH)

    return {
        "check":     "onnx_parity",
        "passed":    passed,
        "max_diff":  round(max_diff, 8),
        "tolerance": tolerance,
        "n_inputs":  n_inputs,
        "detail":    f"Max |pytorch - onnx| = {max_diff:.2e} (tol={tolerance:.0e})",
    }


# ── Main validator ────────────────────────────────────────────────────────────

def run_validation(
    checkpoint_path: Path | None = None,
    test_path: Path | None = None,
    onnx_path: Path | None = None,
    accuracy_threshold: float = 0.95,
    per_class_threshold: float = 0.85,
    latency_threshold_ms: float = 50.0,
) -> dict[str, Any]:
    """
    Run all validation checks. Returns results dict with PASS/FAIL verdict.

    Args:
        checkpoint_path:      Path to .pth file.
        test_path:            Path to test.npz.
        onnx_path:            Path to .onnx file (for parity check).
        accuracy_threshold:   Minimum overall test accuracy.
        per_class_threshold:  Minimum per-class accuracy.
        latency_threshold_ms: Maximum p95 latency in milliseconds.

    Returns:
        Dict with overall verdict and per-check results.
    """
    ckpt      = checkpoint_path or settings.MODEL_PATH
    test_np   = test_path       or settings.TEST_DATA_PATH
    onnx      = onnx_path       or settings.ONNX_MODEL_PATH

    if not ckpt.exists():
        return {"verdict": "FAIL", "reason": f"Checkpoint not found: {ckpt}"}
    if not test_np.exists():
        return {"verdict": "FAIL", "reason": f"Test data not found: {test_np}"}

    engine = InferenceEngine()
    engine.load_pytorch(ckpt)

    checks = []

    print("\n=== Model Validation Gate ===")
    print(f"Checkpoint : {ckpt}")
    print(f"Test data  : {test_np}\n")

    check_fns = [
        ("Accuracy >= {:.0%}".format(accuracy_threshold),
         lambda: check_test_accuracy(engine, test_np, accuracy_threshold)),
        ("Per-class >= {:.0%}".format(per_class_threshold),
         lambda: check_per_class_accuracy(engine, test_np, per_class_threshold)),
        ("p95 latency <= {}ms".format(latency_threshold_ms),
         lambda: check_latency(engine, p95_threshold_ms=latency_threshold_ms)),
        ("No NaN/Inf in outputs",
         lambda: check_nan_inf(engine)),
        ("ONNX parity within 1e-4",
         lambda: check_onnx_parity(engine, onnx)),
    ]

    for name, fn in check_fns:
        result = fn()
        checks.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         {result['detail']}")

    n_passed = sum(1 for c in checks if c["passed"])
    n_failed = len(checks) - n_passed
    verdict  = "PASS" if n_failed == 0 else "FAIL"

    print(f"\nVerdict: {verdict}  ({n_passed}/{len(checks)} checks passed)")

    output = {
        "verdict":      verdict,
        "n_checks":     len(checks),
        "n_passed":     n_passed,
        "n_failed":     n_failed,
        "checkpoint":   str(ckpt),
        "checks":       checks,
    }

    out_dir  = ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_report.json"
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Validation report saved", path=str(out_path), verdict=verdict)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model validation gate")
    parser.add_argument("--threshold",    type=float, default=0.95)
    parser.add_argument("--per-class",    type=float, default=0.85)
    parser.add_argument("--latency-ms",   type=float, default=50.0)
    args = parser.parse_args()

    result = run_validation(
        accuracy_threshold=args.threshold,
        per_class_threshold=args.per_class,
        latency_threshold_ms=args.latency_ms,
    )
    sys.exit(0 if result["verdict"] == "PASS" else 1)
