"""
evaluation/noise_robustness.py — SNR robustness evaluation.

Adds AWGN (Additive White Gaussian Noise) at calibrated SNR levels to
clean test windows and measures CNN accuracy degradation.

SNR levels tested: 30dB, 20dB, 15dB, 10dB, 5dB, 0dB
  - 30dB: barely perceptible noise
  - 20dB: light noise (typical sensor noise floor)
  - 10dB: significant noise (realistic industrial environment)
   - 5dB:  heavy noise
   - 0dB:  noise power = signal power

SNR formula (signal power already normalized to 1 after z-score):
  SNR_dB = 10 * log10(P_signal / P_noise)
  P_noise = P_signal / 10^(SNR_dB / 10)
  sigma_noise = sqrt(P_noise)

For z-score normalized signals: P_signal = E[x^2] = 1 (unit variance)
  sigma_noise = 1 / sqrt(10^(SNR_dB/10)) = 10^(-SNR_dB/20)

Results saved to: outputs/reports/noise_robustness.json

Usage:
    python -m src.evaluation.noise_robustness
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.core.config import settings
from src.core.logger import get_logger
from src.ml.constants import CLASS_NAMES
from src.ml.inference_engine import BearingCNN

log = get_logger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent

SNR_LEVELS_DB = [30, 20, 15, 10, 5, 0]
ACCURACY_DROP_THRESHOLD = 0.90   # flag SNR where accuracy drops below 90%


def _add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add Additive White Gaussian Noise at a specified SNR.

    For unit-variance z-score normalised signals:
      sigma_noise = 10^(-SNR_dB / 20)

    Args:
        signal: 1-D float array (z-score normalised, unit variance).
        snr_db: Target SNR in decibels.
        rng:    NumPy random generator for reproducibility.

    Returns:
        Noisy signal with same shape as input.
    """
    signal_power = float(np.mean(signal ** 2))
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    sigma_noise  = float(np.sqrt(noise_power))
    noise        = rng.normal(0.0, sigma_noise, size=signal.shape).astype(signal.dtype)
    return signal + noise


def _evaluate_at_snr(
    model: BearingCNN,
    X: np.ndarray,
    y: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    batch_size: int = 256,
) -> dict:
    """
    Evaluate model accuracy on noise-augmented test set at a given SNR.

    Returns per-class accuracy and overall accuracy.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_clean = X[i: i + batch_size]
            # Add noise to each window in the batch
            batch_noisy = np.stack([
                _add_awgn(sig, snr_db, rng) for sig in batch_clean
            ])
            t = torch.from_numpy(batch_noisy[:, np.newaxis, :])
            preds = model(t).argmax(dim=1).numpy()
            all_preds.append(preds)

    preds       = np.concatenate(all_preds)
    overall_acc = float((preds == y).mean())

    per_class = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = y == cls_id
        if mask.sum() == 0:
            continue
        per_class[cls_name] = round(float((preds[mask] == cls_id).mean()), 4)

    return {
        "snr_db":           snr_db,
        "overall_accuracy": round(overall_acc, 4),
        "per_class":        per_class,
        "per_class_min":    round(min(per_class.values()), 4) if per_class else 0.0,
        "n_windows":        len(X),
    }


def run_noise_robustness(
    checkpoint_path: Path | None = None,
    test_npz: Path | None = None,
    snr_levels: list[float] | None = None,
    seed: int = 42,
) -> dict:
    """
    Run full noise robustness evaluation.

    Args:
        checkpoint_path: Path to model checkpoint (.pth).
        test_npz:        Path to test data (.npz with X and y arrays).
        snr_levels:      SNR levels in dB to test (default: 30,20,15,10,5,0).
        seed:            RNG seed for reproducibility.

    Returns:
        Results dict with per-SNR accuracy and SNR threshold.
    """
    ckpt     = checkpoint_path or settings.MODEL_PATH
    test_path = test_npz or settings.TEST_DATA_PATH
    levels   = snr_levels or SNR_LEVELS_DB

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    # Load model
    model = BearingCNN()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Load test data
    data = np.load(test_path)
    X    = data["X"].astype(np.float32)
    y    = data["y"].astype(np.int64)

    rng = np.random.default_rng(seed)
    log.info(
        "Starting noise robustness evaluation",
        n_windows=len(X),
        snr_levels=levels,
        checkpoint=str(ckpt),
    )

    results_by_snr = {}
    snr_threshold  = None   # first SNR where accuracy drops below 90%

    print(f"\n{'SNR (dB)':>10}  {'Accuracy':>10}  {'Worst Class':>12}  {'Status':>10}")
    print("-" * 50)

    for snr_db in levels:
        r   = _evaluate_at_snr(model, X, y, snr_db, rng)
        key = f"{snr_db}dB"
        results_by_snr[key] = r

        status = "OK" if r["overall_accuracy"] >= ACCURACY_DROP_THRESHOLD else "< 90%"
        if r["overall_accuracy"] < ACCURACY_DROP_THRESHOLD and snr_threshold is None:
            snr_threshold = snr_db

        print(f"{snr_db:>10}  {r['overall_accuracy']:>10.4f}  "
              f"{r['per_class_min']:>12.4f}  {status:>10}")

    # Baseline (no noise, infinite SNR)
    r_clean = _evaluate_at_snr(model, X, y, snr_db=999, rng=rng)
    print(f"{'clean':>10}  {r_clean['overall_accuracy']:>10.4f}  "
          f"{r_clean['per_class_min']:>12.4f}  {'baseline':>10}")

    output = {
        "evaluation_type":         "noise_robustness",
        "checkpoint":               str(ckpt),
        "n_test_windows":           len(X),
        "snr_levels_tested":        levels,
        "accuracy_threshold":       ACCURACY_DROP_THRESHOLD,
        "snr_threshold_below_90pct": snr_threshold,
        "baseline_accuracy":        r_clean["overall_accuracy"],
        "results_by_snr":           results_by_snr,
        "note": (
            "CWRU is a clean lab dataset. Real industrial deployment would "
            "typically see 10-20dB SNR. The SNR threshold indicates the "
            "noise floor where this model requires signal filtering."
        ),
    }

    # Save
    out_dir  = ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "noise_robustness.json"
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Results saved", path=str(out_path))

    return output


if __name__ == "__main__":
    print("\n=== Noise Robustness Evaluation ===")
    print(f"SNR levels: {SNR_LEVELS_DB} dB")
    print(f"Threshold : accuracy < {ACCURACY_DROP_THRESHOLD:.0%} flagged\n")

    results = run_noise_robustness()

    thresh = results["snr_threshold_below_90pct"]
    print(f"\nSNR threshold (accuracy < 90%): {thresh} dB"
          if thresh else "\nAccuracy stays >= 90% at all tested SNR levels")
    print(f"Baseline (clean) accuracy: {results['baseline_accuracy']:.4f}")
    print("Results saved to outputs/reports/noise_robustness.json")
