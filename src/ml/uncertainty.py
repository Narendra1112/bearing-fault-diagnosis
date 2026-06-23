"""
ml/uncertainty.py — Monte Carlo Dropout uncertainty estimation.

Naive softmax confidence is overconfident — a model can output 99% confidence
on out-of-distribution inputs. MC Dropout replaces this with a calibrated
uncertainty estimate by running N stochastic forward passes with dropout active
and measuring the variance across predictions.

Theory:
    During training, dropout randomly zeroes neurons. At inference time it is
    normally disabled. MC Dropout re-enables it at inference and treats each
    forward pass as a sample from an approximate posterior over model weights.
    The variance across N samples is a proxy for epistemic uncertainty.

Usage:
    from src.ml.uncertainty import MCDropoutEstimator
    estimator = MCDropoutEstimator(model, n_samples=30)
    result = estimator.estimate(signal)
    if result.is_uncertain:
        log.warning("Uncertain prediction", std=result.uncertainty_std)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.core.config import settings
from src.core.logger import get_logger
from src.ml.constants import CLASS_NAMES

log = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class UncertaintyResult:
    """
    Result of an MC Dropout uncertainty estimation.

    Attributes:
        predicted_class:  Most likely class name.
        class_id:         Integer class index.
        confidence:       Mean softmax probability for the predicted class.
        uncertainty_std:  Standard deviation of predicted-class probability
                          across MC samples. Higher = more uncertain.
        ci_lower:         Lower bound of 95% confidence interval.
        ci_upper:         Upper bound of 95% confidence interval.
        is_uncertain:     True if uncertainty_std > threshold.
        n_samples:        Number of MC forward passes used.
        mean_proba:       Mean probability vector across all MC samples (10,).
    """
    predicted_class:  str
    class_id:         int
    confidence:       float
    uncertainty_std:  float
    ci_lower:         float
    ci_upper:         float
    is_uncertain:     bool
    n_samples:        int
    mean_proba:       np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class":  self.predicted_class,
            "class_id":         self.class_id,
            "confidence":       round(self.confidence, 6),
            "uncertainty_std":  round(self.uncertainty_std, 6),
            "ci_lower":         round(self.ci_lower, 6),
            "ci_upper":         round(self.ci_upper, 6),
            "is_uncertain":     self.is_uncertain,
            "n_samples":        self.n_samples,
        }


# ── MC Dropout estimator ──────────────────────────────────────────────────────

class MCDropoutEstimator:
    """
    Monte Carlo Dropout uncertainty estimator for BearingCNN.

    Wraps any nn.Module that contains Dropout layers. During estimation,
    dropout is kept active (model in train mode) to sample stochastic
    predictions.

    Args:
        model:     Loaded BearingCNN (or any model with Dropout layers).
        n_samples: Number of forward passes per estimate (default 30).
        threshold: std threshold above which predictions are flagged uncertain.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 30,
        threshold: float | None = None,
    ) -> None:
        self._model     = model
        self.n_samples  = n_samples
        self.threshold  = threshold or settings.DRIFT_UNCERTAINTY_THRESHOLD
        self._lock      = threading.Lock()

        # Count dropout layers to confirm the model is compatible
        n_dropout = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
        log.info(
            "MCDropoutEstimator initialised",
            n_dropout_layers=n_dropout,
            n_samples=n_samples,
            threshold=self.threshold,
        )
        if n_dropout == 0:
            log.warning(
                "Model has no Dropout layers — MC Dropout will produce "
                "zero variance (all samples identical). "
                "Add Dropout layers to the model for meaningful uncertainty."
            )

    def _enable_dropout(self) -> None:
        """Set all Dropout layers to training mode (activates stochastic dropout)."""
        for m in self._model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def estimate(
        self,
        signal: np.ndarray,
        n_samples: int | None = None,
    ) -> UncertaintyResult:
        """
        Run N stochastic forward passes and compute uncertainty statistics.

        The model is set to eval() mode first (disables BatchNorm statistics
        accumulation), then Dropout layers are re-enabled selectively.

        Args:
            signal:   1-D float array of shape (SIGNAL_LENGTH,).
            n_samples: Override default number of samples.

        Returns:
            UncertaintyResult with mean prediction, std, confidence interval.
        """
        n = n_samples or self.n_samples
        sig = np.asarray(signal, dtype=np.float32)
        x   = torch.from_numpy(sig[np.newaxis, np.newaxis, :])   # (1, 1, 1024)

        samples = np.zeros((n, len(CLASS_NAMES)), dtype=np.float32)

        with self._lock:
            # eval() fixes BatchNorm running stats; dropout re-enabled below
            self._model.eval()
            self._enable_dropout()

            try:
                with torch.no_grad():
                    for i in range(n):
                        logits     = self._model(x)               # (1, 10)
                        proba      = torch.softmax(logits, dim=1)
                        samples[i] = proba.numpy()[0]
            finally:
                # CRITICAL: restore full eval() so BatchNorm does not stay in
                # training mode and corrupt running statistics on the next
                # /predict call (which assumes the model is in eval mode).
                self._model.eval()

        # Statistics across MC samples
        mean_proba = samples.mean(axis=0)                          # (10,)
        std_proba  = samples.std(axis=0)                           # (10,)

        pred_id    = int(np.argmax(mean_proba))
        pred_name  = CLASS_NAMES[pred_id]
        confidence = float(mean_proba[pred_id])
        unc_std    = float(std_proba[pred_id])

        # 95% confidence interval for the predicted class probability
        pred_samples = samples[:, pred_id]
        ci_lower = float(np.percentile(pred_samples, 2.5))
        ci_upper = float(np.percentile(pred_samples, 97.5))

        is_uncertain = unc_std > self.threshold

        log.debug(
            "MC Dropout estimate",
            predicted=pred_name,
            confidence=round(confidence, 4),
            uncertainty_std=round(unc_std, 4),
            is_uncertain=is_uncertain,
            n_samples=n,
        )

        return UncertaintyResult(
            predicted_class=pred_name,
            class_id=pred_id,
            confidence=confidence,
            uncertainty_std=unc_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_uncertain=is_uncertain,
            n_samples=n,
            mean_proba=mean_proba,
        )

    def estimate_batch(
        self,
        signals: np.ndarray,
        n_samples: int | None = None,
    ) -> list[UncertaintyResult]:
        """
        Estimate uncertainty for a batch of signals.

        Args:
            signals: Array of shape (N, SIGNAL_LENGTH).
            n_samples: Number of MC samples per signal.

        Returns:
            List of UncertaintyResult, one per signal.
        """
        return [self.estimate(s, n_samples) for s in signals]
