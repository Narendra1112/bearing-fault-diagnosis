"""
core/exceptions.py — Custom exception hierarchy for the bearing diagnosis system.

Every exception carries:
  error_code   — machine-readable string (e.g. "MODEL_NOT_LOADED")
  message      — human-readable description
  http_status  — HTTP status code for the FastAPI handler
  details      — optional dict with extra context

FastAPI exception handlers are registered in src/api/main.py.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


# ── Base ──────────────────────────────────────────────────────────────────────

class BearingDiagnosisError(Exception):
    """
    Base class for all application exceptions.
    Subclasses only need to set class-level defaults.
    """

    error_code:  str = "INTERNAL_ERROR"
    http_status: int = 500
    default_message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the standard JSON error response body."""
        return {
            "error": {
                "code":    self.error_code,
                "message": self.message,
                "details": self.details,
            }
        }


# ── Model / Inference ─────────────────────────────────────────────────────────

class ModelNotLoadedError(BearingDiagnosisError):
    """Raised when inference is requested but no model is in memory."""
    error_code      = "MODEL_NOT_LOADED"
    http_status     = 503
    default_message = (
        "Model is not loaded. The service may still be starting up "
        "or the checkpoint file is missing."
    )


class InferenceTimeoutError(BearingDiagnosisError):
    """Raised when a single inference exceeds the configured timeout."""
    error_code      = "INFERENCE_TIMEOUT"
    http_status     = 504
    default_message = "Inference did not complete within the allowed time limit."


class BatchSizeLimitError(BearingDiagnosisError):
    """Raised when a batch request exceeds MAX_BATCH_SIZE."""
    error_code      = "BATCH_SIZE_EXCEEDED"
    http_status     = 422
    default_message = "Batch size exceeds the configured maximum."


# ── Signal validation ─────────────────────────────────────────────────────────

class InvalidSignalError(BearingDiagnosisError):
    """Raised when the input signal fails validation."""
    error_code      = "INVALID_SIGNAL"
    http_status     = 422
    default_message = "The provided signal is invalid."


class SignalLengthError(InvalidSignalError):
    """Wrong number of samples."""
    error_code      = "SIGNAL_LENGTH_MISMATCH"
    default_message = "Signal length does not match the expected window size."


class NonFiniteSignalError(InvalidSignalError):
    """Signal contains NaN or Inf."""
    error_code      = "SIGNAL_NON_FINITE"
    default_message = "Signal contains NaN or infinite values."


# ── Drift / Monitoring ────────────────────────────────────────────────────────

class DriftDetectedError(BearingDiagnosisError):
    """
    Raised (optionally) when drift is critical and the caller wants to
    reject the prediction rather than return a potentially unreliable result.
    In most cases drift is returned as a warning, not raised.
    """
    error_code      = "DRIFT_CRITICAL"
    http_status     = 200   # drift is informational; still return 200
    default_message = "Critical data drift detected in the input signal."


# ── Database ──────────────────────────────────────────────────────────────────

class DatabaseError(BearingDiagnosisError):
    """Raised when a database operation fails."""
    error_code      = "DATABASE_ERROR"
    http_status     = 500
    default_message = "A database error occurred."


# ── Authentication ────────────────────────────────────────────────────────────

class AuthenticationError(BearingDiagnosisError):
    """Raised when API key is missing or invalid."""
    error_code      = "AUTHENTICATION_FAILED"
    http_status     = 401
    default_message = "Invalid or missing API key. Provide X-API-Key header."


class RateLimitError(BearingDiagnosisError):
    """Raised when the rate limit is exceeded."""
    error_code      = "RATE_LIMIT_EXCEEDED"
    http_status     = 429
    default_message = "Rate limit exceeded. Retry after 60 seconds."


# ── FastAPI exception handlers ────────────────────────────────────────────────

async def bearing_exception_handler(
    request: Request,
    exc: BearingDiagnosisError,
) -> JSONResponse:
    """
    Global handler — converts any BearingDiagnosisError subclass into a
    consistent JSON error response.

    Register in main.py:
        app.add_exception_handler(BearingDiagnosisError, bearing_exception_handler)
    """
    from src.core.logger import get_logger, get_correlation_id
    log = get_logger(__name__)
    log.warning(
        "Application exception",
        error_code=exc.error_code,
        http_status=exc.http_status,
        exc_message=exc.message,
        details=exc.details,
        path=str(request.url),
        cid=get_correlation_id(),
    )
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict(),
        headers={"X-Correlation-ID": get_correlation_id()},
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch-all handler for any unexpected exception.
    Logs the full traceback but returns a sanitised response (no stack traces to clients).
    """
    from src.core.logger import get_logger, get_correlation_id
    log = get_logger(__name__)
    log.exception(
        "Unhandled exception",
        path=str(request.url),
        exc_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code":    "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Check server logs.",
                "details": {"correlation_id": get_correlation_id()},
            }
        },
        headers={"X-Correlation-ID": get_correlation_id()},
    )
