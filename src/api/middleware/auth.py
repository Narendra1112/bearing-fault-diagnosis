"""
api/middleware/auth.py — API key authentication for protected endpoints.

Public endpoints (no key required):
  GET /health
  GET /metrics

Protected endpoints (X-API-Key header required):
  POST /predict
  POST /batch_predict
  GET  /drift
  GET  /drift/history
  GET  /metrics/summary
  GET  /health/bearing
  POST /drift/reset
  GET  /alerts/recent
  GET  /models
  GET  /models/active
  POST /models/{version}/activate

Usage as a FastAPI dependency:
    from src.api.middleware.auth import require_api_key
    @router.post("/predict")
    async def predict(auth: None = Depends(require_api_key)):
        ...
"""

from __future__ import annotations

import hmac

from fastapi import Security
from fastapi.security import APIKeyHeader

from src.core.config import settings
from src.core.exceptions import AuthenticationError
from src.core.logger import get_logger

log = get_logger(__name__)

# ── API key extractor from header ─────────────────────────────────────────────

_api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,   # We handle errors ourselves for consistent JSON format
    description="API key for protected endpoints. Include in X-API-Key header.",
)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.

    Returns the validated key on success.
    Raises AuthenticationError (→ 401) if missing or invalid.

    Usage:
        @router.post("/predict")
        async def predict(
            req: PredictRequest,
            _: str = Depends(require_api_key),
        ):
            ...
    """
    if not api_key:
        log.warning("Missing API key", endpoint="protected")
        raise AuthenticationError(
            "X-API-Key header is required. "
            "Include your API key: X-API-Key: <key>"
        )

    # Constant-time comparison to prevent timing attacks (byte-by-byte key recovery).
    if not hmac.compare_digest(api_key, settings.API_KEY):
        log.warning("Invalid API key", key_prefix=api_key[:4] + "***")
        raise AuthenticationError(
            "Invalid API key. Check your X-API-Key header value."
        )

    return api_key


async def optional_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """
    Soft auth dependency — returns the key if valid, None if absent.
    Used for endpoints that behave differently for authenticated users
    but don't strictly require auth.
    """
    if not api_key:
        return None
    if not hmac.compare_digest(api_key, settings.API_KEY):
        raise AuthenticationError("Invalid API key.")
    return api_key
