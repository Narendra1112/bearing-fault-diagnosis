"""
core/logger.py — Structured JSON logging for production.

Every log line is a JSON object:
  {"ts": "2026-01-01T00:00:00Z", "level": "INFO", "cid": "abc123",
   "svc": "bearing-api", "msg": "Prediction complete", "data": {...}}

Usage:
    from src.core.logger import get_logger
    log = get_logger(__name__)
    log.info("Prediction complete", latency_ms=12.3, predicted_class="normal")

Correlation IDs:
    Set via set_correlation_id(cid) in FastAPI middleware.
    All subsequent log calls in that request context carry the same cid.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from src.core.config import settings

# ── Context variable for correlation ID ───────────────────────────────────────
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current async context."""
    _correlation_id.set(cid)


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get()


# ── JSON formatter ────────────────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """
    Formats every log record as a single-line JSON object.
    Extra keyword arguments passed to log.info(..., key=val) appear in "data".
    """

    SERVICE = "bearing-api"

    def format(self, record: logging.LogRecord) -> str:
        # Base payload
        payload: dict[str, Any] = {
            "ts":    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "cid":   _correlation_id.get(),
            "svc":   self.SERVICE,
            "logger": record.name,
            "msg":   record.getMessage(),
        }

        # Extra fields passed as keyword args to log calls
        data: dict[str, Any] = {}
        skip = {
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "message",
            "module", "msecs", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "taskName",
            "thread", "threadName",
        }
        for k, v in record.__dict__.items():
            if k not in skip:
                data[k] = v

        if data:
            payload["data"] = data

        # Exception info
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exc"] = record.exc_text

        # Source location (only in DEBUG)
        if record.levelno == logging.DEBUG:
            payload["src"] = f"{record.filename}:{record.lineno}"

        try:
            return json.dumps(payload, default=str)
        except Exception:
            # Never let the logger crash the application
            payload["data"] = {"serialisation_error": traceback.format_exc()}
            return json.dumps(payload, default=str)


# ── Root handler setup (called once) ─────────────────────────────────────────

def _configure_root_logger() -> None:
    """
    Configure the root logger with a JSON handler on stdout.
    Called once at import time.
    """
    root = logging.getLogger()

    # Don't reconfigure if already set up (prevents duplicate handlers in tests)
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    root.addHandler(handler)

    level = getattr(logging, settings.effective_log_level, logging.INFO)
    root.setLevel(level)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "mlflow", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_root_logger()


# ── Public factory ────────────────────────────────────────────────────────────

class _BoundLogger:
    """
    Thin wrapper that adds keyword-argument support to stdlib Logger.

    log.info("msg", latency_ms=12.3, class_id=0)
    → {"msg": "msg", "data": {"latency_ms": 12.3, "class_id": 0}}
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._log = logger

    def _emit(self, level: int, msg: str, **kwargs: Any) -> None:
        if self._log.isEnabledFor(level):
            self._log.log(level, msg, extra=kwargs, stacklevel=3)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.CRITICAL, msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        """Log ERROR with current exception traceback."""
        self._log.exception(msg, extra=kwargs, stacklevel=2)


def get_logger(name: str) -> _BoundLogger:
    """
    Get a structured JSON logger for the given module name.

    Usage:
        log = get_logger(__name__)
        log.info("model loaded", params=168490, backend="pytorch")
    """
    return _BoundLogger(logging.getLogger(name))
