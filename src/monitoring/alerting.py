"""
monitoring/alerting.py — Webhook-based alerting for drift and health events.

Alerts are fired asynchronously in a background thread so they never block
the inference path. Failed deliveries are retried with exponential backoff
(3 attempts: 1s, 2s, 4s).

Alert types:
  drift_warning   — triggered when drift severity is WARNING or CRITICAL
  health_critical — triggered when bearing_health_index < threshold

Alert payload:
  {
    "alert_type":  "drift_warning",
    "severity":    "CRITICAL",
    "timestamp":   "2026-01-01T00:00:00Z",
    "service":     "bearing-api",
    "bearing_id":  "cwru-drive-end",
    "details":     {...}
  }

Configuration:
  WEBHOOK_URL                    — POST destination (empty = alerts disabled)
  BEARING_HEALTH_ALERT_THRESHOLD — health index below which alert fires

Usage:
    from src.monitoring.alerting import alerter
    alerter.send_drift_alert(drift_event)
    alerter.send_health_alert(health_index=0.21)
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

# Maximum number of undelivered alerts in the queue before dropping
_QUEUE_MAXSIZE = 500


# ── Alert payload builder ─────────────────────────────────────────────────────

def _build_payload(
    alert_type: str,
    severity: str,
    details: dict[str, Any],
    bearing_id: str = "cwru-drive-end",
) -> dict[str, Any]:
    return {
        "alert_type": alert_type,
        "severity":   severity,
        "timestamp":  datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "service":    "bearing-api",
        "bearing_id": bearing_id,
        "environment": settings.ENVIRONMENT,
        "details":    details,
    }


# ── HTTP sender with retry ────────────────────────────────────────────────────

def _post_with_retry(
    url: str,
    payload: dict[str, Any],
    max_attempts: int = 3,
) -> bool:
    """
    POST the alert payload to the webhook URL with exponential backoff.

    Args:
        url:          Webhook endpoint URL.
        payload:      JSON-serialisable alert dict.
        max_attempts: Maximum delivery attempts (default 3).

    Returns:
        True if delivered successfully, False if all attempts failed.
    """
    import json
    try:
        import urllib.request
        data  = json.dumps(payload).encode("utf-8")
        delay = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status < 300:
                        log.info(
                            "Alert delivered",
                            alert_type=payload.get("alert_type"),
                            severity=payload.get("severity"),
                            attempt=attempt,
                            status=resp.status,
                        )
                        return True
            except Exception as exc:
                log.warning(
                    "Alert delivery failed",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=str(exc),
                    alert_type=payload.get("alert_type"),
                )
                if attempt < max_attempts:
                    time.sleep(delay)
                    delay *= 2.0   # exponential backoff: 1s → 2s → 4s

        log.error(
            "Alert permanently failed after all retries",
            alert_type=payload.get("alert_type"),
            url=url,
        )
        return False

    except Exception as exc:
        log.error("Unexpected error in alert delivery", error=str(exc))
        return False


# ── Background worker ─────────────────────────────────────────────────────────

class _AlertWorker(threading.Thread):
    """
    Daemon thread that consumes alerts from the queue and delivers them.
    Runs for the lifetime of the process — never blocks the main thread.
    """

    def __init__(self, q: queue.Queue) -> None:
        super().__init__(daemon=True, name="alert-worker")
        self._queue = q
        self._n_sent:    int = 0
        self._n_failed:  int = 0
        self._n_dropped: int = 0   # alerts dropped because the queue was full

    def run(self) -> None:
        log.info("Alert worker started")
        while True:
            try:
                url, payload = self._queue.get(timeout=1.0)
                success = _post_with_retry(url, payload)
                if success:
                    self._n_sent   += 1
                else:
                    self._n_failed += 1
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                log.error("Alert worker error", error=str(exc))

    @property
    def stats(self) -> dict[str, int]:
        return {"sent": self._n_sent, "failed": self._n_failed,
                "dropped": self._n_dropped, "queue_size": self._queue.qsize()}


# ── Public alerter ────────────────────────────────────────────────────────────

class WebhookAlerter:
    """
    Asynchronous webhook alerter for drift and health events.

    All send_* methods enqueue the alert and return immediately.
    Delivery happens in a background daemon thread.

    If WEBHOOK_URL is empty, alerting is disabled and calls are no-ops.
    """

    def __init__(self) -> None:
        self._url   = settings.WEBHOOK_URL
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._worker = _AlertWorker(self._queue)
        self._worker.start()

        if not self._url:
            log.info("Alerting disabled — WEBHOOK_URL not configured")
        else:
            log.info("Alerting enabled", webhook_url=self._url)

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def _enqueue(self, payload: dict[str, Any]) -> None:
        """Enqueue alert for background delivery. Drops silently if queue full."""
        if not self._url:
            return
        try:
            self._queue.put_nowait((self._url, payload))
        except queue.Full:
            self._worker._n_dropped += 1
            log.warning(
                "Alert queue full — dropping alert",
                alert_type=payload.get("alert_type"),
                queue_size=self._queue.qsize(),
                total_dropped=self._worker._n_dropped,
            )

    def send_drift_alert(self, drift_event: dict[str, Any]) -> None:
        """
        Send a drift detection alert.

        Args:
            drift_event: Dict from DriftResult.to_dict() or drift DB record.
        """
        severity = drift_event.get("severity", "WARNING")
        payload  = _build_payload(
            alert_type="drift_warning",
            severity=severity,
            details={
                "detection_method": drift_event.get("detection_method"),
                "n_checked":        drift_event.get("n_checked"),
                "flags":            drift_event.get("flags", {}),
                "mmd_score":        drift_event.get("mmd_score"),
            },
        )
        log.info(
            "Queuing drift alert",
            severity=severity,
            method=drift_event.get("detection_method"),
        )
        self._enqueue(payload)

    def send_health_alert(
        self,
        health_index: float,
        bearing_id: str = "cwru-drive-end",
    ) -> None:
        """
        Send a bearing health critical alert.

        Args:
            health_index: Current health index value [0, 1].
            bearing_id:   Identifier of the bearing being monitored.
        """
        if health_index > settings.BEARING_HEALTH_ALERT_THRESHOLD:
            return   # Not critical — no alert needed

        payload = _build_payload(
            alert_type="health_critical",
            severity="CRITICAL",
            bearing_id=bearing_id,
            details={
                "health_index": round(health_index, 4),
                "threshold":    settings.BEARING_HEALTH_ALERT_THRESHOLD,
                "message": (
                    f"Bearing health index {health_index:.3f} is below "
                    f"critical threshold {settings.BEARING_HEALTH_ALERT_THRESHOLD}. "
                    "Immediate inspection recommended."
                ),
            },
        )
        log.warning(
            "Queuing health critical alert",
            health_index=health_index,
            threshold=settings.BEARING_HEALTH_ALERT_THRESHOLD,
        )
        self._enqueue(payload)

    def get_stats(self) -> dict[str, Any]:
        """Return delivery statistics from the background worker."""
        return {
            "enabled":     self.enabled,
            "webhook_url": self._url or None,
            **self._worker.stats,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

alerter: WebhookAlerter = WebhookAlerter()
