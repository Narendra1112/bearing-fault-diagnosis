"""
core/config.py — Centralised configuration via environment variables.

All configuration is loaded from environment variables (or a .env file).
Nothing is hardcoded — every path, URI, secret, and tunable is here.

Usage:
    from src.core.config import settings
    print(settings.MODEL_PATH)
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent   # project root


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Pydantic-settings validates types and provides clear error messages
    when required variables are missing.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ────────────────────────────────────────────────────────
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Database ───────────────────────────────────────────────────────────
    DATABASE_URL: str = f"sqlite+aiosqlite:///{ROOT}/data/bearing_prod.db"

    # ── Redis ──────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── MLflow ────────────────────────────────────────────────────────────
    MLFLOW_URI: str = str(ROOT / "mlruns")

    # ── Model ─────────────────────────────────────────────────────────────
    MODEL_PATH: Path = ROOT / "models" / "best_cnn.pth"
    ONNX_MODEL_PATH: Path = ROOT / "models" / "best_cnn.onnx"
    TRAIN_DATA_PATH: Path = ROOT / "data" / "processed" / "train.npz"
    TEST_DATA_PATH: Path = ROOT / "data" / "processed" / "test.npz"

    # ── API Security ───────────────────────────────────────────────────────
    # No insecure default that silently authenticates. In development a random
    # ephemeral key is generated at startup (see _post_init guard below) and
    # logged; production MUST set API_KEY explicitly via environment.
    API_KEY: str = ""

    # CORS — comma-separated list of allowed origins. Empty => no cross-origin
    # access is granted (same-origin only). Never use "*" in production.
    CORS_ALLOWED_ORIGINS: str = ""

    # ── Model versioning ──────────────────────────────────────────────────
    MODEL_VERSION: str = "v1.0"

    # ── Rate limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_RPM: int = Field(default=100, ge=1, le=10_000)

    # ── Inference ─────────────────────────────────────────────────────────
    MAX_BATCH_SIZE: int = Field(default=32, ge=1, le=128)
    SIGNAL_LENGTH: int = Field(default=1024, ge=256)
    INFERENCE_TIMEOUT_S: float = Field(default=5.0, ge=0.1)
    WARMUP_RUNS: int = Field(default=10, ge=1)

    # ── Drift detection ───────────────────────────────────────────────────
    DRIFT_ZSCORE_THRESHOLD: float = Field(default=2.0, ge=0.5)
    DRIFT_CUSUM_H: float = Field(default=5.0, ge=1.0)
    DRIFT_MMD_WINDOW: int = Field(default=50, ge=10)
    DRIFT_UNCERTAINTY_THRESHOLD: float = Field(default=0.15, ge=0.01)

    # ── Alerting ──────────────────────────────────────────────────────────
    WEBHOOK_URL: str = ""
    BEARING_HEALTH_ALERT_THRESHOLD: float = Field(default=0.3, ge=0.0, le=1.0)

    # ── Bearing physics (SKF 6205-2RS JEM, CWRU Drive End) ────────────────
    BEARING_NB: int = 9              # number of balls
    BEARING_BD: float = 0.3126      # ball diameter (inches)
    BEARING_PD: float = 1.537       # pitch diameter (inches)
    BEARING_CONTACT_ANGLE: float = 0.0   # contact angle (degrees)
    BEARING_FS: float = 12_000.0    # sampling frequency (Hz)

    # ── Validators ────────────────────────────────────────────────────────
    @field_validator("MODEL_PATH", "ONNX_MODEL_PATH", "TRAIN_DATA_PATH",
                     "TEST_DATA_PATH", mode="before")
    @classmethod
    def coerce_path(cls, v: object) -> Path:
        return Path(str(v))

    @model_validator(mode="after")
    def _enforce_api_key(self) -> "Settings":
        """
        Never authenticate against a hardcoded default.
          - production/staging: API_KEY MUST be provided, else fail fast.
          - development: generate a random ephemeral key per process.
        """
        if not self.API_KEY:
            if self.ENVIRONMENT in ("production", "staging"):
                raise ValueError(
                    "API_KEY must be set via environment in "
                    f"{self.ENVIRONMENT}. Refusing to start without it."
                )
            # development: ephemeral random key, surfaced in logs at startup
            object.__setattr__(self, "API_KEY", secrets.token_hex(16))
            object.__setattr__(self, "_api_key_generated", True)
        return self

    # ── Convenience properties ────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def effective_log_level(self) -> str:
        """DEBUG in dev, INFO otherwise."""
        if self.is_development:
            return "DEBUG"
        return self.LOG_LEVEL

    @property
    def cors_origins(self) -> list[str]:
        """Parse CORS_ALLOWED_ORIGINS (comma-separated) into a list."""
        if not self.CORS_ALLOWED_ORIGINS:
            return []
        return [o.strip() for o in self.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.
    Cached so environment is read only once per process.
    Call get_settings.cache_clear() in tests to reset.
    """
    return Settings()


# Module-level alias — most code imports this directly
settings: Settings = get_settings()
