"""
db/session.py — Async SQLAlchemy engine and session factory.

Works with:
  SQLite  (dev)  — DATABASE_URL=sqlite+aiosqlite:///./data/bearing_prod.db
  PostgreSQL (prod) — DATABASE_URL=postgresql+asyncpg://user:pass@host/db

The get_db() async generator is used as a FastAPI dependency:
    async def my_endpoint(db: AsyncSession = Depends(get_db)):
        ...

The init_db() function creates all tables on startup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import settings
from src.core.logger import get_logger
from src.db.models import Base

log = get_logger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────

def _create_engine() -> AsyncEngine:
    """
    Create the async SQLAlchemy engine with appropriate pool settings.

    SQLite: uses NullPool (no connection pooling — SQLite handles concurrency
            poorly with async; one write at a time is safest).
    PostgreSQL: uses default AsyncAdaptedQueuePool with reasonable limits.
    """
    db_url = settings.DATABASE_URL
    is_sqlite = "sqlite" in db_url

    if is_sqlite:
        from sqlalchemy.pool import StaticPool
        engine = create_async_engine(
            db_url,
            echo=settings.is_development,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_async_engine(
            db_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    log.info(
        "Database engine created",
        url=db_url.split("@")[-1] if "@" in db_url else db_url,
        backend="sqlite" if is_sqlite else "postgresql",
    )
    return engine


engine: AsyncEngine = _create_engine()

# ── Session factory ───────────────────────────────────────────────────────────

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async generator that yields a database session and ensures it is
    closed after the request completes — even if an exception is raised.

    Usage in FastAPI:
        from fastapi import Depends
        from src.db.session import get_db

        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Context manager (for non-FastAPI use) ─────────────────────────────────────

@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions outside of FastAPI.

    Usage:
        async with db_session() as db:
            record = await crud.save_prediction(db, data)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Table creation ────────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Create all tables defined in db/models.py if they do not exist.
    Called during application startup (lifespan).

    This is NOT a migration system — use Alembic for schema changes in prod.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables initialised", tables=list(Base.metadata.tables.keys()))


async def close_db() -> None:
    """Dispose the engine connection pool. Called during application shutdown."""
    await engine.dispose()
    log.info("Database engine disposed")
