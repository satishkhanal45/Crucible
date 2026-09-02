"""Async engine, session factory, and their lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class DatabaseUnavailable(RuntimeError):
    """Raised when a session is requested after the pool has been closed."""


class Database:
    """Owns one async engine and its session factory."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self._engine: AsyncEngine = create_async_engine(url, echo=echo, pool_pre_ping=True)
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine, expire_on_commit=False, autoflush=False
        )
        self._closed = False

    @property
    def engine(self) -> AsyncEngine:
        if self._closed:
            raise DatabaseUnavailable("the database pool is closed")
        return self._engine

    @property
    def closed(self) -> bool:
        return self._closed

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A session that commits on success and rolls back on failure."""
        if self._closed:
            raise DatabaseUnavailable("the database pool is closed")
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._engine.dispose()
