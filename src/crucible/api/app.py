"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from crucible.api.middleware import RequestIdMiddleware
from crucible.api.routes import health_router
from crucible.config import Settings, get_settings
from crucible.db.session import Database
from crucible.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, *, database: Database | None = None) -> FastAPI:
    """Build the app.

    `database` is injectable so tests can point the app at a throwaway
    container without going through the environment.
    """
    active = settings or get_settings()
    configure_logging(active.log_level_number)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = database is None
        app.state.database = database or Database(active.DATABASE_URL)
        logger.info("app.startup", extra={"env": active.ENV})
        try:
            yield
        finally:
            if owned:
                await app.state.database.close()
            logger.info("app.shutdown", extra={"env": active.ENV})

    app = FastAPI(
        title="Crucible",
        version="0.1.0",
        summary="A co-evolutionary red-team loop for RAG systems.",
        lifespan=lifespan,
    )
    app.state.settings = active
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health_router)
    return app
