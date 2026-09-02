"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from crucible.config import Settings
from crucible.db.session import Database
from crucible.schemas.health import Liveness, Readiness
from crucible.services.health import check_readiness

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Liveness)
async def health(request: Request) -> Liveness:
    """Liveness: the process is up. Deliberately does not touch the database."""
    settings: Settings = request.app.state.settings
    return Liveness(status="ok", env=settings.ENV)


@router.get("/ready", response_model=Readiness)
async def ready(request: Request, response: Response) -> Readiness:
    """Readiness: a database round-trip plus a pgvector expression."""
    database: Database = request.app.state.database
    report = await check_readiness(database)
    if report.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report
