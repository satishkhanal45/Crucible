"""Readiness checks. The API layer calls these; it never touches the ORM."""

from __future__ import annotations

from sqlalchemy import text

from crucible.db.session import Database, DatabaseUnavailable
from crucible.logging import get_logger
from crucible.schemas.health import Readiness

logger = get_logger(__name__)


async def check_readiness(database: Database) -> Readiness:
    """Round-trip the database and confirm pgvector answers.

    Never raises: an unready process reports, it does not crash.
    """
    try:
        async with database.session() as session:
            await session.execute(text("SELECT 1"))
            await session.execute(text("SELECT '[1,2,3]'::vector"))
    except DatabaseUnavailable as error:
        logger.warning("readiness.pool_closed", extra={"error": str(error)})
        return Readiness(status="unavailable", database=False, pgvector=False, detail=str(error))
    except Exception as error:  # readiness classifies failures; it never propagates them
        logger.warning("readiness.failed", extra={"error": str(error)})
        return Readiness(status="unavailable", database=False, pgvector=False, detail=str(error))
    return Readiness(status="ready", database=True, pgvector=True)
