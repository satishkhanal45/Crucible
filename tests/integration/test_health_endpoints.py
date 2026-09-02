"""Verification test 5: liveness needs no database; readiness reports the truth."""

from __future__ import annotations

import pytest
from alembic.config import Config

from crucible.api.app import create_app
from crucible.config import Settings, load_settings
from crucible.db.session import Database
from tests.fixtures.asgi import running

UNREACHABLE = "postgresql+asyncpg://crucible:crucible@127.0.0.1:1/crucible"


@pytest.fixture
def settings_without_database(env: dict[str, str]) -> Settings:
    del env
    return load_settings(_env_file=None, DATABASE_URL=UNREACHABLE)


async def test_health_is_200_without_a_database(settings_without_database: Settings) -> None:
    app = create_app(settings_without_database, database=Database(UNREACHABLE))
    async with running(app) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "env": "test"}


async def test_ready_is_503_when_the_database_is_unreachable(
    settings_without_database: Settings,
) -> None:
    app = create_app(settings_without_database, database=Database(UNREACHABLE))
    async with running(app) as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["database"] is False
    assert body["pgvector"] is False


async def test_ready_is_200_against_a_live_pgvector_database(
    integration_settings: Settings, database_url: str, migrated: Config
) -> None:
    del migrated
    app = create_app(integration_settings, database=Database(database_url))
    async with running(app) as client:
        response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "database": True,
        "pgvector": True,
        "detail": None,
    }


async def test_ready_is_503_with_the_pool_closed(
    integration_settings: Settings, database_url: str, migrated: Config
) -> None:
    del migrated
    database = Database(database_url)
    app = create_app(integration_settings, database=database)

    async with running(app) as client:
        assert (await client.get("/ready")).status_code == 200

        await database.close()

        response = await client.get("/ready")
        assert response.status_code == 503
        assert response.json()["detail"] == "the database pool is closed"
        # Liveness is unaffected: the process is up, it just cannot serve.
        assert (await client.get("/health")).status_code == 200
