"""Integration fixtures: a throwaway pgvector database.

Uses `CRUCIBLE_TEST_DATABASE_URL` when it is set (CI provides a service
container) and otherwise starts one with testcontainers.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from crucible.config import Settings, load_settings

POSTGRES_IMAGE = "pgvector/pgvector:pg16"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.integration


async def _wait_until_accepting_connections(url: str, attempts: int = 60) -> None:
    last: Exception | None = None
    for _ in range(attempts):
        engine = create_async_engine(url)
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return
        except Exception as error:  # the container may still be starting
            last = error
            await asyncio.sleep(1)
        finally:
            await engine.dispose()
    raise RuntimeError(f"database at {url} never became available: {last}")


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    external = os.environ.get("CRUCIBLE_TEST_DATABASE_URL")
    if external:
        asyncio.run(_wait_until_accepting_connections(external))
        yield external
        return

    from testcontainers.core.container import DockerContainer
    from testcontainers.core.wait_strategies import LogMessageWaitStrategy

    container = (
        DockerContainer(POSTGRES_IMAGE)
        .with_env("POSTGRES_USER", "crucible")
        .with_env("POSTGRES_PASSWORD", "crucible")
        .with_env("POSTGRES_DB", "crucible")
        .with_exposed_ports(5432)
        .waiting_for(
            LogMessageWaitStrategy(
                "database system is ready to accept connections"
            ).with_startup_timeout(120)
        )
    )
    container.start()
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        url = f"postgresql+asyncpg://crucible:crucible@{host}:{port}/crucible"
        asyncio.run(_wait_until_accepting_connections(url))
        yield url
    finally:
        container.stop()


@pytest.fixture(scope="session")
def alembic_config(database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@pytest.fixture(scope="session")
def migrated(alembic_config: Config) -> Iterator[Config]:
    """The database at `head`. Every schema-dependent test depends on this."""
    command.upgrade(alembic_config, "head")
    yield alembic_config


@pytest.fixture
def integration_settings(env: dict[str, str], database_url: str) -> Settings:
    del env
    return load_settings(_env_file=None, DATABASE_URL=database_url)
