"""Integration fixtures: a throwaway pgvector database.

Uses `CRUCIBLE_TEST_DATABASE_URL` when it is set (CI provides a service
container) and otherwise starts one with testcontainers.
"""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import AsyncIterator, Callable, Iterator
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from crucible.archive.seeds import load_seed_attacks
from crucible.archive.service import ArchiveService
from crucible.attacker.state import AttackerMode
from crucible.config import Settings, load_settings
from crucible.db.session import Database
from crucible.defender.state import DefenderState
from crucible.defenses.config import DefenseConfig
from crucible.evaluation.benign import load_benign_tasks
from crucible.loop.runner import LoopFactories, LoopRunner, LoopSettings, build_components
from crucible.schemas.attack import Attack
from crucible.services.embeddings import HashingEmbedder
from crucible.target.reference.llm import ScriptedTargetLLM
from tests.fixtures.loop_harness import (
    HARDENED,
    CyclingAttackerLLM,
    Harness,
    RecordingDefenderState,
    ScriptedProposals,
    classifier_client,
    corpus_subset,
)

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


@pytest.fixture
async def archive(database_url: str, migrated: Config) -> AsyncIterator[ArchiveService]:
    """An `ArchiveService` over an empty archive.

    The archive tables are cleared first: novelty is measured against whatever
    is already stored, so a leftover row from another test would change a score.
    """
    del migrated
    database = Database(database_url)
    try:
        async with database.session() as session:
            for table in ("novelty_rejections", "cells", "attempts", "attacks"):
                await session.execute(text(f"DELETE FROM {table}"))
        yield ArchiveService(database, HashingEmbedder(), rng=random.Random(20260903))
    finally:
        await database.close()


@pytest.fixture
async def build_loop(
    database_url: str, migrated: Config, archive: ArchiveService
) -> AsyncIterator[Callable[..., object]]:
    """Builds a loop over a freshly seeded archive."""
    del migrated
    databases: list[Database] = []

    async def _build(
        *,
        rounds: int = 1,
        configs: list[DefenseConfig] | None = None,
        payloads: list[str] | None = None,
        budget: Decimal = Decimal("50.00"),
        seed_attacks: list[Attack] | None = None,
        force_holdout: bool | None = None,
        cells_per_round: int = 2,
        candidates: int = 3,
        utility_tasks: int = 4,
    ) -> Harness:
        database = Database(database_url)
        databases.append(database)

        attacker_llm = CyclingAttackerLLM(payloads)
        defender_llm = ScriptedProposals(configs or [HARDENED])
        settings = LoopSettings(
            rounds=rounds,
            mode=AttackerMode.BLACK_BOX,
            budget_usd=budget,
            concurrency=1,
            candidates_per_round=candidates,
            cells_per_round=cells_per_round,
        )
        components = await build_components(
            database,
            settings=settings,
            factories=LoopFactories(
                target_llm=ScriptedTargetLLM,
                attacker_llm=lambda: attacker_llm,
                defender_llm=lambda: defender_llm,
                classifier_client=classifier_client,
            ),
            embedder=HashingEmbedder(),
            corpus=corpus_subset(),
        )
        components.evaluation._tasks = load_benign_tasks()[:utility_tasks]

        recorded = RecordingDefenderState()
        original_run = components.defender.run

        async def recording_run(state: DefenderState) -> DefenderState:
            recorded.states.append(dict(state))  # type: ignore[arg-type]
            return await original_run(state)

        components.defender.run = recording_run  # type: ignore[method-assign]

        # Seeded through the run's own archive service, so the holdout split is
        # reproducible from the run seed.
        await components.archive.admit_many(
            seed_attacks or load_seed_attacks()[:6],
            round_number=0,
            holdout=force_holdout,
        )
        runner = LoopRunner(database, components, settings=settings, checkpointer=InMemorySaver())
        return Harness(
            runner=runner,
            database=database,
            archive=archive,
            attacker_llm=attacker_llm,
            defender_llm=defender_llm,
            defender_states=recorded,
        )

    yield _build

    for database in databases:
        await database.close()
