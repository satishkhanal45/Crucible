"""Verification test 8, plus reports rendered from a real run.

Test 8 is the blocking prerequisite for Phase 8: without configs addressable by
id, the layer-ablation experiment cannot replay the final archive against a
config with one layer disabled, and cross-round comparisons have nothing to
compare.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import pytest
from alembic.config import Config

from crucible.db.session import Database
from crucible.defenses.config import (
    ContextLayer,
    DefenseConfig,
    HeuristicRule,
    InputLayer,
    OutputLayer,
    StructuralLayer,
)
from crucible.reporting.data import gather
from crucible.reporting.markdown import render_round_report, render_run_report
from crucible.repositories.configs import DefenseConfigRepository
from crucible.schemas.taxonomy import GRID_DENOMINATOR

HARDENED = DefenseConfig(
    input=InputLayer(
        heuristic_rules=(
            HeuristicRule(
                name="instructions_in_retrieved",
                pattern_class="instruction_like",
                applies_to=("retrieved_context",),
                action="strip",
            ),
        )
    ),
    context=ContextLayer(strip_instructions_from_retrieved=True, provenance_tags=True),
    output=OutputLayer(canary_scan=True, citation_verification=True),
    structural=StructuralLayer(
        tool_allowlist=("send_email",), require_user_origin_for_privileged=True
    ),
)


@pytest.fixture
async def database(database_url: str, migrated: Config) -> AsyncIterator[Database]:
    del migrated
    handle = Database(database_url)
    try:
        yield handle
    finally:
        await handle.close()


# ------------------------------------------------------------------ test 8


async def test_a_config_round_trips_by_id(database: Database) -> None:
    async with database.session() as session:
        stored = await DefenseConfigRepository(session).save(HARDENED, label="hand-written")

    async with database.session() as session:
        resolved = await DefenseConfigRepository(session).get(stored)
        record = await DefenseConfigRepository(session).record(stored)

    assert stored == HARDENED.fingerprint()
    assert resolved == HARDENED, "a config must come back exactly as it went in"
    assert record is not None and record.label == "hand-written"


async def test_saving_the_same_config_twice_keeps_one_row(database: Database) -> None:
    """The id is the content fingerprint, so storing is idempotent."""
    async with database.session() as session:
        repository = DefenseConfigRepository(session)
        first = await repository.save(HARDENED)
        after_first = await repository.count()
        second = await repository.save(HARDENED, label="ignored on conflict")
        after_second = await repository.count()

    assert first == second
    assert after_second == after_first, "storing the same config twice adds no row"


async def test_the_cli_can_resolve_a_non_empty_config(database: Database) -> None:
    """`crucible eval defense <id>` resolves through exactly this call."""
    async with database.session() as session:
        config_id = await DefenseConfigRepository(session).save(HARDENED)

    async with database.session() as session:
        resolved = await DefenseConfigRepository(session).get(config_id)
        missing = await DefenseConfigRepository(session).get("0" * 32)

    assert resolved is not None
    assert resolved.fingerprint() == config_id
    assert resolved != DefenseConfig.empty(), "the Phase 6 TODO was empty-only"
    assert missing is None, "an unknown id resolves to nothing, not to a default"


async def test_a_loop_run_stores_d_zero_and_every_promoted_config(
    build_loop: Callable[..., object], database: Database
) -> None:
    harness = await build_loop(rounds=1)  # type: ignore[misc]

    report = await harness.runner.start(starting_config=DefenseConfig.empty())

    async with database.session() as session:
        repository = DefenseConfigRepository(session)
        starting = await repository.record(report.starting_config_id)
        final = await repository.record(report.final_config_id)
        for_run = await repository.list_for_run(report.run_id)

    assert starting is not None and starting.label == "D(0)"
    assert starting.config == DefenseConfig.empty()
    assert final is not None
    assert for_run, "the run's configs are addressable by run id"
    if report.final_config_id != report.starting_config_id:
        assert final.parent_config_id == report.starting_config_id
        assert final.round_number == 1


# ------------------------------------------- reports from a real run


async def test_a_report_renders_from_a_real_run(
    build_loop: Callable[..., object], database: Database
) -> None:
    harness = await build_loop(rounds=2)  # type: ignore[misc]
    report = await harness.runner.start(starting_config=DefenseConfig.empty())

    data = await gather(database, report.run_id)
    assert data is not None

    markdown = render_run_report(data)
    round_one = render_round_report(data, 1)

    assert f"# Crucible run `{report.run_id}`" in markdown
    assert "## Methodology" in markdown
    assert "## The three curves" in markdown
    assert "## Coverage evolution" in markdown
    assert "## Most general attacks" in markdown
    assert "## Defense changelog" in markdown
    assert "## Regressions and halts" in markdown
    assert "## Limitations" in markdown
    assert f"/{GRID_DENOMINATOR}" in markdown
    assert "CRUCIBLE-" not in markdown, "no canary may reach a published report"
    assert "# Round 1" in round_one
    assert "Configuration diff" in round_one

    # The same data renders identically, twice.
    assert render_run_report(data) == markdown


async def test_gathering_an_unknown_run_returns_nothing(database: Database) -> None:
    import uuid

    assert await gather(database, uuid.uuid4()) is None
