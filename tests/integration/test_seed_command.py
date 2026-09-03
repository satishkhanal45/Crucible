"""`make seed`: load the corpus, plant the canaries, verify placement.

Seeding that silently failed to plant a canary would make every later
measurement wrong, so the command verifies its own work and this test verifies
the verification.
"""

from __future__ import annotations

import pytest
from alembic.config import Config

from crucible.cli.seed import seed
from crucible.config import Settings, load_settings
from crucible.services.embeddings import HASHING_MODEL_NAME
from crucible.target.canary import active_canaries

EXPECTED_CHECKS = {
    # Phase 1: the corpus and the canaries.
    "corpus loaded",
    "confidential documents present",
    "DOCSECRET_CANARY planted",
    "SYSPROMPT_CANARY planted",
    "SESSION_CANARY planted",
    "TOOL_CANARY returned by privileged tool",
    "retrieval excludes confidential documents",
    "benign query retrieves policy documents",
    "committed corpus contains no canary",
    # Phase 3: the seed attacks and the holdout reservation.
    "seed attacks archived",
    "holdout reserved before execution",
    "seed coverage of the grid",
}


@pytest.fixture
def seed_settings(env: dict[str, str], database_url: str, migrated: Config) -> Settings:
    del env, migrated
    # The hashing embedder keeps the test offline: no model download, no network.
    return load_settings(
        _env_file=None, DATABASE_URL=database_url, EMBEDDING_MODEL=HASHING_MODEL_NAME
    )


async def test_seed_passes_every_placement_check(seed_settings: Settings) -> None:
    checks = await seed(seed_settings)

    failed = [check.name for check in checks if not check.passed]
    assert failed == [], f"seed verification failed: {failed}"
    assert {check.name for check in checks} == EXPECTED_CHECKS


async def test_seed_reserves_a_fifth_of_the_seed_attacks_before_execution(
    seed_settings: Settings,
) -> None:
    """The holdout reservation is part of seeding, not of the first round."""
    checks = {check.name: check for check in await seed(seed_settings)}

    holdout = checks["holdout reserved before execution"]
    coverage = checks["seed coverage of the grid"]

    assert holdout.passed is True
    assert "20%" in holdout.detail
    assert "/96" in coverage.detail, "coverage must always carry its denominator"


async def test_seed_never_puts_a_canary_in_its_own_output(seed_settings: Settings) -> None:
    checks = await seed(seed_settings)

    printed = "\n".join(f"{check.name} {check.detail}" for check in checks)
    assert "CRUCIBLE-" not in printed


async def test_seed_clears_the_active_canary_set_when_it_finishes(
    seed_settings: Settings,
) -> None:
    await seed(seed_settings)

    assert active_canaries() is None
