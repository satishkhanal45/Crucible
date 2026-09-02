"""Verification tests 7 and 8: migrations round-trip, and models match head.

Test 8 is permanent: from Phase 0 onward, `alembic autogenerate` against head
must produce an empty diff, or a schema change has been made without a revision.
"""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory


def test_upgrade_downgrade_upgrade_runs_clean(migrated: Config) -> None:
    command.upgrade(migrated, "head")
    command.downgrade(migrated, "base")
    command.upgrade(migrated, "head")


def test_head_is_single_and_migrations_are_linear(migrated: Config) -> None:
    script = ScriptDirectory.from_config(migrated)
    assert len(script.get_heads()) == 1, "multiple heads: the migration graph has branched"


def test_autogenerate_produces_an_empty_diff(migrated: Config) -> None:
    """If this fails, add the missing Alembic revision. Never edit it away."""
    command.check(migrated)
