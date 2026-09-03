"""Which model produced a run's numbers, and whether it was a stub

Revision ID: 0012_run_provenance
Revises: 0011_experiment_results
Create Date: Phase 8

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012_run_provenance"
down_revision: str | None = "0011_experiment_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A stubbed run produces results shaped exactly like a real one. Without
    # this flag, a test double is indistinguishable from a measurement.
    op.add_column(
        "runs",
        sa.Column("stubbed", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "runs",
        sa.Column(
            "provenance",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    # Runs recorded before provenance existed cannot prove they were live, and
    # an unprovable run must not be readable as one.
    op.execute("UPDATE runs SET stubbed = true WHERE provenance = '{}'::jsonb")
    # The server defaults existed only to backfill: the ORM owns the default, and
    # leaving them would show up forever as autogenerate drift.
    op.alter_column("runs", "stubbed", server_default=None)
    op.alter_column("runs", "provenance", server_default=None)


def downgrade() -> None:
    op.drop_column("runs", "provenance")
    op.drop_column("runs", "stubbed")
