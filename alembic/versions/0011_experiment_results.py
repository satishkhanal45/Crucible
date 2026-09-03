"""What each Phase 8 experiment produced

Revision ID: 0011_experiment_results
Revises: 0010_classifier_agreement
Create Date: Phase 8

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_experiment_results"
down_revision: str | None = "0010_classifier_agreement"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Every number in docs/findings.md is regenerated from stored data.
    op.create_table(
        "experiment_results",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("experiment", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name=op.f("fk_experiment_results_run_id_runs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_experiment_results")),
    )
    op.create_index(op.f("ix_experiment_results_experiment"), "experiment_results", ["experiment"])


def downgrade() -> None:
    op.drop_index(op.f("ix_experiment_results_experiment"), table_name="experiment_results")
    op.drop_table("experiment_results")
