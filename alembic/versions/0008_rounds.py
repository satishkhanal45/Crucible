"""Runs and rounds: what a co-evolutionary run records

Revision ID: 0008_rounds
Revises: 0007_lineage
Create Date: Phase 6

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_rounds"
down_revision: str | None = "0007_lineage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Rates are stored with their Wilson bounds and their counts: a rate without
    # an interval is not a reportable number (docs/spec.md section 15).
    op.create_table(
        "runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("attacker_mode", sa.String(length=16), nullable=False),
        sa.Column("rounds_planned", sa.Integer(), nullable=False),
        sa.Column("rounds_completed", sa.Integer(), nullable=False),
        sa.Column("starting_config_id", sa.String(length=64), nullable=False),
        sa.Column("current_config_id", sa.String(length=64), nullable=False),
        sa.Column("budget_usd", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("halt_reason", sa.String(length=32), nullable=True),
        sa.Column("settings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runs")),
    )
    op.create_table(
        "rounds",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("round_number", sa.Integer(), nullable=False),
        sa.Column("attacker_mode", sa.String(length=16), nullable=False),
        sa.Column("defense_before", sa.String(length=64), nullable=False),
        sa.Column("defense_after", sa.String(length=64), nullable=False),
        sa.Column("attacks_generated", sa.Integer(), nullable=False),
        sa.Column("attacks_rejected_novelty", sa.Integer(), nullable=False),
        sa.Column("breaches_found", sa.Integer(), nullable=False),
        sa.Column("archive_successes", sa.Integer(), nullable=False),
        sa.Column("archive_trials", sa.Integer(), nullable=False),
        sa.Column("archive_block_rate", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.Column("archive_block_rate_ci_low", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.Column("archive_block_rate_ci_high", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.Column("holdout_successes", sa.Integer(), nullable=False),
        sa.Column("holdout_trials", sa.Integer(), nullable=False),
        sa.Column("holdout_block_rate", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.Column("holdout_ci_low", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.Column("holdout_ci_high", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.Column("overfit_gap", sa.Numeric(precision=7, scale=5), nullable=False),
        sa.Column("utility_successes", sa.Integer(), nullable=False),
        sa.Column("utility_trials", sa.Integer(), nullable=False),
        sa.Column("utility_pass_rate", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.Column("mean_novelty", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.Column("cells_occupied", sa.Integer(), nullable=False),
        sa.Column("new_cells", sa.Integer(), nullable=False),
        sa.Column("regressions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("config_promoted", sa.Boolean(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("halt_reason", sa.String(length=32), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_rounds_run_id_runs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rounds")),
        sa.UniqueConstraint("run_id", "round_number", name="uq_rounds_run_round"),
    )
    op.create_index(op.f("ix_rounds_run_id"), "rounds", ["run_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_rounds_run_id"), table_name="rounds")
    op.drop_table("rounds")
    op.drop_table("runs")
