"""Every DefenseConfig, addressable by its content fingerprint

Revision ID: 0009_defense_configs
Revises: 0008_rounds
Create Date: Phase 7

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009_defense_configs"
down_revision: str | None = "0008_rounds"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Phase 8's layer ablation replays the archive against configs by id.
    op.create_table(
        "defense_configs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("round_number", sa.Integer(), nullable=True),
        sa.Column("run_id", sa.UUID(), nullable=True),
        sa.Column("parent_config_id", sa.String(length=64), nullable=True),
        sa.Column("label", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["parent_config_id"],
            ["defense_configs.id"],
            name=op.f("fk_defense_configs_parent_config_id_defense_configs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name=op.f("fk_defense_configs_run_id_runs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_defense_configs")),
    )


def downgrade() -> None:
    # Phase 8's layer ablation replays the archive against configs by id.
    op.drop_table("defense_configs")
