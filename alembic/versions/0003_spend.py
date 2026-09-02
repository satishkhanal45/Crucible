"""The cost meter's ledger.

Revision ID: 0003_spend
Revises: 0002_vector_smoke
Create Date: Phase 0

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_spend"
down_revision: str | None = "0002_vector_smoke"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "spend",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        # TODO(phase-6): add a foreign key to rounds.id once that table exists.
        sa.Column("round_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("estimated_cost_usd", sa.Numeric(precision=14, scale=8), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_spend")),
    )
    op.create_index(op.f("ix_spend_round_id"), "spend", ["round_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_spend_round_id"), table_name="spend")
    op.drop_table("spend")
