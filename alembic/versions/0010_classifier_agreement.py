"""Measured agreement between the real classifier and the hand-declared seeds

Revision ID: 0010_classifier_agreement
Revises: 0009_defense_configs
Create Date: Phase 8 prerequisite

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_classifier_agreement"
down_revision: str | None = "0009_defense_configs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Coverage is only as trustworthy as the seed labels; reports quote this.
    op.create_table(
        "classifier_agreement",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("objective_agreed", sa.Integer(), nullable=False),
        sa.Column("technique_agreed", sa.Integer(), nullable=False),
        sa.Column("combined_agreed", sa.Integer(), nullable=False),
        sa.Column("unclassified", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_classifier_agreement")),
    )


def downgrade() -> None:
    op.drop_table("classifier_agreement")
