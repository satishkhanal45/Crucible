"""Smoke table proving pgvector and HNSW work before Phase 3 depends on them.

Revision ID: 0002_vector_smoke
Revises: 0001_vector_extension
Create Date: Phase 0

"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "0002_vector_smoke"
down_revision: str | None = "0001_vector_extension"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vector_smoke",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("label", sa.String(length=64), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=384), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_vector_smoke")),
    )
    op.create_index(
        "ix_vector_smoke_embedding_hnsw",
        "vector_smoke",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_vector_smoke_embedding_hnsw", table_name="vector_smoke")
    op.drop_table("vector_smoke")
