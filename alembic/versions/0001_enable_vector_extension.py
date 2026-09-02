"""Enable the pgvector extension.

Revision ID: 0001_vector_extension
Revises:
Create Date: Phase 0

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001_vector_extension"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
