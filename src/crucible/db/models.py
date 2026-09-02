"""ORM models.

Phase 0 owns two tables: `spend` (the cost meter's ledger) and `vector_smoke`
(proof that pgvector and its HNSW index work before Phase 3 depends on them).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from crucible.db.base import Base

EMBEDDING_DIMENSIONS = 384


class Spend(Base):
    """One metered LLM call.

    `round_id` has no foreign key yet: the `rounds` table arrives in Phase 6.
    TODO(phase-6): add the FK to `rounds.id` in that phase's migration.
    """

    __tablename__ = "spend"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    # NULL means "model not in the price table": recorded, warned about, not billed.
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class VectorSmoke(Base):
    """Smoke table proving the `vector` extension and HNSW indexing work.

    TODO(phase-3): the real archive embeddings land in `attacks`; this table may
    be dropped once that migration exists.
    """

    __tablename__ = "vector_smoke"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)

    __table_args__ = (
        Index(
            "ix_vector_smoke_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
