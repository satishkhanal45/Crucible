"""Attempts, the outcome cache, and private corpus namespaces.

Revision ID: 0005_attempts
Revises: 0004_target_documents
Create Date: Phase 2

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_attempts"
down_revision: str | None = "0004_target_documents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Retrieval over the target corpus must be exact. HNSW is approximate, and a
    # document injected by an indirect attack is an outlier the graph search can
    # fail to reach — which would score an undelivered attack as blocked. The
    # corpus is small enough to scan exactly; the Phase 3 archive keeps HNSW.
    op.drop_index("ix_target_documents_embedding_hnsw", table_name="target_documents")

    # Each concurrent executor worker owns a private copy of the corpus, so that
    # one attempt's injected or deleted documents cannot reach another's.
    op.add_column(
        "target_documents",
        sa.Column("namespace", sa.String(length=64), nullable=False, server_default=""),
    )
    op.alter_column("target_documents", "namespace", server_default=None)
    op.drop_constraint(op.f("pk_target_documents"), "target_documents", type_="primary")
    op.create_primary_key(op.f("pk_target_documents"), "target_documents", ["namespace", "doc_id"])

    op.create_table(
        "attempts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        # TODO(phase-3): FK to attacks.id once the archive table exists.
        sa.Column("attack_id", postgresql.UUID(as_uuid=True), nullable=False),
        # TODO(phase-5): FK to defense_configs.id once that table exists.
        sa.Column("defense_config_id", sa.String(length=64), nullable=False),
        # TODO(phase-6): FK to rounds.id once that table exists.
        sa.Column("round_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("vector", sa.String(length=32), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("tier", sa.Integer(), nullable=True),
        sa.Column(
            "canaries_hit",
            postgresql.ARRAY(sa.String()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "unauthorized_tools",
            postgresql.ARRAY(sa.String()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("judge_score", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column("judge_rationale", sa.Text(), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("trace", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_attempts")),
        # The outcome cache: one row per (attack, defense config) pair.
        sa.UniqueConstraint("attack_id", "defense_config_id", name="uq_attempts_attack_defense"),
    )
    op.create_index(op.f("ix_attempts_round_id"), "attempts", ["round_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_attempts_round_id"), table_name="attempts")
    op.drop_table("attempts")

    # Worker corpus copies are derived data: seeding recreates them. They have to
    # go before the single-column primary key can come back, because the same
    # doc_id exists once per namespace.
    op.execute("DELETE FROM target_documents WHERE namespace <> ''")
    op.drop_constraint(op.f("pk_target_documents"), "target_documents", type_="primary")
    op.create_primary_key(op.f("pk_target_documents"), "target_documents", ["doc_id"])
    op.drop_column("target_documents", "namespace")

    op.create_index(
        "ix_target_documents_embedding_hnsw",
        "target_documents",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
