"""The archive: attacks, cells, novelty rejections, and blocked tool calls.

Revision ID: 0006_archive
Revises: 0005_attempts
Create Date: Phase 3

"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "0006_archive"
down_revision: str | None = "0005_attempts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The archive keeps its HNSW index, unlike target_documents: novelty is a
    # heuristic over a growing table and tolerates approximation, where
    # retrieval did not. See the note on AttackRow.
    op.create_table(
        "attacks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("round_generated", sa.Integer(), nullable=False),
        sa.Column("parent_id", sa.UUID(), nullable=True),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("vector", sa.String(length=32), nullable=False),
        sa.Column("objective", sa.String(length=32), nullable=True),
        sa.Column("technique", sa.String(length=32), nullable=True),
        sa.Column("cell_key", sa.String(length=96), nullable=True),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=384), nullable=False),
        sa.Column("novelty_score", sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column("first_breach_round", sa.Integer(), nullable=True),
        sa.Column("total_attempts", sa.Integer(), nullable=False),
        sa.Column("total_breaches", sa.Integer(), nullable=False),
        sa.Column("is_holdout", sa.Boolean(), nullable=False),
        sa.Column("retired", sa.Boolean(), nullable=False),
        sa.Column("benign_user_input", sa.Text(), nullable=True),
        sa.Column("carrier_title", sa.String(length=512), nullable=True),
        sa.Column("carrier_doc_id", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "vector IN ('direct', 'indirect_document')", name=op.f("ck_attacks_executable_vector")
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["attacks.id"],
            name=op.f("fk_attacks_parent_id_attacks"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_attacks")),
    )
    op.create_index(op.f("ix_attacks_cell_key"), "attacks", ["cell_key"], unique=False)
    op.create_index(
        "ix_attacks_embedding_hnsw",
        "attacks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_attacks_is_holdout", "attacks", ["is_holdout"], unique=False)
    op.create_table(
        "cells",
        sa.Column("cell_key", sa.String(length=96), nullable=False),
        sa.Column("elite_attack_id", sa.UUID(), nullable=True),
        sa.Column("elite_fitness", sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column("occupancy", sa.Integer(), nullable=False),
        sa.Column("last_updated_round", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["elite_attack_id"],
            ["attacks.id"],
            name=op.f("fk_cells_elite_attack_id_attacks"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("cell_key", name=op.f("pk_cells")),
    )
    op.create_table(
        "novelty_rejections",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("round_number", sa.Integer(), nullable=False),
        sa.Column("novelty_score", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.Column("threshold", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.Column("nearest_neighbour_id", sa.UUID(), nullable=True),
        sa.Column("nearest_distance", sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["nearest_neighbour_id"],
            ["attacks.id"],
            name=op.f("fk_novelty_rejections_nearest_neighbour_id_attacks"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_novelty_rejections")),
    )
    # Phase 2 addendum: distinguish "Layer 5 stopped a hijack" from "no hijack
    # was ever attempted". Both stay `blocked` for scoring.
    op.add_column(
        "attempts",
        sa.Column(
            "blocked_tools", sa.ARRAY(sa.String()), server_default=sa.text("'{}'"), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_column("attempts", "blocked_tools")
    op.drop_table("novelty_rejections")
    op.drop_table("cells")
    op.drop_index("ix_attacks_is_holdout", table_name="attacks")
    op.drop_index(
        "ix_attacks_embedding_hnsw",
        table_name="attacks",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.drop_index(op.f("ix_attacks_cell_key"), table_name="attacks")
    op.drop_table("attacks")
