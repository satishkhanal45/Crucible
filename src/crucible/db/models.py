"""ORM models.

Phase 0 owns two tables: `spend` (the cost meter's ledger) and `vector_smoke`
(proof that pgvector and its HNSW index work before Phase 3 depends on them).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
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


class TargetDocument(Base):
    """One document in the reference target's retrievable corpus.

    `confidential` is enforced in SQL by the repository: a confidential document
    is never returned by retrieval, at any k, for any query.

    `namespace` gives each concurrent executor worker a private copy of the
    corpus. docs/spec.md section 9 requires bounded concurrency with no shared
    mutable target state, and attacks inject and delete documents, so two
    attempts running at once must not be able to see each other's corpus.

    There is deliberately **no HNSW index on `embedding`**. HNSW is approximate:
    a graph search can fail to reach a poorly connected node, and a document
    injected by an `indirect_document` attack is by construction an outlier in
    the corpus, so it is exactly the node the graph is worst at reaching. An
    attack whose carrier document is silently not retrieved would be scored as
    blocked when it was never delivered. The corpus is small (a few hundred
    rows), so an exact scan is both affordable and reproducible, which is what
    the outcome cache in section 9 requires. The archive in Phase 3 is large and
    measures novelty statistically, so it can and should use HNSW.
    """

    __tablename__ = "target_documents"

    namespace: Mapped[str] = mapped_column(String(64), primary_key=True, default="")
    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidential: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="corpus")
    #: True for documents added at attack time via `inject_document`.
    injected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: sha256 of title+text, so `reset()` can tell modified rows from pristine ones.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Attempt(Base):
    """One execution of one attack against one defense configuration.

    The unique constraint on `(attack_id, defense_config_id)` is the outcome
    cache from docs/spec.md section 9. The target runs at temperature 0 and a
    config's id is a content fingerprint, so the pair determines the outcome:
    full-archive re-evaluation stays semantically intact while being paid for
    once per pair.
    """

    __tablename__ = "attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # TODO(phase-3): foreign key to attacks.id once the archive table exists.
    attack_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    # The DefenseConfig fingerprint, not a surrogate key: it is order-independent,
    # so semantically identical configs share one cache entry.
    # TODO(phase-5): foreign key to defense_configs.id once that table exists.
    defense_config_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # TODO(phase-6): foreign key to rounds.id once that table exists.
    round_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    vector: Mapped[str] = mapped_column(String(32), nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    #: Which tier decided. Kept as a column so restoring the judge (D4) is a
    #: drop-in rather than a migration.
    tier: Mapped[int | None] = mapped_column(Integer, nullable=True)
    canaries_hit: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, server_default=text("'{}'")
    )
    unauthorized_tools: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, server_default=text("'{}'")
    )
    #: Privileged tools the model asked for and the structural layer stopped.
    #: Scoring is unchanged — these attempts are `blocked` — but the Phase 8
    #: layer-ablation table needs to say what Layer 5 actually caught, which is
    #: not the same question as "was the model never hijacked at all".
    blocked_tools: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, server_default=text("'{}'")
    )
    judge_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    judge_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Everything `replay(attempt_id)` needs, including this attempt's canary
    #: values. Never logged, never put in a report unredacted.
    trace: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("attack_id", "defense_config_id", name="uq_attempts_attack_defense"),
    )


class AttackRow(Base):
    """The archive: every attack ever generated.

    Unlike `target_documents`, this table **does** carry an HNSW index. Novelty
    is a heuristic score over a large and growing table, and it tolerates
    approximation in a way retrieval did not: a slightly wrong mean distance
    shifts a ranking, whereas a missed document silently fails to deliver an
    attack. `AttackRepository.nearest_neighbours` still pins `hnsw.ef_search`
    high and breaks ties on id, so the same archive always produces the same
    novelty ordering.
    """

    __tablename__ = "attacks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Lineage. NULL for seeds; every other attack is a mutation of an elite.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("attacks.id", ondelete="SET NULL"), nullable=True
    )
    #: The second parent, for operators that combine two attacks.
    recombined_with: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("attacks.id", ondelete="SET NULL"), nullable=True
    )
    #: Which named mutation operator produced this attack. NULL for seeds.
    mutation_operator: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    #: Only the two executable vectors are storable; D3 vectors are rejected.
    vector: Mapped[str] = mapped_column(String(32), nullable=False)
    #: NULL when the classifier returned out-of-enum twice: `unclassified`.
    objective: Mapped[str | None] = mapped_column(String(32), nullable=True)
    technique: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: NULL for an unclassified attack, which occupies no cell.
    cell_key: Mapped[str | None] = mapped_column(String(96), nullable=True, index=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    #: Novelty at the time of generation, against the archive as it then was.
    novelty_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 5), nullable=True)
    first_breach_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_breaches: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Never shown to an agent. Enforced in the repository, not by convention.
    is_holdout: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retired: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    benign_user_input: Mapped[str | None] = mapped_column(Text, nullable=True)
    carrier_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    carrier_doc_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "vector IN ('direct', 'indirect_document')",
            name="executable_vector",
        ),
        Index(
            "ix_attacks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_attacks_is_holdout", "is_holdout"),
    )


class CellRow(Base):
    """One MAP-Elites cell and the elite that holds it."""

    __tablename__ = "cells"

    cell_key: Mapped[str] = mapped_column(String(96), primary_key=True)
    elite_attack_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("attacks.id", ondelete="SET NULL"), nullable=True
    )
    elite_fitness: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    occupancy: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_updated_round: Mapped[int | None] = mapped_column(Integer, nullable=True)


class RejectionRow(Base):
    """An attack the novelty filter refused before it could be executed.

    Kept so that `crucible archive stats` can report a rejection rate and so
    that Phase 6's collapse detection can see a rediscovery-only round. The
    payload is stored only as a hash: a rejected attack is a near-duplicate of
    one already archived, and a shadow corpus of rejects has no use.
    """

    __tablename__ = "novelty_rejections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    novelty_score: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    threshold: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    nearest_neighbour_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("attacks.id", ondelete="SET NULL"), nullable=True
    )
    nearest_distance: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RunRow(Base):
    """One co-evolutionary run.

    A halted run is a valid experiment: `status` distinguishes `halted` from
    `failed`, and `halt_reason` says which signal in docs/spec.md section 14
    tripped.
    """

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    attacker_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    rounds_planned: Mapped[int] = mapped_column(Integer, nullable=False)
    rounds_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: D(0). The technical director's decision: always the empty config, so the
    #: loop starts weak and can be shown to harden.
    starting_config_id: Mapped[str] = mapped_column(String(64), nullable=False)
    current_config_id: Mapped[str] = mapped_column(String(64), nullable=False)
    budget_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    halt_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: The full configuration of the run, so a resume rebuilds the same loop.
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RoundRow(Base):
    """One round of one run.

    Rates are stored with their Wilson bounds *and* their counts, so a
    `RoundReport` reconstructs exactly: a rate without an interval is not a
    reportable number (docs/spec.md section 15).
    """

    __tablename__ = "rounds"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    attacker_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    defense_before: Mapped[str] = mapped_column(String(64), nullable=False)
    defense_after: Mapped[str] = mapped_column(String(64), nullable=False)

    attacks_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attacks_rejected_novelty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    breaches_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    archive_successes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    archive_trials: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    archive_block_rate: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    archive_block_rate_ci_low: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    archive_block_rate_ci_high: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)

    holdout_successes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    holdout_trials: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    holdout_block_rate: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    holdout_ci_low: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    holdout_ci_high: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)

    #: archive - holdout. This is the overfitting measure.
    overfit_gap: Mapped[Decimal] = mapped_column(Numeric(7, 5), nullable=False)

    utility_successes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    utility_trials: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    utility_pass_rate: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)

    mean_novelty: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False, default=0)
    cells_occupied: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    new_cells: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Attacks the newly selected config reopened, with their ids.
    regressions: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    config_promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False, default=0)
    halt_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (UniqueConstraint("run_id", "round_number", name="uq_rounds_run_round"),)


class DefenseConfigRow(Base):
    """Every `DefenseConfig` the loop has ever used, addressable by id.

    The id is the config's own content fingerprint, which is what the outcome
    cache keys on, so a config is stored exactly once however many rounds use
    it. Phase 8's layer ablation replays the final archive against configs with
    individual layers disabled, and the cross-round ablation compares configs by
    id — neither is possible without this table.
    """

    __tablename__ = "defense_configs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    #: The round that selected it. NULL for D(0) and for hand-written configs.
    round_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    #: The config this one was derived from. NULL for D(0).
    parent_config_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("defense_configs.id", ondelete="SET NULL"), nullable=True
    )
    label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
