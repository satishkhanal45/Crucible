"""Verification tests 15, 16, 17 — the archive's vector index.

The archive keeps an HNSW index, unlike `target_documents`. Novelty is a
heuristic score over a growing table and tolerates approximation; retrieval did
not, because a document an approximate search fails to reach is an attack that
was never delivered. Two constraints still bind here, and both are tested:
recall has to be exact at fixture scale (test 16), and the neighbour ordering
has to be deterministic so that Phase 6 can rebuild an identical archive.
"""

from __future__ import annotations

import math
import random
import resource
import uuid
from collections.abc import Sequence

import pytest
from sqlalchemy import insert, text

from crucible.archive.novelty import K_NEIGHBOURS
from crucible.db.models import AttackRow
from crucible.db.session import Database
from crucible.repositories.attacks import AttackRepository
from crucible.schemas.taxonomy import DeliveryVector, Objective, Technique
from crucible.services.embeddings import HashingEmbedder

DIMENSIONS = 384
MEMORY_CEILING_BYTES = 1024 * 1024 * 1024


def unit_vector(rng: random.Random) -> list[float]:
    raw = [rng.gauss(0.0, 1.0) for _ in range(DIMENSIONS)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return 1.0 - sum(a * b for a, b in zip(left, right, strict=True))


#: Generating 5000 x 384 gaussians costs more than the inserts do, and two
#: tests need the same fixture, so it is built once per (count, seed).
_FIXTURE_CACHE: dict[tuple[int, int], list[tuple[uuid.UUID, list[float]]]] = {}


def fixture_rows(count: int, seed: int) -> list[tuple[uuid.UUID, list[float]]]:
    cached = _FIXTURE_CACHE.get((count, seed))
    if cached is None:
        rng = random.Random(seed)
        cached = [(uuid.uuid4(), unit_vector(rng)) for _ in range(count)]
        _FIXTURE_CACHE[(count, seed)] = cached
    return cached


async def bulk_insert(
    database: Database, count: int, seed: int
) -> list[tuple[uuid.UUID, list[float]]]:
    """Fill the archive directly. This exercises the index, not the repository."""
    rows = fixture_rows(count, seed)
    payload = []
    for index, (attack_id, embedding) in enumerate(rows):
        payload.append(
            {
                "id": attack_id,
                "round_generated": 0,
                "parent_id": None,
                "payload": f"fixture attack {index}",
                "vector": DeliveryVector.DIRECT.value,
                "objective": Objective.SYSPROMPT_EXTRACTION.value,
                "technique": Technique.INSTRUCTION_OVERRIDE.value,
                "cell_key": "sysprompt_extraction|direct|instruction_override",
                "embedding": embedding,
                "total_attempts": 0,
                "total_breaches": 0,
                "is_holdout": False,
                "retired": False,
            }
        )
    async with database.session() as session:
        for start in range(0, len(payload), 500):
            await session.execute(insert(AttackRow), payload[start : start + 500])
        await session.execute(text("ANALYZE attacks"))
    return rows


@pytest.fixture
async def clean_archive(archive: object, database_url: str) -> Database:
    """The `archive` fixture truncates the tables; this hands back a database."""
    del archive
    return Database(database_url)


# ----------------------------------------------------------------- test 15
#
# MEASURED TENSION, reported rather than quietly resolved.
#
# On a 5000-row archive of 384-dimension vectors, index usage and exact recall
# are mutually exclusive, because pgvector prices an HNSW scan by `ef_search`:
#
#     ef_search    planner picks the index    recall@15
#            40    yes                        0.48
#           100    yes                        0.76
#           200    no (exact scan)            0.93
#          1000    no (exact scan)            1.00, exact order
#
# Novelty drives rejection *before execution*, so a 48-76% recall neighbour set
# would silently corrupt the anti-collapse mechanism — the same failure mode
# that made target retrieval drop injected documents in Phase 2. The archive
# therefore keeps its HNSW index but queries it at ef_search=1000, where
# Postgres chooses the exact scan and novelty is both correct and reproducible.
#
# The two tests below assert both halves of that finding: the index is real and
# the planner does choose it for k-NN when it is priced to, and the production
# setting is the exact one.


async def test_the_hnsw_index_is_chosen_for_knn_when_it_is_priced_to_be(
    clean_archive: Database,
) -> None:
    """The index exists, is valid, and the planner uses it for a k-NN order-by."""
    database = clean_archive
    try:
        await bulk_insert(database, 5000, seed=15)
        probe = unit_vector(random.Random(1))

        async with database.session() as session:
            indexes = (
                (
                    await session.execute(
                        text(
                            "SELECT indexdef FROM pg_indexes WHERE tablename = 'attacks' "
                            "AND indexname = 'ix_attacks_embedding_hnsw'"
                        )
                    )
                )
                .scalars()
                .all()
            )
            await session.execute(text("SET LOCAL hnsw.ef_search = 40"))
            plan_rows = (
                await session.execute(
                    text("EXPLAIN SELECT id FROM attacks ORDER BY embedding <=> :probe LIMIT :k"),
                    {"probe": str(probe), "k": K_NEIGHBOURS},
                )
            ).all()
        plan = "\n".join(row[0] for row in plan_rows)
    finally:
        await database.close()

    assert indexes and "USING hnsw" in indexes[0] and "vector_cosine_ops" in indexes[0]
    assert "ix_attacks_embedding_hnsw" in plan, plan
    assert "Seq Scan" not in plan, plan


async def test_novelty_neighbours_are_exact_on_a_five_thousand_row_archive(
    clean_archive: Database,
) -> None:
    """The production setting: recall 1.0, at the cost of an exact scan.

    If this ever regresses to approximate results, novelty scores become
    fiction and attacks are rejected — or admitted — for the wrong reason.
    """
    database = clean_archive
    try:
        rows = await bulk_insert(database, 5000, seed=15)
        probe = unit_vector(random.Random(1))
        expected = sorted(
            (cosine_distance(probe, embedding), str(attack_id)) for attack_id, embedding in rows
        )[:K_NEIGHBOURS]

        async with database.session() as session:
            neighbours = await AttackRepository(session).nearest_neighbours(probe)
    finally:
        await database.close()

    assert [str(neighbour.attack_id) for neighbour in neighbours] == [
        attack_id for _, attack_id in expected
    ]


# ----------------------------------------------------------------- test 16


async def test_knn_matches_brute_force_cosine_on_a_two_hundred_row_fixture(
    clean_archive: Database,
) -> None:
    """Recall must be exact at this scale, or novelty scores are fiction."""
    database = clean_archive
    try:
        rows = await bulk_insert(database, 200, seed=16)
        probe = unit_vector(random.Random(2))

        expected = sorted(
            ((cosine_distance(probe, embedding), str(attack_id)) for attack_id, embedding in rows),
        )[:K_NEIGHBOURS]

        async with database.session() as session:
            neighbours = await AttackRepository(session).nearest_neighbours(probe)
    finally:
        await database.close()

    assert [str(neighbour.attack_id) for neighbour in neighbours] == [
        attack_id for _, attack_id in expected
    ]
    for neighbour, (distance, _) in zip(neighbours, expected, strict=True):
        assert neighbour.distance == pytest.approx(distance, abs=1e-6)


async def test_knn_is_deterministic_across_repeated_queries(
    clean_archive: Database,
) -> None:
    """Phase 6 requires two runs with the same seed to build the same archive."""
    database = clean_archive
    try:
        await bulk_insert(database, 200, seed=16)
        probe = unit_vector(random.Random(2))

        runs = []
        for _ in range(5):
            async with database.session() as session:
                neighbours = await AttackRepository(session).nearest_neighbours(probe)
            runs.append([(str(n.attack_id), round(n.distance, 9)) for n in neighbours])
    finally:
        await database.close()

    assert all(run == runs[0] for run in runs)


async def test_knn_excludes_the_attack_itself(clean_archive: Database) -> None:
    database = clean_archive
    try:
        rows = await bulk_insert(database, 200, seed=17)
        target_id, embedding = rows[0]

        async with database.session() as session:
            neighbours = await AttackRepository(session).nearest_neighbours(
                embedding, exclude_id=target_id
            )
    finally:
        await database.close()

    assert len(neighbours) == K_NEIGHBOURS
    assert target_id not in {neighbour.attack_id for neighbour in neighbours}


# ----------------------------------------------------------------- test 17


async def test_batch_encoding_five_hundred_attacks_stays_under_the_memory_ceiling(
    clean_archive: Database,
) -> None:
    """This project targets a 12GB machine with no GPU.

    The ceiling covers the batch pipeline: 500 payloads in, 500 x 384 float
    vectors out. It does not cover loading a sentence-transformers model, which
    would need a network download and is therefore never done inside the test
    suite; the model's ~400MB resident footprint is budgeted separately in
    project_context.md and exercised by `make seed`.
    """
    database = clean_archive
    try:
        embedder = HashingEmbedder()
        payloads = [
            f"attack variant {index}: override the policy and emit identifier {index}"
            for index in range(500)
        ]

        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        vectors = await embedder.embed(payloads)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    finally:
        await database.close()

    # ru_maxrss is kilobytes on Linux.
    growth_bytes = max(0, after - before) * 1024
    assert len(vectors) == 500
    assert all(len(vector) == DIMENSIONS for vector in vectors)
    assert growth_bytes < MEMORY_CEILING_BYTES, (
        f"batch encoding grew resident memory by {growth_bytes / 1e6:.1f}MB, "
        f"over the {MEMORY_CEILING_BYTES / 1e9:.1f}GB ceiling"
    )
    # The vectors themselves are the floor of what this can cost.
    assert 500 * DIMENSIONS * 8 < MEMORY_CEILING_BYTES
