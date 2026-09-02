"""Verification test 6: the extension, a vector(384) round-trip, and the HNSW index."""

from __future__ import annotations

from alembic.config import Config
from sqlalchemy import delete, select, text

from crucible.db.models import EMBEDDING_DIMENSIONS, VectorSmoke
from crucible.db.session import Database

INDEX_NAME = "ix_vector_smoke_embedding_hnsw"


def _unit(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return vector


async def test_vector_extension_is_installed(database_url: str, migrated: Config) -> None:
    del migrated
    database = Database(database_url)
    try:
        async with database.session() as session:
            installed = await session.scalar(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
    finally:
        await database.close()

    assert installed == "vector"


async def test_insert_and_cosine_query_a_vector_384_row(
    database_url: str, migrated: Config
) -> None:
    del migrated
    database = Database(database_url)
    try:
        async with database.session() as session:
            await session.execute(delete(VectorSmoke))
            session.add_all(
                [
                    VectorSmoke(label="x-axis", embedding=_unit(0)),
                    VectorSmoke(label="y-axis", embedding=_unit(1)),
                    VectorSmoke(label="z-axis", embedding=_unit(2)),
                ]
            )

        async with database.session() as session:
            query = _unit(1)
            rows = (
                await session.execute(
                    select(
                        VectorSmoke.label,
                        VectorSmoke.embedding.cosine_distance(query).label("distance"),
                    ).order_by(VectorSmoke.embedding.cosine_distance(query))
                )
            ).all()
    finally:
        await database.close()

    assert [row.label for row in rows] == ["y-axis", "x-axis", "z-axis"]
    assert rows[0].distance == 0.0
    assert rows[1].distance == 1.0


async def test_hnsw_index_exists_with_cosine_ops(database_url: str, migrated: Config) -> None:
    del migrated
    database = Database(database_url)
    try:
        async with database.session() as session:
            definition = await session.scalar(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE tablename = 'vector_smoke' AND indexname = :name"
                ),
                {"name": INDEX_NAME},
            )
    finally:
        await database.close()

    assert definition is not None, f"{INDEX_NAME} is missing"
    assert "USING hnsw" in definition
    assert "vector_cosine_ops" in definition
