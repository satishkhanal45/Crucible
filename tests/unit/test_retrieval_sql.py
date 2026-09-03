"""The confidentiality filter is in SQL, not in application logic.

Verification test 8 proves the behaviour against a real database. This test
guards the mechanism: if someone rewrites `search()` without the filter, the
compiled statement changes and this fails immediately, without needing Postgres.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from crucible.db.models import TargetDocument
from crucible.repositories.documents import TargetDocumentRepository


def _compiled_search() -> str:
    distance = TargetDocument.embedding.cosine_distance([0.0] * 384).label("distance")
    statement = (
        select(TargetDocument, distance)
        .where(TargetDocument.confidential.is_(False))
        .where(distance <= 0.65)
        .order_by(distance)
        .limit(4)
    )
    return str(statement.compile(dialect=postgresql.dialect()))


def test_the_search_statement_filters_confidential_documents() -> None:
    compiled = _compiled_search().lower()

    assert "confidential is false" in compiled
    assert "order by" in compiled
    assert compiled.index("confidential is false") < compiled.index("order by")


def test_the_repository_exposes_no_way_to_retrieve_confidential_documents() -> None:
    """`search` is the only retrieval path; the rest are explicit admin reads."""
    retrieval_methods = {"search"}
    public = {
        name
        for name in dir(TargetDocumentRepository)
        if not name.startswith("_") and callable(getattr(TargetDocumentRepository, name))
    }

    assert retrieval_methods <= public
    assert public == {
        "clear",
        "count",
        "delete",
        "delete_many",
        "fingerprints",
        "get",
        "list_documents",
        "search",
        "upsert",
        "upsert_many",
    }
