"""The only module that reads or writes `target_documents`.

The confidentiality rule lives here, in SQL: `search()` filters
`confidential = false` before the ORDER BY, so no value of k and no query text
can surface a confidential document.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from crucible.db.models import TargetDocument
from crucible.schemas.corpus import DocumentRecord, RetrievedDocument
from crucible.target.adapter import Document

#: Retrieval over the target corpus is an exact scan, not an approximate index
#: search: see the note on `TargetDocument`. Ties break on `doc_id` so that an
#: identical query returns an identical ordering every time, which is what makes
#: the outcome cache in docs/spec.md section 9 sound.


def _to_document(row: TargetDocument) -> Document:
    return Document(
        doc_id=row.doc_id,
        title=row.title,
        text=row.text,
        confidential=row.confidential,
        source=row.source,
    )


#: The base corpus. Executor workers use their own namespace (see the model).
BASE_NAMESPACE = ""


class TargetDocumentRepository:
    """Scoped to one corpus namespace. Nothing here can read another one."""

    def __init__(self, session: AsyncSession, namespace: str = BASE_NAMESPACE) -> None:
        self._session = session
        self._namespace = namespace

    @property
    def namespace(self) -> str:
        return self._namespace

    async def upsert(self, record: DocumentRecord) -> str:
        values = {
            "namespace": self._namespace,
            "doc_id": record.document.doc_id,
            "title": record.document.title,
            "text": record.document.text,
            "confidential": record.document.confidential,
            "source": record.document.source,
            "injected": record.injected,
            "content_hash": record.content_hash,
            "embedding": list(record.embedding),
        }
        statement = insert(TargetDocument).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[TargetDocument.namespace, TargetDocument.doc_id],
            set_={
                key: statement.excluded[key] for key in values if key not in {"namespace", "doc_id"}
            },
        )
        await self._session.execute(statement)
        return record.document.doc_id

    async def upsert_many(self, records: Sequence[DocumentRecord]) -> int:
        for record in records:
            await self.upsert(record)
        return len(records)

    async def search(
        self, embedding: Sequence[float], *, k: int, max_distance: float
    ) -> list[RetrievedDocument]:
        """Nearest non-confidential documents within the relevance threshold."""
        distance = TargetDocument.embedding.cosine_distance(list(embedding)).label("distance")
        statement = (
            select(TargetDocument, distance)
            .where(TargetDocument.namespace == self._namespace)
            .where(TargetDocument.confidential.is_(False))
            .where(distance <= max_distance)
            .order_by(distance, TargetDocument.doc_id)
            .limit(k)
        )
        rows = (await self._session.execute(statement)).all()
        return [
            RetrievedDocument(document=_to_document(row[0]), distance=float(row.distance))
            for row in rows
        ]

    async def get(self, doc_id: str) -> Document | None:
        row = await self._session.get(TargetDocument, (self._namespace, doc_id))
        return None if row is None else _to_document(row)

    async def delete(self, doc_id: str) -> bool:
        row = await self._session.get(TargetDocument, (self._namespace, doc_id))
        if row is None:
            return False
        await self._session.delete(row)
        return True

    async def delete_many(self, doc_ids: Sequence[str]) -> int:
        if not doc_ids:
            return 0
        wanted = list(doc_ids)
        present = (
            (
                await self._session.execute(
                    select(TargetDocument.doc_id)
                    .where(TargetDocument.namespace == self._namespace)
                    .where(TargetDocument.doc_id.in_(wanted))
                )
            )
            .scalars()
            .all()
        )
        if present:
            await self._session.execute(
                delete(TargetDocument)
                .where(TargetDocument.namespace == self._namespace)
                .where(TargetDocument.doc_id.in_(list(present)))
            )
        return len(present)

    async def fingerprints(self) -> dict[str, str]:
        """`doc_id -> content_hash` for every stored document."""
        rows = await self._session.execute(
            select(TargetDocument.doc_id, TargetDocument.content_hash).where(
                TargetDocument.namespace == self._namespace
            )
        )
        return {row.doc_id: row.content_hash for row in rows.all()}

    async def count(self) -> int:
        total = await self._session.scalar(
            select(func.count())
            .select_from(TargetDocument)
            .where(TargetDocument.namespace == self._namespace)
        )
        return int(total or 0)

    async def list_documents(self, *, include_confidential: bool = True) -> list[Document]:
        statement = (
            select(TargetDocument)
            .where(TargetDocument.namespace == self._namespace)
            .order_by(TargetDocument.doc_id)
        )
        if not include_confidential:
            statement = statement.where(TargetDocument.confidential.is_(False))
        rows = (await self._session.execute(statement)).scalars().all()
        return [_to_document(row) for row in rows]

    async def clear(self) -> int:
        total = await self.count()
        await self._session.execute(
            delete(TargetDocument).where(TargetDocument.namespace == self._namespace)
        )
        return total
