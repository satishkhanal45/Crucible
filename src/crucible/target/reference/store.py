"""pgvector-backed document store for the reference target."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from crucible.db.session import Database
from crucible.logging import get_logger
from crucible.repositories.documents import BASE_NAMESPACE, TargetDocumentRepository
from crucible.schemas.corpus import DocumentRecord, RetrievedDocument
from crucible.services.embeddings import Embedder
from crucible.target.adapter import Document

logger = get_logger(__name__)

#: Retrieval default. The distance threshold comes from the embedder, since it
#: is a property of the vector space (see `Embedder.max_relevant_distance`).
DEFAULT_TOP_K = 4


class RestoreReport(BaseModel):
    """What `reset()` had to undo."""

    model_config = ConfigDict(frozen=True)

    removed: tuple[str, ...] = ()
    restored: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.removed or self.restored)


class DocumentStore:
    """Embeds, stores, and retrieves documents. Sessions live in the target."""

    def __init__(
        self,
        database: Database,
        embedder: Embedder,
        *,
        namespace: str = BASE_NAMESPACE,
        top_k: int = DEFAULT_TOP_K,
        max_distance: float | None = None,
    ) -> None:
        self._database = database
        self._embedder = embedder
        self._namespace = namespace
        self.top_k = top_k
        self.max_distance = (
            max_distance if max_distance is not None else embedder.max_relevant_distance
        )

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def namespace(self) -> str:
        return self._namespace

    def for_namespace(self, namespace: str) -> DocumentStore:
        """A view of a private corpus copy, for one concurrent executor worker."""
        return DocumentStore(
            self._database,
            self._embedder,
            namespace=namespace,
            top_k=self.top_k,
            max_distance=self.max_distance,
        )

    async def records_for(
        self, documents: Sequence[Document], *, injected: bool = False
    ) -> tuple[DocumentRecord, ...]:
        """Embed a batch of documents in one pass."""
        if not documents:
            return ()
        vectors = await self._embedder.embed(
            [f"{document.title}\n\n{document.text}" for document in documents]
        )
        return tuple(
            DocumentRecord(document=document, embedding=tuple(vector), injected=injected)
            for document, vector in zip(documents, vectors, strict=True)
        )

    async def load(self, records: Sequence[DocumentRecord]) -> int:
        async with self._database.session() as session:
            return await TargetDocumentRepository(session, self._namespace).upsert_many(records)

    async def upsert_document(self, document: Document, *, injected: bool = False) -> str:
        (record,) = await self.records_for([document], injected=injected)
        async with self._database.session() as session:
            return await TargetDocumentRepository(session, self._namespace).upsert(record)

    async def search(
        self, query: str, *, k: int | None = None, max_distance: float | None = None
    ) -> list[RetrievedDocument]:
        """Top-k relevant, non-confidential documents for a query."""
        embedding = await self._embedder.embed_one(query)
        async with self._database.session() as session:
            return await TargetDocumentRepository(session, self._namespace).search(
                embedding,
                k=k if k is not None else self.top_k,
                max_distance=max_distance if max_distance is not None else self.max_distance,
            )

    async def get(self, doc_id: str) -> Document | None:
        async with self._database.session() as session:
            return await TargetDocumentRepository(session, self._namespace).get(doc_id)

    async def delete(self, doc_id: str) -> bool:
        async with self._database.session() as session:
            return await TargetDocumentRepository(session, self._namespace).delete(doc_id)

    async def count(self) -> int:
        async with self._database.session() as session:
            return await TargetDocumentRepository(session, self._namespace).count()

    async def documents(self, *, include_confidential: bool = True) -> list[Document]:
        async with self._database.session() as session:
            return await TargetDocumentRepository(session, self._namespace).list_documents(
                include_confidential=include_confidential
            )

    async def clear(self) -> int:
        async with self._database.session() as session:
            return await TargetDocumentRepository(session, self._namespace).clear()

    async def restore(self, pristine: Sequence[DocumentRecord]) -> RestoreReport:
        """Bring the corpus back to the snapshot, touching only what changed."""
        expected = {record.document.doc_id: record for record in pristine}
        async with self._database.session() as session:
            repository = TargetDocumentRepository(session, self._namespace)
            current = await repository.fingerprints()

            extra = [doc_id for doc_id in current if doc_id not in expected]
            await repository.delete_many(extra)

            stale = [
                record
                for doc_id, record in expected.items()
                if current.get(doc_id) != record.content_hash
            ]
            await repository.upsert_many(stale)

        return RestoreReport(
            removed=tuple(sorted(extra)),
            restored=tuple(sorted(record.document.doc_id for record in stale)),
        )
