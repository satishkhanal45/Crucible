"""Boundary schemas for the target's document store."""

from __future__ import annotations

from hashlib import sha256

from pydantic import BaseModel, ConfigDict

from crucible.target.adapter import Document


def content_hash(document: Document) -> str:
    """Stable hash of the parts `reset()` cares about."""
    payload = f"{document.title}\x00{document.text}\x00{document.confidential}"
    return sha256(payload.encode()).hexdigest()


class DocumentRecord(BaseModel):
    """A document plus everything the store needs to persist it."""

    model_config = ConfigDict(frozen=True)

    document: Document
    embedding: tuple[float, ...]
    injected: bool = False

    @property
    def content_hash(self) -> str:
        return content_hash(self.document)


class RetrievedDocument(BaseModel):
    """One retrieval hit. Confidential documents can never appear here."""

    model_config = ConfigDict(frozen=True)

    document: Document
    distance: float
