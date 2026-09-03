"""Embeddings, as a service.

Phase 1 uses these for retrieval; Phase 3 reuses the same service for novelty
scoring, which is why it lives here and not inside the target.

`all-MiniLM-L6-v2` runs on CPU, produces 384-dimension vectors, and is loaded
once per process (the model costs ~400MB resident and several seconds to load).
Everything is batched: `Sequence[str]` in, one vector per input out.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Sequence
from functools import lru_cache
from hashlib import blake2b
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from crucible.config import Settings
from crucible.db.models import EMBEDDING_DIMENSIONS
from crucible.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cost is paid only at runtime
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32

#: Cosine-distance floor for "relevant". It is a property of the vector space,
#: not of the application, so each embedder declares its own.
MINILM_MAX_RELEVANT_DISTANCE = 0.65
HASHING_MAX_RELEVANT_DISTANCE = 0.85

#: Model name that selects the dependency-free `HashingEmbedder`.
HASHING_MODEL_NAME = "hashing-bow-384"

Vector = list[float]


@runtime_checkable
class Embedder(Protocol):
    """The embedding surface the rest of Crucible depends on."""

    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def max_relevant_distance(self) -> float:
        """Cosine distance beyond which a hit is not worth retrieving."""
        ...

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a batch. One vector per input, in input order."""
        ...

    async def embed_one(self, text: str) -> Vector: ...


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Load and cache a model process-wide. Import is deferred: torch is heavy."""
    from sentence_transformers import SentenceTransformer

    logger.info("embeddings.loading_model", extra={"embedding_model": model_name})
    return SentenceTransformer(model_name, device="cpu")


class SentenceTransformerEmbedder:
    """CPU sentence-transformers embeddings, batched, L2-normalised."""

    def __init__(self, model_name: str, *, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._model_name = model_name
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return EMBEDDING_DIMENSIONS

    @property
    def max_relevant_distance(self) -> float:
        return MINILM_MAX_RELEVANT_DISTANCE

    def _encode(self, texts: list[str]) -> list[Vector]:
        model = _load_sentence_transformer(self._model_name)
        encoded = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors: list[Vector] = [[float(value) for value in row] for row in encoded]
        if vectors and len(vectors[0]) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"{self._model_name} produced {len(vectors[0])}-dimension vectors, "
                f"but the schema stores {EMBEDDING_DIMENSIONS}"
            )
        return vectors

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        # Encoding is CPU-bound and releases the GIL inside torch; keep the loop free.
        return await asyncio.to_thread(self._encode, list(texts))

    async def embed_one(self, text: str) -> Vector:
        return (await self.embed([text]))[0]


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words hashing embeddings. No model, no download.

    Used by tests and by offline development: it is genuinely lexical, so
    retrieval, thresholds, and cosine ordering behave sensibly without pulling a
    400MB model into every test run. It is not a semantic model and is never the
    default for a real run.
    """

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return HASHING_MODEL_NAME

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def max_relevant_distance(self) -> float:
        """Higher than a semantic model: lexical overlap scores lower."""
        return HASHING_MAX_RELEVANT_DISTANCE

    def _encode_one(self, text: str) -> Vector:
        counts: dict[str, int] = {}
        for token in _TOKEN.findall(text.lower()):
            counts[token] = counts.get(token, 0) + 1

        vector = [0.0] * self._dimensions
        for token, count in counts.items():
            digest = blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._encode_one(text) for text in texts]

    async def embed_one(self, text: str) -> Vector:
        return self._encode_one(text)


@lru_cache(maxsize=4)
def get_embedder(model_name: str) -> Embedder:
    """The process-wide embedder for a model name."""
    if model_name == HASHING_MODEL_NAME:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)


def embedder_from_settings(settings: Settings) -> Embedder:
    return get_embedder(settings.EMBEDDING_MODEL)
