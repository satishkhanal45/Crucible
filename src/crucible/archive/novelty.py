"""Novelty: the anti-collapse mechanism.

    novelty(a) = mean cosine distance from a's embedding to its k = 15 nearest
                 archive neighbours, excluding a itself
               = 1.0 when the archive holds fewer than k attacks

Attacks below `MIN_NOVELTY` are rejected **before execution**. That is the
primary defence against an attacker collapsing onto three working templates,
and it is also what keeps the budget from being spent re-running rediscoveries.
Every rejection records the nearest neighbour's id, so a report can show that
rejection is working rather than asserting it.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from hashlib import sha256

from pydantic import BaseModel, ConfigDict

#: docs/spec.md section 10.
K_NEIGHBOURS = 15
MIN_NOVELTY = 0.15


class Neighbour(BaseModel):
    """One archive neighbour and its cosine distance."""

    model_config = ConfigDict(frozen=True)

    attack_id: uuid.UUID
    distance: float


class NoveltyScore(BaseModel):
    """A novelty computation, with the evidence that produced it."""

    model_config = ConfigDict(frozen=True)

    value: float
    archive_size: int
    k: int = K_NEIGHBOURS
    neighbours: tuple[Neighbour, ...] = ()

    @property
    def nearest(self) -> Neighbour | None:
        return self.neighbours[0] if self.neighbours else None

    def is_novel(self, threshold: float = MIN_NOVELTY) -> bool:
        return self.value >= threshold


class NoveltyRejection(BaseModel):
    """Why an attack never reached the executor."""

    model_config = ConfigDict(frozen=True)

    novelty: float
    threshold: float
    #: The archived attack this one was too close to.
    nearest_neighbour_id: uuid.UUID | None
    nearest_distance: float | None
    #: The payload is not stored: a rejected attack is a near-duplicate of one
    #: already in the archive, and a shadow corpus of rejects has no use.
    payload_hash: str
    round_number: int = 0

    def __str__(self) -> str:
        neighbour = self.nearest_neighbour_id or "<none>"
        return f"novelty {self.novelty:.3f} < {self.threshold:.3f}; nearest neighbour {neighbour}"


def payload_fingerprint(payload: str) -> str:
    return sha256(payload.encode()).hexdigest()


def score(
    neighbours: Sequence[Neighbour], archive_size: int, k: int = K_NEIGHBOURS
) -> NoveltyScore:
    """Mean distance to the k nearest neighbours, or 1.0 for a small archive."""
    if archive_size < k:
        return NoveltyScore(
            value=1.0, archive_size=archive_size, k=k, neighbours=tuple(neighbours[:k])
        )
    nearest = tuple(neighbours[:k])
    value = sum(neighbour.distance for neighbour in nearest) / len(nearest) if nearest else 1.0
    return NoveltyScore(
        value=max(0.0, min(1.0, value)), archive_size=archive_size, k=k, neighbours=nearest
    )
