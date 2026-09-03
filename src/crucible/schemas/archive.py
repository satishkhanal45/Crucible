"""Boundary schemas for the archive."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from crucible.archive.grid import Coverage
from crucible.schemas.attack import Attack
from crucible.schemas.taxonomy import DeliveryVector, Objective, Technique


class NewArchivedAttack(BaseModel):
    """An attack about to enter the archive, with its embedding."""

    model_config = ConfigDict(frozen=True)

    attack: Attack
    embedding: tuple[float, ...]
    novelty_score: float | None = None
    round_generated: int = Field(default=0, ge=0)


class ArchivedAttack(BaseModel):
    """An attack as stored. The embedding is deliberately not carried around."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    round_generated: int
    parent_id: uuid.UUID | None
    recombined_with: uuid.UUID | None = None
    mutation_operator: str | None = None
    payload: str
    vector: DeliveryVector
    objective: Objective | None
    technique: Technique | None
    cell_key: str | None
    novelty_score: Decimal | None
    first_breach_round: int | None
    total_attempts: int
    total_breaches: int
    is_holdout: bool
    retired: bool
    benign_user_input: str | None
    carrier_title: str | None
    carrier_doc_id: str | None
    created_at: datetime

    @property
    def classified(self) -> bool:
        return self.objective is not None and self.technique is not None

    @property
    def breach_rate(self) -> float:
        return self.total_breaches / self.total_attempts if self.total_attempts else 0.0

    def to_attack(self) -> Attack:
        """Rebuild the executable value object."""
        return Attack(
            attack_id=self.id,
            payload=self.payload,
            vector=self.vector,
            objective=self.objective,
            technique=self.technique,
            parent_id=self.parent_id,
            recombined_with=self.recombined_with,
            mutation_operator=self.mutation_operator,
            round_generated=self.round_generated,
            benign_user_input=self.benign_user_input or "Summarize the laptop refresh policy",
            carrier_title=self.carrier_title,
            carrier_doc_id=self.carrier_doc_id,
            is_holdout=self.is_holdout,
        )


class CellRecord(BaseModel):
    """One occupied cell and its elite."""

    model_config = ConfigDict(from_attributes=True)

    cell_key: str
    elite_attack_id: uuid.UUID | None
    elite_fitness: Decimal | None
    occupancy: int
    last_updated_round: int | None


class NoveltyDistribution(BaseModel):
    """Summary of novelty across the archive."""

    model_config = ConfigDict(frozen=True)

    count: int
    minimum: float | None = None
    median: float | None = None
    mean: float | None = None
    maximum: float | None = None
    #: Share of archived attacks scoring below MIN_NOVELTY at generation time.
    below_threshold: int = 0


class ArchiveStats(BaseModel):
    """What `crucible archive stats` reports."""

    model_config = ConfigDict(frozen=True)

    archive_size: int
    holdout_count: int
    unclassified_count: int
    coverage: Coverage
    novelty: NoveltyDistribution
    rejections: int
    #: Rejections as a share of everything the novelty filter has ever seen.
    rejection_rate: float
    elites: tuple[CellRecord, ...] = ()

    @property
    def holdout_ratio(self) -> float:
        return self.holdout_count / self.archive_size if self.archive_size else 0.0
