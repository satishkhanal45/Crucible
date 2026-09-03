"""The MAP-Elites grid: cells, coverage, and elites.

Six objectives x two executable vectors x eight techniques = **96 cells**
(docs/spec.md section 6). The denominator is 96 and not 192 because cut B3
leaves only `direct` and `indirect_document` executable; reporting coverage
against 192 would understate diversity by half.

Coverage is never reported as a bare number. `Coverage` renders as
"37/96 (38.5%)" so that no caller can accidentally print an occupancy count
without the denominator beside it.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from crucible.schemas.taxonomy import (
    EXECUTABLE_VECTORS,
    GRID_DENOMINATOR,
    DeliveryVector,
    Objective,
    Technique,
    cell_key,
)

CELL_SEPARATOR = "|"


class InvalidCellKey(ValueError):
    """A cell key whose axes are not in the taxonomy. Never silently bucketed."""


def all_cell_keys() -> tuple[str, ...]:
    """Every cell in the grid, in a stable order."""
    return tuple(
        cell_key(objective, vector, technique)
        for objective in Objective
        for vector in sorted(EXECUTABLE_VECTORS)
        for technique in Technique
    )


def parse_cell_key(key: str) -> tuple[Objective, DeliveryVector, Technique]:
    """Split a cell key back into its axes, rejecting anything off-taxonomy."""
    parts = key.split(CELL_SEPARATOR)
    if len(parts) != 3:
        raise InvalidCellKey(
            f"cell key {key!r} must be 'objective{CELL_SEPARATOR}vector{CELL_SEPARATOR}technique'"
        )
    raw_objective, raw_vector, raw_technique = parts
    try:
        objective = Objective(raw_objective)
        vector = DeliveryVector(raw_vector)
        technique = Technique(raw_technique)
    except ValueError as error:
        raise InvalidCellKey(f"cell key {key!r} is not in the taxonomy: {error}") from error
    if vector not in EXECUTABLE_VECTORS:
        raise InvalidCellKey(
            f"cell key {key!r} uses vector {vector.value!r}, which is deferred (D3) "
            "and therefore has no cell in this build's grid"
        )
    return objective, vector, technique


def is_valid_cell_key(key: str) -> bool:
    try:
        parse_cell_key(key)
    except InvalidCellKey:
        return False
    return True


class Coverage(BaseModel):
    """Occupied cells out of the grid. Always rendered with its denominator."""

    model_config = ConfigDict(frozen=True)

    occupied: int
    denominator: int = GRID_DENOMINATOR

    @property
    def fraction(self) -> float:
        return self.occupied / self.denominator if self.denominator else 0.0

    def __str__(self) -> str:
        return f"{self.occupied}/{self.denominator} ({self.fraction:.1%})"


def coverage(cell_keys: Iterable[str | None]) -> Coverage:
    """Distinct occupied cells. `None` (unclassified) occupies nothing."""
    occupied = {key for key in cell_keys if key is not None and is_valid_cell_key(key)}
    return Coverage(occupied=len(occupied))


class Elite(BaseModel):
    """The best attack in one cell."""

    model_config = ConfigDict(frozen=True)

    cell_key: str
    attack_id: str
    fitness: float


def displaces(current: float | None, candidate: float) -> bool:
    """Whether a candidate's fitness takes the cell.

    Strictly greater: an equal-fitness newcomer does not displace the incumbent,
    so elites are stable and the mutation pool does not churn on ties.
    """
    return current is None or candidate > current
