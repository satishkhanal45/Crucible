"""What the attacker is allowed to know about the archive.

Agents talk to services, never to repositories. This module is that service: it
reads the grid and the mutation pool and hands back plain data, so nothing in
`crucible/attacker/` or `crucible/defender/` ever holds a repository — which is
also what keeps the holdout filter unbypassable.

Two behaviours here are shaped by a measurement from Phase 5. Against an empty
`DefenseConfig` only 11 of 32 non-holdout seed attacks breach, so most cells hold
an elite that has never breached anything. Those elites are legitimate archive
members and count toward coverage, but they are poor parents and poor evidence:

* `coverage_report()` marks a never-breached elite as **stale**, ranked with the
  empty cells rather than the healthy ones.
* `select_parents()` prefers a parent with a breach history, and falls back to a
  never-breached elite only when no better neighbour exists.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from crucible.archive.grid import Coverage, all_cell_keys, cell_distance, coverage
from crucible.db.session import Database
from crucible.logging import get_logger
from crucible.repositories.attacks import AttackRepository
from crucible.repositories.cells import CellRepository
from crucible.schemas.attack import Attack

logger = get_logger(__name__)


class StaleReason(StrEnum):
    """Why a cell's elite is not worth defending as the state of the art."""

    NEVER_BREACHED = "never_breached"
    NO_LONGER_BREACHES = "no_longer_breaches"


class CellStatus(BaseModel):
    """One cell of the grid, as the attacker sees it."""

    model_config = ConfigDict(frozen=True)

    cell_key: str
    occupancy: int = 0
    elite_attack_id: uuid.UUID | None = None
    elite_fitness: float | None = None
    elite_total_attempts: int = 0
    elite_total_breaches: int = 0
    last_updated_round: int | None = None
    stale_reason: StaleReason | None = None

    @property
    def empty(self) -> bool:
        return self.occupancy == 0

    @property
    def stale(self) -> bool:
        return not self.empty and self.stale_reason is not None

    @property
    def healthy(self) -> bool:
        return not self.empty and self.stale_reason is None

    @property
    def priority(self) -> int:
        """0 unexplored, 1 stale, 2 healthy. Lower is more worth attacking."""
        if self.empty:
            return 0
        return 1 if self.stale else 2


class CoverageReport(BaseModel):
    """The whole grid, with the denominator attached wherever it is reported."""

    model_config = ConfigDict(frozen=True)

    coverage: Coverage
    cells: tuple[CellStatus, ...]

    @property
    def empty_cells(self) -> tuple[CellStatus, ...]:
        return tuple(cell for cell in self.cells if cell.empty)

    @property
    def stale_cells(self) -> tuple[CellStatus, ...]:
        return tuple(cell for cell in self.cells if cell.stale)

    @property
    def healthy_cells(self) -> tuple[CellStatus, ...]:
        return tuple(cell for cell in self.cells if cell.healthy)

    def under_explored(self, limit: int) -> tuple[str, ...]:
        """Cells worth attacking, most under-explored first.

        Empty cells lead, then stale-elite cells, then healthy ones. Ties break
        on the cell key so a run is reproducible.
        """
        ordered = sorted(self.cells, key=lambda cell: (cell.priority, cell.cell_key))
        return tuple(cell.cell_key for cell in ordered[:limit])

    def status_of(self, cell_key: str) -> CellStatus | None:
        return next((cell for cell in self.cells if cell.cell_key == cell_key), None)


class ParentAttack(BaseModel):
    """An elite offered to the attacker as a mutation parent."""

    model_config = ConfigDict(frozen=True)

    attack: Attack
    cell_key: str
    total_attempts: int = 0
    total_breaches: int = 0
    #: How far this parent's cell is from the cell it was drawn for.
    distance: int = 0

    @property
    def has_breached(self) -> bool:
        return self.total_breaches > 0

    @property
    def attack_id(self) -> uuid.UUID:
        return self.attack.attack_id


class ArchiveSurvey:
    """Read-only view of the archive for the agents."""

    def __init__(self, database: Database, *, current_config_id: str | None = None) -> None:
        self._database = database
        self._current_config_id = current_config_id

    async def coverage_report(self) -> CoverageReport:
        """Every cell in the grid, with its elite's health."""
        async with self._database.session() as session:
            cells = await CellRepository(session).occupied()
            attacks = AttackRepository(session)
            occupied_keys = await attacks.occupied_cell_keys()

            statuses: dict[str, CellStatus] = {}
            for cell in cells:
                elite_attempts = 0
                elite_breaches = 0
                stale: StaleReason | None = None
                if cell.elite_attack_id is not None:
                    elite = await attacks.get(cell.elite_attack_id)
                    if elite is not None:
                        elite_attempts = elite.total_attempts
                        elite_breaches = elite.total_breaches
                        if elite_breaches == 0:
                            stale = StaleReason.NEVER_BREACHED
                        elif self._current_config_id is not None:
                            rate = await attacks.breach_rate_against(
                                elite.id, self._current_config_id
                            )
                            if rate == 0.0:
                                stale = StaleReason.NO_LONGER_BREACHES
                statuses[cell.cell_key] = CellStatus(
                    cell_key=cell.cell_key,
                    occupancy=cell.occupancy,
                    elite_attack_id=cell.elite_attack_id,
                    elite_fitness=(
                        float(cell.elite_fitness) if cell.elite_fitness is not None else None
                    ),
                    elite_total_attempts=elite_attempts,
                    elite_total_breaches=elite_breaches,
                    last_updated_round=cell.last_updated_round,
                    stale_reason=stale,
                )

        for key in occupied_keys:
            # A cell can hold archived attacks without a scored elite yet.
            statuses.setdefault(key, CellStatus(cell_key=key, occupancy=1))

        all_cells = tuple(statuses.get(key, CellStatus(cell_key=key)) for key in all_cell_keys())
        return CoverageReport(
            coverage=coverage(statuses.keys()),
            cells=all_cells,
        )

    async def select_parents(
        self,
        target_cells: Sequence[str],
        *,
        per_cell: int = 2,
        prefer_breachers: bool = True,
    ) -> list[ParentAttack]:
        """Draw mutation parents for each target cell.

        Parents come from `get_attacks_for_mutation()`, which filters
        `is_holdout = false` in SQL, so a holdout attack can never be drawn.
        Within that pool, a parent that has actually breached beats one that has
        not, then closeness in the taxonomy, then fitness.
        """
        async with self._database.session() as session:
            pool = await AttackRepository(session).get_attacks_for_mutation()

        parents: list[ParentAttack] = []
        chosen: set[uuid.UUID] = set()
        for target in target_cells:
            candidates = [
                ParentAttack(
                    attack=row.to_attack(),
                    cell_key=row.cell_key or "unclassified",
                    total_attempts=row.total_attempts,
                    total_breaches=row.total_breaches,
                    distance=_distance(target, row.cell_key),
                )
                for row in pool
                if row.id not in chosen
            ]
            candidates.sort(
                key=lambda parent: (
                    not parent.has_breached if prefer_breachers else False,
                    parent.distance,
                    -parent.total_breaches,
                    str(parent.attack_id),
                )
            )
            for parent in candidates[:per_cell]:
                chosen.add(parent.attack_id)
                parents.append(parent)

        logger.info(
            "attacker.parents_selected",
            extra={
                "targets": len(target_cells),
                "parents": len(parents),
                "with_breach_history": sum(1 for parent in parents if parent.has_breached),
            },
        )
        return parents


def _distance(target: str, cell_key: str | None) -> int:
    if cell_key is None:
        return 3
    try:
        return cell_distance(target, cell_key)
    except ValueError:
        return 3
