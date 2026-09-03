"""The only module that reads or writes `cells` and `novelty_rejections`."""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from crucible.archive.grid import displaces
from crucible.archive.novelty import NoveltyRejection
from crucible.db.models import CellRow, RejectionRow
from crucible.schemas.archive import CellRecord


class CellRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, cell_key: str) -> CellRecord | None:
        row = await self._session.get(CellRow, cell_key)
        return None if row is None else CellRecord.model_validate(row)

    async def record_occupant(
        self,
        cell_key: str,
        attack_id: uuid.UUID,
        fitness: float,
        *,
        round_number: int = 0,
        eligible_for_elite: bool = True,
    ) -> CellRecord:
        """Add an attack to a cell, taking the elite slot only on higher fitness.

        A holdout attack occupies its cell for coverage purposes but is not
        eligible to be the elite: elites are the mutation pool, and a holdout
        attack must never enter it.
        """
        statement = insert(CellRow).values(
            cell_key=cell_key,
            elite_attack_id=attack_id if eligible_for_elite else None,
            elite_fitness=Decimal(str(round(fitness, 6))) if eligible_for_elite else None,
            occupancy=1,
            last_updated_round=round_number,
        )
        upsert = statement.on_conflict_do_update(
            index_elements=[CellRow.cell_key],
            set_={
                "occupancy": CellRow.occupancy + 1,
                "last_updated_round": round_number,
            },
        ).returning(CellRow)
        row = (await self._session.execute(upsert)).scalar_one()

        if eligible_for_elite:
            current = float(row.elite_fitness) if row.elite_fitness is not None else None
            if row.elite_attack_id == attack_id or displaces(current, fitness):
                row.elite_attack_id = attack_id
                row.elite_fitness = Decimal(str(round(fitness, 6)))
                row.last_updated_round = round_number
                await self._session.flush()
                await self._session.refresh(row)
        return CellRecord.model_validate(row)

    async def occupied(self) -> list[CellRecord]:
        rows = (
            (
                await self._session.execute(
                    select(CellRow)
                    .where(CellRow.occupancy > 0)
                    .order_by(CellRow.elite_fitness.desc().nullslast(), CellRow.cell_key)
                )
            )
            .scalars()
            .all()
        )
        return [CellRecord.model_validate(row) for row in rows]

    async def count_occupied(self) -> int:
        total = await self._session.scalar(
            select(func.count()).select_from(CellRow).where(CellRow.occupancy > 0)
        )
        return int(total or 0)


class RejectionRepository:
    """Novelty rejections. Phase 6's collapse detection reads these too."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, rejection: NoveltyRejection) -> None:
        self._session.add(
            RejectionRow(
                round_number=rejection.round_number,
                novelty_score=Decimal(str(round(rejection.novelty, 5))),
                threshold=Decimal(str(round(rejection.threshold, 5))),
                nearest_neighbour_id=rejection.nearest_neighbour_id,
                nearest_distance=(
                    Decimal(str(round(rejection.nearest_distance, 6)))
                    if rejection.nearest_distance is not None
                    else None
                ),
                payload_hash=rejection.payload_hash,
            )
        )

    async def count(self, *, round_number: int | None = None) -> int:
        statement = select(func.count()).select_from(RejectionRow)
        if round_number is not None:
            statement = statement.where(RejectionRow.round_number == round_number)
        total = await self._session.scalar(statement)
        return int(total or 0)
