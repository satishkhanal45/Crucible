"""The only module that reads or writes `defense_configs`."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from crucible.db.models import DefenseConfigRow
from crucible.defenses.config import DefenseConfig


class DefenseConfigRecord(BaseModel):
    """A stored config, with the lineage that produced it."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    config: DefenseConfig
    round_number: int | None = None
    run_id: uuid.UUID | None = None
    parent_config_id: str | None = None
    label: str | None = None
    created_at: datetime | None = None


class DefenseConfigRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(
        self,
        config: DefenseConfig,
        *,
        round_number: int | None = None,
        run_id: uuid.UUID | None = None,
        parent_config_id: str | None = None,
        label: str | None = None,
    ) -> str:
        """Store a config under its fingerprint. Idempotent by construction."""
        fingerprint = config.fingerprint()
        values = {
            "id": fingerprint,
            "config": config.to_dict(),
            "round_number": round_number,
            "run_id": run_id,
            "parent_config_id": parent_config_id,
            "label": label,
        }
        statement = insert(DefenseConfigRow).values(**values)
        # First write wins: a config keeps the round that first produced it.
        await self._session.execute(
            statement.on_conflict_do_nothing(index_elements=[DefenseConfigRow.id])
        )
        return fingerprint

    async def get(self, config_id: str) -> DefenseConfig | None:
        row = await self._session.get(DefenseConfigRow, config_id)
        return None if row is None else DefenseConfig.model_validate(row.config)

    async def record(self, config_id: str) -> DefenseConfigRecord | None:
        row = await self._session.get(DefenseConfigRow, config_id)
        return None if row is None else DefenseConfigRecord.model_validate(row)

    async def list_for_run(self, run_id: uuid.UUID) -> list[DefenseConfigRecord]:
        rows = (
            (
                await self._session.execute(
                    select(DefenseConfigRow)
                    .where(DefenseConfigRow.run_id == run_id)
                    .order_by(DefenseConfigRow.round_number, DefenseConfigRow.created_at)
                )
            )
            .scalars()
            .all()
        )
        return [DefenseConfigRecord.model_validate(row) for row in rows]

    async def count(self) -> int:
        rows = await self._session.execute(select(DefenseConfigRow.id))
        return len(list(rows.scalars().all()))
