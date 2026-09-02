"""The only module that reads or writes the `spend` table."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from crucible.db.models import Spend
from crucible.schemas.spend import NewSpend, SpendRecord


class SpendRepositoryProtocol(Protocol):
    """The surface `CostMeter` depends on, so tests can supply a fake."""

    async def add(self, spend: NewSpend) -> SpendRecord: ...

    async def total_for_round(self, round_id: uuid.UUID | None) -> Decimal: ...


class SpendRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, spend: NewSpend) -> SpendRecord:
        row = Spend(
            round_id=spend.round_id,
            provider=spend.provider,
            model=spend.model,
            prompt_tokens=spend.prompt_tokens,
            completion_tokens=spend.completion_tokens,
            estimated_cost_usd=spend.estimated_cost_usd,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return SpendRecord.model_validate(row)

    async def total_for_round(self, round_id: uuid.UUID | None) -> Decimal:
        """Sum of known costs for a round. Rows with a NULL cost are ignored."""
        criterion = Spend.round_id.is_(None) if round_id is None else Spend.round_id == round_id
        statement = select(func.coalesce(func.sum(Spend.estimated_cost_usd), 0)).where(criterion)
        total = await self._session.scalar(statement)
        return Decimal(total if total is not None else 0)
