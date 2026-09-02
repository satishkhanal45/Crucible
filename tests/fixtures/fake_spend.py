"""An in-memory `SpendRepository` so cost-meter unit tests need no database."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from crucible.schemas.spend import NewSpend, SpendRecord


class FakeSpendRepository:
    """Implements `SpendRepositoryProtocol` against a list."""

    def __init__(self) -> None:
        self.records: list[SpendRecord] = []

    async def add(self, spend: NewSpend) -> SpendRecord:
        record = SpendRecord(
            id=uuid.uuid4(),
            round_id=spend.round_id,
            provider=spend.provider,
            model=spend.model,
            prompt_tokens=spend.prompt_tokens,
            completion_tokens=spend.completion_tokens,
            estimated_cost_usd=spend.estimated_cost_usd,
            created_at=datetime.now(UTC),
        )
        self.records.append(record)
        return record

    async def total_for_round(self, round_id: uuid.UUID | None) -> Decimal:
        # NULL costs are unknown, not zero-cost; SUM ignores them, and so do we.
        return sum(
            (
                record.estimated_cost_usd
                for record in self.records
                if record.round_id == round_id and record.estimated_cost_usd is not None
            ),
            Decimal(0),
        )
