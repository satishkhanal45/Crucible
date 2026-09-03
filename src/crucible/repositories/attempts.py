"""The only module that reads or writes `attempts`.

The outcome cache lives here: `find_cached()` is the read side of the unique
`(attack_id, defense_config_id)` pair, and `add()` is the write side.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from crucible.db.models import Attempt
from crucible.schemas.attempt import AttemptRecord, NewAttempt
from crucible.schemas.outcome import Outcome


class AttemptRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, attempt: NewAttempt) -> AttemptRecord:
        """Insert an attempt, replacing any earlier row for the same pair.

        The pair is unique, so a re-execution (a `force=True` bypass, or a
        replay that finds a different outcome) updates in place rather than
        accumulating rows the archive would later double-count.
        """
        values = {
            "id": uuid.uuid4(),
            "attack_id": attempt.attack_id,
            "defense_config_id": attempt.defense_config_id,
            "round_id": attempt.round_id,
            "vector": attempt.vector.value,
            "outcome": attempt.outcome.value,
            "tier": int(attempt.tier),
            "canaries_hit": list(attempt.canaries_hit),
            "unauthorized_tools": list(attempt.unauthorized_tools),
            "blocked_tools": list(attempt.blocked_tools),
            "judge_score": attempt.judge_score,
            "judge_rationale": attempt.judge_rationale,
            "response_text": attempt.response_text,
            "trace": attempt.trace,
            "latency_ms": attempt.latency_ms,
            "cost_usd": attempt.cost_usd,
        }
        statement = insert(Attempt).values(**values)
        upsert = statement.on_conflict_do_update(
            constraint="uq_attempts_attack_defense",
            set_={key: statement.excluded[key] for key in values if key != "id"},
        ).returning(Attempt)
        row = (await self._session.execute(upsert)).scalar_one()
        return AttemptRecord.model_validate(row)

    async def find_cached(
        self, attack_id: uuid.UUID, defense_config_id: str
    ) -> AttemptRecord | None:
        """The stored outcome for a pair, if it has already been executed."""
        row = await self._session.scalar(
            select(Attempt)
            .where(Attempt.attack_id == attack_id)
            .where(Attempt.defense_config_id == defense_config_id)
        )
        return None if row is None else AttemptRecord.model_validate(row)

    async def get(self, attempt_id: uuid.UUID) -> AttemptRecord | None:
        row = await self._session.get(Attempt, attempt_id)
        return None if row is None else AttemptRecord.model_validate(row)

    async def outcomes_for_config(self, defense_config_id: str) -> list[Outcome]:
        rows = await self._session.execute(
            select(Attempt.outcome).where(Attempt.defense_config_id == defense_config_id)
        )
        return [Outcome(value) for value in rows.scalars().all()]

    async def count_hijacks_blocked(self, defense_config_id: str) -> int:
        """Attempts where a privileged call was emitted and Layer 5 stopped it.

        The Phase 8 layer-ablation table reports this next to the block rate:
        a config that blocks hijacks is doing something a config that never
        provoked one is not.
        """
        rows = await self._session.execute(
            select(Attempt.id)
            .where(Attempt.defense_config_id == defense_config_id)
            .where(func.cardinality(Attempt.blocked_tools) > 0)
        )
        return len(list(rows.scalars().all()))

    async def list_for_round(self, round_id: uuid.UUID) -> list[AttemptRecord]:
        rows = (
            (
                await self._session.execute(
                    select(Attempt).where(Attempt.round_id == round_id).order_by(Attempt.created_at)
                )
            )
            .scalars()
            .all()
        )
        return [AttemptRecord.model_validate(row) for row in rows]

    async def count(self) -> int:
        rows = await self._session.execute(select(Attempt.id))
        return len(list(rows.scalars().all()))

    async def add_many(self, attempts: Sequence[NewAttempt]) -> list[AttemptRecord]:
        return [await self.add(attempt) for attempt in attempts]
