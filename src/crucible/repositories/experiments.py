"""The only module that reads or writes `experiment_results`."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from crucible.db.models import ExperimentResultRow
from crucible.schemas.experiments import ExperimentResult


class ExperimentResultRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self, experiment: str, result: ExperimentResult, *, run_id: uuid.UUID | None = None
    ) -> uuid.UUID:
        """Store one experiment's output. History is kept; findings read the latest."""
        row = ExperimentResultRow(
            id=uuid.uuid4(),
            experiment=experiment,
            kind=result.kind,
            payload=result.model_dump(mode="json"),
            run_id=run_id,
            # Set here rather than from the column default, which is the
            # transaction clock: two results written together must still order.
            created_at=datetime.now(UTC),
        )
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def latest(self, experiment: str) -> dict[str, Any] | None:
        """The most recent stored payload for one experiment."""
        row = (
            await self._session.execute(
                select(ExperimentResultRow)
                .where(ExperimentResultRow.experiment == experiment)
                .order_by(ExperimentResultRow.created_at.desc(), ExperimentResultRow.id)
                .limit(1)
            )
        ).scalar_one_or_none()
        return dict(row.payload) if row is not None else None

    async def all_latest(self) -> dict[str, dict[str, Any]]:
        """The most recent payload for every experiment that has ever run."""
        rows = (
            await self._session.execute(
                select(ExperimentResultRow).order_by(ExperimentResultRow.created_at.asc())
            )
        ).scalars()
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            latest[row.experiment] = dict(row.payload)
        return latest
