"""The only module that reads or writes `classifier_agreement`.

Every report that quotes coverage reads the latest row through here, so no
figure in `docs/findings.md` has to be typed by hand — which is the standing
rule for every number in that document.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from crucible.db.models import ClassifierAgreementRow
from crucible.schemas.agreement import ClassifierAgreement


class ClassifierAgreementRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(self, agreement: ClassifierAgreement) -> uuid.UUID:
        """Store one measurement. History is kept: models and prompts change."""
        row = ClassifierAgreementRow(
            id=uuid.uuid4(),
            # Set here rather than left to the column's `now()` default, which
            # is the *transaction's* clock: two measurements written in one
            # transaction would share a timestamp and `latest()` could not order
            # them.
            created_at=datetime.now(UTC),
            model=agreement.model_name,
            total=agreement.total,
            objective_agreed=agreement.objective_agreed,
            technique_agreed=agreement.technique_agreed,
            combined_agreed=agreement.combined_agreed,
            unclassified=agreement.unclassified,
        )
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def latest(self) -> ClassifierAgreement | None:
        """The most recent measurement, or None when none has ever been made."""
        row = (
            await self._session.execute(
                select(ClassifierAgreementRow)
                .order_by(ClassifierAgreementRow.created_at.desc(), ClassifierAgreementRow.id)
                .limit(1)
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return ClassifierAgreement(
            model_name=row.model,
            total=row.total,
            objective_agreed=row.objective_agreed,
            technique_agreed=row.technique_agreed,
            combined_agreed=row.combined_agreed,
            unclassified=row.unclassified,
        )

    async def count(self) -> int:
        rows = await self._session.execute(select(ClassifierAgreementRow.id))
        return len(rows.scalars().all())
