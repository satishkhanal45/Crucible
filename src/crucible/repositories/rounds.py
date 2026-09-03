"""The only module that reads or writes `runs` and `rounds`."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from crucible.db.models import RoundRow, RunRow
from crucible.loop.reports import HaltReason, Regression, RoundReport, RunStatus
from crucible.loop.statistics import Proportion


def _decimal(value: float, places: str = "0.00001") -> Decimal:
    return Decimal(str(value)).quantize(Decimal(places))


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        run_id: uuid.UUID,
        attacker_mode: str,
        rounds_planned: int,
        starting_config_id: str,
        budget_usd: Decimal,
        seed: int,
        settings: dict[str, Any],
    ) -> RunRow:
        row = RunRow(
            id=run_id,
            status=RunStatus.RUNNING.value,
            attacker_mode=attacker_mode,
            rounds_planned=rounds_planned,
            starting_config_id=starting_config_id,
            current_config_id=starting_config_id,
            budget_usd=budget_usd,
            seed=seed,
            settings=settings,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get(self, run_id: uuid.UUID) -> RunRow | None:
        return await self._session.get(RunRow, run_id)

    async def list_recent(self, limit: int = 20) -> list[RunRow]:
        rows = (
            (
                await self._session.execute(
                    select(RunRow).order_by(RunRow.started_at.desc()).limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return list(rows)

    async def finish(
        self,
        run_id: uuid.UUID,
        *,
        status: RunStatus,
        halt_reason: HaltReason | None,
        rounds_completed: int,
        current_config_id: str,
        ended_at: Any = None,
    ) -> None:
        row = await self._session.get(RunRow, run_id)
        if row is None:
            return
        row.status = status.value
        row.halt_reason = halt_reason.value if halt_reason else None
        row.rounds_completed = rounds_completed
        row.current_config_id = current_config_id
        row.ended_at = ended_at


class RoundRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(self, report: RoundReport) -> RoundRow:
        """Persist one round. Idempotent on `(run_id, round_number)`."""
        report.validate_intervals()
        archive_low, archive_high = report.archive_block.interval
        holdout_low, holdout_high = report.holdout_block.interval

        values: dict[str, Any] = {
            "id": uuid.uuid4(),
            "run_id": report.run_id,
            "round_number": report.round_number,
            "attacker_mode": report.attacker_mode,
            "defense_before": report.defense_before,
            "defense_after": report.defense_after,
            "attacks_generated": report.attacks_generated,
            "attacks_rejected_novelty": report.attacks_rejected_novelty,
            "breaches_found": report.breaches_found,
            "archive_successes": report.archive_block.successes,
            "archive_trials": report.archive_block.trials,
            "archive_block_rate": _decimal(report.archive_block.rate),
            "archive_block_rate_ci_low": _decimal(archive_low),
            "archive_block_rate_ci_high": _decimal(archive_high),
            "holdout_successes": report.holdout_block.successes,
            "holdout_trials": report.holdout_block.trials,
            "holdout_block_rate": _decimal(report.holdout_block.rate),
            "holdout_ci_low": _decimal(holdout_low),
            "holdout_ci_high": _decimal(holdout_high),
            "overfit_gap": _decimal(report.overfit_gap),
            "utility_successes": report.utility_pass.successes,
            "utility_trials": report.utility_pass.trials,
            "utility_pass_rate": _decimal(report.utility_pass.rate),
            "mean_novelty": _decimal(report.mean_novelty),
            "cells_occupied": report.cells_occupied,
            "new_cells": report.new_cells,
            "regressions": [
                regression.model_dump(mode="json") for regression in report.regressions
            ],
            "config_promoted": report.config_promoted,
            "cost_usd": report.cost_usd,
            "halt_reason": report.halt_reason.value if report.halt_reason else None,
            "started_at": report.started_at,
            "ended_at": report.ended_at,
        }
        statement = insert(RoundRow).values(**values)
        upsert = statement.on_conflict_do_update(
            constraint="uq_rounds_run_round",
            set_={key: statement.excluded[key] for key in values if key != "id"},
        ).returning(RoundRow)
        return (await self._session.execute(upsert)).scalar_one()

    async def list_for_run(self, run_id: uuid.UUID) -> list[RoundReport]:
        rows = (
            (
                await self._session.execute(
                    select(RoundRow)
                    .where(RoundRow.run_id == run_id)
                    .order_by(RoundRow.round_number)
                )
            )
            .scalars()
            .all()
        )
        return [_to_report(row) for row in rows]

    async def get(self, run_id: uuid.UUID, round_number: int) -> RoundReport | None:
        row = await self._session.scalar(
            select(RoundRow)
            .where(RoundRow.run_id == run_id)
            .where(RoundRow.round_number == round_number)
        )
        return None if row is None else _to_report(row)

    async def latest(self, run_id: uuid.UUID) -> RoundReport | None:
        row = await self._session.scalar(
            select(RoundRow)
            .where(RoundRow.run_id == run_id)
            .order_by(RoundRow.round_number.desc())
            .limit(1)
        )
        return None if row is None else _to_report(row)


def _to_report(row: RoundRow) -> RoundReport:
    return RoundReport(
        run_id=row.run_id,
        round_number=row.round_number,
        attacker_mode=row.attacker_mode,
        defense_before=row.defense_before,
        defense_after=row.defense_after,
        attacks_generated=row.attacks_generated,
        attacks_rejected_novelty=row.attacks_rejected_novelty,
        breaches_found=row.breaches_found,
        archive_block=Proportion(successes=row.archive_successes, trials=row.archive_trials),
        holdout_block=Proportion(successes=row.holdout_successes, trials=row.holdout_trials),
        utility_pass=Proportion(successes=row.utility_successes, trials=row.utility_trials),
        mean_novelty=float(row.mean_novelty),
        cells_occupied=row.cells_occupied,
        new_cells=row.new_cells,
        regressions=tuple(Regression.model_validate(item) for item in (row.regressions or [])),
        config_promoted=row.config_promoted,
        cost_usd=row.cost_usd,
        halt_reason=HaltReason(row.halt_reason) if row.halt_reason else None,
        started_at=row.started_at,
        ended_at=row.ended_at,
    )
