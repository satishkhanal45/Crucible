"""Gathering what a report needs, in one pass over the database."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict

from crucible.archive.grid import Coverage, coverage
from crucible.db.session import Database
from crucible.defenses.config import DefenseConfig
from crucible.loop.reports import RoundReport, RunReport
from crucible.repositories.attacks import AttackRepository
from crucible.repositories.cells import CellRepository
from crucible.repositories.configs import DefenseConfigRepository
from crucible.repositories.rounds import RoundRepository, RunRepository
from crucible.schemas.archive import ArchivedAttack, CellRecord


class GeneralAttack(BaseModel):
    """An archived attack ranked by how many past configs it still breaches."""

    model_config = ConfigDict(frozen=True)

    attack: ArchivedAttack
    breached_configs: int
    total_configs: int
    #: Counted from the attempts table, which includes full-archive
    #: re-evaluation; the attack row's own counters do not.
    attempts: int = 0
    breaches: int = 0

    @property
    def generality(self) -> float:
        return self.breached_configs / self.total_configs if self.total_configs else 0.0


class ReportData(BaseModel):
    """Everything a run report is rendered from."""

    model_config = ConfigDict(frozen=True)

    run: RunReport
    configs: dict[str, DefenseConfig] = {}
    attacks: tuple[ArchivedAttack, ...] = ()
    cells: tuple[CellRecord, ...] = ()
    top_general: tuple[GeneralAttack, ...] = ()
    archive_size: int = 0
    holdout_size: int = 0
    unclassified: int = 0

    @property
    def rounds(self) -> tuple[RoundReport, ...]:
        return self.run.rounds

    @property
    def coverage(self) -> Coverage:
        return coverage(attack.cell_key for attack in self.attacks)

    @property
    def by_id(self) -> dict[uuid.UUID, ArchivedAttack]:
        return {attack.id: attack for attack in self.attacks}


async def gather(database: Database, run_id: uuid.UUID, *, top: int = 10) -> ReportData | None:
    """Read one run and the archive it produced."""
    async with database.session() as session:
        run_row = await RunRepository(session).get(run_id)
        if run_row is None:
            return None
        rounds = await RoundRepository(session).list_for_run(run_id)

        attacks_repo = AttackRepository(session)
        attacks = await attacks_repo.list_all()
        holdout = await attacks_repo.count_holdout()
        unclassified = await attacks_repo.count_unclassified()
        all_configs = await attacks_repo.all_config_ids()
        totals = await attacks_repo.attempt_totals()

        ranked: list[GeneralAttack] = []
        for attack in attacks:
            breached = await attacks_repo.breached_config_ids(attack.id)
            if not breached:
                continue
            attempts, breaches = totals.get(attack.id, (0, 0))
            ranked.append(
                GeneralAttack(
                    attack=attack,
                    breached_configs=len(breached & all_configs),
                    total_configs=len(all_configs),
                    attempts=attempts,
                    breaches=breaches,
                )
            )

        cells = await CellRepository(session).occupied()
        stored = await DefenseConfigRepository(session).list_for_run(run_id)
        configs = {record.id: record.config for record in stored}
        for round_report in rounds:
            for fingerprint in (round_report.defense_before, round_report.defense_after):
                if fingerprint not in configs:
                    found = await DefenseConfigRepository(session).get(fingerprint)
                    if found is not None:
                        configs[fingerprint] = found

    ranked.sort(key=lambda item: (-item.generality, -item.breaches, str(item.attack.id)))
    from crucible.loop.reports import HaltReason, RunStatus

    run = RunReport(
        run_id=run_id,
        status=RunStatus(run_row.status),
        attacker_mode=run_row.attacker_mode,
        starting_config_id=run_row.starting_config_id,
        final_config_id=run_row.current_config_id,
        rounds=tuple(rounds),
        halt_reason=HaltReason(run_row.halt_reason) if run_row.halt_reason else None,
    )
    return ReportData(
        run=run,
        configs=configs,
        attacks=tuple(attacks),
        cells=tuple(cells),
        top_general=tuple(ranked[:top]),
        archive_size=len(attacks),
        holdout_size=holdout,
        unclassified=unclassified,
    )
