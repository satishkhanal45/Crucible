"""`crucible` — the command line.

    crucible seed
    crucible loop start --rounds 8 --mode black_box --budget 5.00
    crucible loop resume --run-id <id>
    crucible loop status
    crucible eval defense <config_id> --set archive|holdout|utility
    crucible report round <n> --run-id <id>
    crucible report run --format md|json
    crucible archive stats

Reporting here is deliberately minimal text: Phase 7 does the real thing.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import AsyncExitStack
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from crucible.archive.classifier import ScriptedClassifierClient
from crucible.attacker.llm import ScriptedAttackerLLM
from crucible.attacker.state import AttackerMode
from crucible.cli import archive as archive_cli
from crucible.cli import seed as seed_cli
from crucible.config import Settings, get_settings
from crucible.db.session import Database
from crucible.defender.llm import ScriptedDefenderLLM
from crucible.defenses.config import DefenseConfig
from crucible.logging import configure_logging
from crucible.loop.reports import RunReport, RunStatus
from crucible.loop.runner import (
    LoopFactories,
    LoopRunner,
    LoopSettings,
    build_components,
    default_embedder,
    load_run_report,
    postgres_checkpointer,
)
from crucible.repositories.rounds import RoundRepository, RunRepository
from crucible.target.reference.llm import ScriptedTargetLLM

console = Console()
app = typer.Typer(add_completion=False, help="Crucible — a co-evolutionary red-team loop.")
loop_app = typer.Typer(help="Run and resume the co-evolution loop.")
report_app = typer.Typer(help="Read what a run recorded.")
eval_app = typer.Typer(help="Evaluate one defense config.")
archive_app = typer.Typer(help="Inspect the archive.")
app.add_typer(loop_app, name="loop")
app.add_typer(report_app, name="report")
app.add_typer(eval_app, name="eval")
app.add_typer(archive_app, name="archive")


class EvalSet(StrEnum):
    ARCHIVE = "archive"
    HOLDOUT = "holdout"
    UTILITY = "utility"


class ReportFormat(StrEnum):
    MD = "md"
    JSON = "json"


def _settings() -> Settings:
    settings = get_settings()
    configure_logging(settings.log_level_number)
    return settings


def _scripted_factories() -> LoopFactories:
    """Offline clients.

    TODO(phase-8): wire the Groq clients here behind `--provider groq`. The loop
    itself is provider-agnostic; only these four factories change.
    """
    return LoopFactories(
        target_llm=ScriptedTargetLLM,
        attacker_llm=ScriptedAttackerLLM,
        defender_llm=ScriptedDefenderLLM,
        classifier_client=ScriptedClassifierClient,
    )


@app.command()
def seed() -> None:
    """Load the corpus, plant canaries, load seed attacks, verify placement."""
    raise typer.Exit(code=seed_cli.main())


@archive_app.command("stats")
def archive_stats() -> None:
    """Coverage out of 96, novelty distribution, rejection rate, elites."""
    raise typer.Exit(code=archive_cli.main())


@loop_app.command("start")
def loop_start(
    rounds: Annotated[int, typer.Option(help="How many rounds to run.")] = 8,
    mode: Annotated[str, typer.Option(help="black_box or white_box.")] = "black_box",
    budget: Annotated[float, typer.Option(help="Per-round spend cap in USD.")] = 5.00,
    seed_value: Annotated[int, typer.Option("--seed", help="Run seed.")] = 20260906,
) -> None:
    """Start a run. D(0) is always the empty config, so the loop starts weak."""
    settings = _settings()
    loop_settings = LoopSettings(
        rounds=rounds,
        mode=AttackerMode(mode),
        budget_usd=Decimal(str(budget)),
        seed=seed_value,
    )
    report = asyncio.run(_start(settings, loop_settings))
    _print_run(report)


async def _start(settings: Settings, loop_settings: LoopSettings) -> RunReport:
    database = Database(settings.DATABASE_URL)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(database.close)
        checkpointer = await postgres_checkpointer(stack, settings)
        components = await build_components(
            database,
            settings=loop_settings,
            factories=_scripted_factories(),
            embedder=default_embedder(settings),
            allowlist=tuple(settings.TARGET_ALLOWLIST),
        )
        runner = LoopRunner(database, components, settings=loop_settings, checkpointer=checkpointer)
        return await runner.start(starting_config=DefenseConfig.empty())
    raise RuntimeError("unreachable")


@loop_app.command("resume")
def loop_resume(
    run_id: Annotated[str, typer.Option("--run-id", help="The run to continue.")],
) -> None:
    """Continue a checkpointed run from where it stopped."""
    settings = _settings()
    report = asyncio.run(_resume(settings, uuid.UUID(run_id)))
    _print_run(report)


async def _resume(settings: Settings, run_id: uuid.UUID) -> RunReport:
    database = Database(settings.DATABASE_URL)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(database.close)
        async with database.session() as session:
            row = await RunRepository(session).get(run_id)
        if row is None:
            console.print(f"[red]no run {run_id}[/red]")
            raise typer.Exit(code=1)

        loop_settings = LoopSettings.model_validate(row.settings)
        checkpointer = await postgres_checkpointer(stack, settings)
        components = await build_components(
            database,
            settings=loop_settings,
            factories=_scripted_factories(),
            embedder=default_embedder(settings),
            allowlist=tuple(settings.TARGET_ALLOWLIST),
        )
        runner = LoopRunner(database, components, settings=loop_settings, checkpointer=checkpointer)
        return await runner.resume(run_id)
    raise RuntimeError("unreachable")


@loop_app.command("status")
def loop_status(
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
) -> None:
    """Show recent runs, or one run's rounds."""
    settings = _settings()
    asyncio.run(_status(settings, uuid.UUID(run_id) if run_id else None))


async def _status(settings: Settings, run_id: uuid.UUID | None) -> None:
    database = Database(settings.DATABASE_URL)
    try:
        if run_id is None:
            async with database.session() as session:
                runs = await RunRepository(session).list_recent()
            table = Table(title="runs")
            for column in ("run id", "status", "mode", "rounds", "halt reason"):
                table.add_column(column)
            for row in runs:
                table.add_row(
                    str(row.id),
                    row.status,
                    row.attacker_mode,
                    f"{row.rounds_completed}/{row.rounds_planned}",
                    row.halt_reason or "-",
                )
            console.print(table)
            return

        report = await load_run_report(database, run_id)
        if report is None:
            console.print(f"[red]no run {run_id}[/red]")
            return
        _print_run(report)
    finally:
        await database.close()


def _print_run(report: RunReport) -> None:
    colour = {
        RunStatus.COMPLETED: "green",
        RunStatus.HALTED: "yellow",
        RunStatus.FAILED: "red",
        RunStatus.RUNNING: "cyan",
    }[report.status]
    console.print(
        f"[{colour}]run {report.run_id} — {report.status.value}"
        + (f" ({report.halt_reason.value})" if report.halt_reason else "")
        + f"[/{colour}]"
    )
    console.print(
        f"  D(0) {report.starting_config_id[:12]} -> D(n) {report.final_config_id[:12]}, "
        f"{report.rounds_completed} rounds, {report.total_regressions} regressions"
    )
    table = Table(show_header=True)
    for column in (
        "round",
        "archive block",
        "holdout block",
        "overfit gap",
        "utility",
        "cells",
        "regressions",
    ):
        table.add_column(column)
    for round_report in report.rounds:
        table.add_row(
            str(round_report.round_number),
            str(round_report.archive_block),
            str(round_report.holdout_block),
            f"{round_report.overfit_gap:+.3f}",
            str(round_report.utility_pass),
            f"{round_report.cells_occupied}/96 (+{round_report.new_cells})",
            str(len(round_report.regressions)) if round_report.regressions else "-",
        )
    console.print(table)


@eval_app.command("defense")
def eval_defense(
    config_id: Annotated[str, typer.Argument(help="A DefenseConfig fingerprint.")],
    which: Annotated[EvalSet, typer.Option("--set")] = EvalSet.ARCHIVE,
) -> None:
    """Evaluate one config against the archive, the holdout set, or utility.

    TODO(phase-7): resolve a fingerprint to its stored config. Until the
    `defense_configs` table exists, only the empty config can be named.
    """
    settings = _settings()
    if config_id != DefenseConfig.empty().fingerprint():
        console.print(
            f"[yellow]only the empty config ({DefenseConfig.empty().fingerprint()}) can be "
            "resolved until Phase 7 stores configs by id[/yellow]"
        )
        raise typer.Exit(code=1)
    asyncio.run(_eval(settings, DefenseConfig.empty(), which))


async def _eval(settings: Settings, config: DefenseConfig, which: EvalSet) -> None:
    database = Database(settings.DATABASE_URL)
    try:
        loop_settings = LoopSettings()
        components = await build_components(
            database,
            settings=loop_settings,
            factories=_scripted_factories(),
            embedder=default_embedder(settings),
            allowlist=tuple(settings.TARGET_ALLOWLIST),
        )
        service = components.evaluation
        if which is EvalSet.UTILITY:
            utility = await service.evaluate_utility(config)
            console.print(
                f"utility {utility.passed}/{utility.total} ({utility.pass_rate:.1%}), "
                f"hard negatives {utility.hard_negative_passed}/{utility.hard_negative_total}"
            )
            return
        if which is EvalSet.HOLDOUT:
            holdout = await service.evaluate_holdout(config)
            console.print(
                f"holdout block rate {holdout.block_rate:.1%} over {holdout.evaluated} attacks"
            )
            return
        full = await service.evaluate_full(config, include_holdout=True)
        console.print(
            f"archive block rate {full.archive_block_rate:.1%} over "
            f"{full.archive.evaluated} attacks; holdout {full.holdout_block_rate}"
        )
    finally:
        await database.close()


@report_app.command("round")
def report_round(
    number: Annotated[int, typer.Argument(help="Round number.")],
    run_id: Annotated[str, typer.Option("--run-id")],
) -> None:
    """Print one round's report. Phase 7 does the real reporting."""
    settings = _settings()
    asyncio.run(_report_round(settings, uuid.UUID(run_id), number))


async def _report_round(settings: Settings, run_id: uuid.UUID, number: int) -> None:
    database = Database(settings.DATABASE_URL)
    try:
        async with database.session() as session:
            report = await RoundRepository(session).get(run_id, number)
        if report is None:
            console.print(f"[red]no round {number} for run {run_id}[/red]")
            return
        console.print(report.summary())
    finally:
        await database.close()


@report_app.command("run")
def report_run(
    run_id: Annotated[str, typer.Option("--run-id")],
    output: Annotated[ReportFormat, typer.Option("--format")] = ReportFormat.MD,
) -> None:
    """Print a whole run. Markdown or JSON; Phase 7 does the real reporting."""
    settings = _settings()
    asyncio.run(_report_run(settings, uuid.UUID(run_id), output))


async def _report_run(settings: Settings, run_id: uuid.UUID, output: ReportFormat) -> None:
    database = Database(settings.DATABASE_URL)
    try:
        report = await load_run_report(database, run_id)
        if report is None:
            console.print(f"[red]no run {run_id}[/red]")
            return
        if output is ReportFormat.JSON:
            console.print_json(json.dumps(report.model_dump(mode="json")))
            return
        console.print(f"# Run {report.run_id}\n")
        console.print(f"- status: {report.status.value}")
        console.print(f"- mode: {report.attacker_mode}")
        console.print(f"- rounds: {report.rounds_completed}")
        console.print(f"- regressions: {report.total_regressions}")
        if report.halt_reason:
            console.print(f"- halted: {report.halt_reason.value}")
        for round_report in report.rounds:
            console.print("")
            console.print(round_report.summary())
    finally:
        await database.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
