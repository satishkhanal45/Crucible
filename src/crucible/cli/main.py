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
from collections.abc import Callable, Iterator
from contextlib import AsyncExitStack, contextmanager
from decimal import Decimal
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.exc import SQLAlchemyError

from crucible.archive.classifier import ScriptedClassifierClient
from crucible.attacker.llm import ScriptedAttackerLLM
from crucible.attacker.state import AttackerMode
from crucible.cli import archive as archive_cli
from crucible.cli import reclassify as reclassify_cli
from crucible.cli import seed as seed_cli
from crucible.config import ConfigurationError, Settings, get_settings
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
from crucible.reporting.charts import coverage_grid, coverage_strip, three_curves
from crucible.reporting.data import ReportData, gather
from crucible.reporting.markdown import render_round_report, render_run_report
from crucible.reporting.redaction import present_payload
from crucible.repositories.configs import DefenseConfigRepository
from crucible.repositories.rounds import RunRepository
from crucible.services.cost_meter import BudgetExceeded
from crucible.services.retry import (
    AuthenticationFailed,
    ModelNotFound,
    ProviderError,
    RateLimited,
)
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


#: Set by `--debug`. Off, a failure prints one actionable line; on, the
#: traceback is preserved so a real defect is still debuggable.
_debug = False


@app.callback()
def cli(
    debug: Annotated[
        bool, typer.Option("--debug", help="Show full tracebacks instead of one-line errors.")
    ] = False,
) -> None:
    """Crucible — a co-evolutionary red-team loop."""
    global _debug
    _debug = debug


def _fail(message: str, hint: str = "") -> None:
    console.print(f"[red]{message}[/red]" + (f"\n[dim]{hint}[/dim]" if hint else ""))
    raise typer.Exit(code=1)


@contextmanager
def _handled() -> Iterator[None]:
    """Turn the failures a run actually meets into one line and exit 1.

    A refused database connection and an expired API key are operator errors,
    not defects, and a sixty-line rich traceback buries the one fact that says
    what to change. `--debug` puts the traceback back.
    """
    try:
        yield
    except typer.Exit:
        raise
    except ConfigurationError as error:
        if _debug:
            raise
        _fail(str(error), "Set the missing variables in .env; see .env.example.")
    except AuthenticationFailed as error:
        if _debug:
            raise
        _fail(str(error))
    except ModelNotFound as error:
        if _debug:
            raise
        _fail(str(error), "See https://console.groq.com/docs/deprecations for retirements.")
    except RateLimited as error:
        if _debug:
            raise
        _fail(
            f"rate limit not cleared after retrying: {error}",
            "Raise PROVIDER_MIN_INTERVAL_SECONDS or lower PROVIDER_MAX_CONCURRENCY in .env, "
            "then resume the run.",
        )
    except BudgetExceeded as error:
        if _debug:
            raise
        _fail(str(error), "Raise ROUND_BUDGET_USD in .env, or accept the partial results.")
    except ProviderError as error:
        if _debug:
            raise
        _fail(f"provider call failed: {error}")
    except (ConnectionError, OSError) as error:
        if _debug:
            raise
        _fail(
            f"cannot reach a service Crucible needs: {error}",
            "Is the database up? `make up`, then check DATABASE_URL in .env.",
        )
    except SQLAlchemyError as error:
        if _debug:
            raise
        _fail(
            f"database error: {type(error).__name__}: {_first_line(error)}",
            "Is the database up and migrated? `make up && make migrate`.",
        )


def _first_line(error: BaseException) -> str:
    return str(error).strip().splitlines()[0] if str(error).strip() else repr(error)


def handled[**P, R](command: Callable[P, R]) -> Callable[P, R | None]:
    """Apply `_handled` to a typer command without hiding its signature."""

    @wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        with _handled():
            return command(*args, **kwargs)
        return None  # pragma: no cover - _handled always raises or returns

    return wrapper


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
@handled
def seed() -> None:
    """Load the corpus, plant canaries, load seed attacks, verify placement."""
    raise typer.Exit(code=seed_cli.main())


@archive_app.command("reclassify")
@handled
def archive_reclassify(
    dry_run: Annotated[
        bool, typer.Option("--dry-run/--no-dry-run", help="Never writes to the archive.")
    ] = True,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Defaults to CLASSIFIER_MODEL from the environment."),
    ] = None,
) -> None:
    """Run the REAL classifier over the 40 committed seeds and report agreement.

    Makes live API calls. Coverage and the coverage grid are only as trustworthy
    as the number this prints.
    """
    settings = _settings()
    asyncio.run(reclassify_cli.reclassify(settings, console, dry_run=dry_run, model=model))


@archive_app.command("stats")
@handled
def archive_stats() -> None:
    """Coverage out of 96, novelty distribution, rejection rate, elites."""
    raise typer.Exit(code=archive_cli.main())


@loop_app.command("start")
@handled
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
        provider_max_concurrency=settings.PROVIDER_MAX_CONCURRENCY,
        provider_min_interval_seconds=settings.PROVIDER_MIN_INTERVAL_SECONDS,
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
@handled
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
@handled
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
@handled
def eval_defense(
    config_id: Annotated[str, typer.Argument(help="A DefenseConfig fingerprint.")],
    which: Annotated[EvalSet, typer.Option("--set")] = EvalSet.ARCHIVE,
) -> None:
    """Evaluate one stored config against the archive, the holdout set, or utility."""
    settings = _settings()
    asyncio.run(_eval(settings, config_id, which))


async def _eval(settings: Settings, config_id: str, which: EvalSet) -> None:
    database = Database(settings.DATABASE_URL)
    try:
        async with database.session() as session:
            config = await DefenseConfigRepository(session).get(config_id)
        if config is None:
            console.print(
                f"[red]no config {config_id}[/red]. Stored configs: "
                "`crucible report run --run-id <id>` lists the ones a run used."
            )
            raise typer.Exit(code=1)

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
@handled
def report_round(
    number: Annotated[int, typer.Argument(help="Round number.")],
    run_id: Annotated[str, typer.Option("--run-id")],
    include_payloads: Annotated[
        bool, typer.Option("--include-payloads", help="Publish raw payload text.")
    ] = False,
    out: Annotated[str | None, typer.Option("--out", help="Write to a file.")] = None,
) -> None:
    """One round: its rates and the structured config diff that produced it."""
    settings = _settings()
    asyncio.run(_report_round(settings, uuid.UUID(run_id), number, include_payloads, out))


async def _report_round(
    settings: Settings,
    run_id: uuid.UUID,
    number: int,
    include_payloads: bool,
    out: str | None,
) -> None:
    database = Database(settings.DATABASE_URL)
    try:
        data = await gather(database, run_id)
        if data is None:
            console.print(f"[red]no run {run_id}[/red]")
            raise typer.Exit(code=1)
        del include_payloads  # a round report carries no payloads
        text = render_round_report(data, number)
        _emit(text, out)
    finally:
        await database.close()


@report_app.command("run")
@handled
def report_run(
    run_id: Annotated[str, typer.Option("--run-id")],
    output: Annotated[ReportFormat, typer.Option("--format")] = ReportFormat.MD,
    include_payloads: Annotated[
        bool, typer.Option("--include-payloads", help="Publish raw payload text.")
    ] = False,
    out: Annotated[str | None, typer.Option("--out", help="Write to a file.")] = None,
    charts: Annotated[
        str | None, typer.Option("--charts", help="Directory for the generated visuals.")
    ] = None,
) -> None:
    """The whole run, as the Markdown document that goes in the repository."""
    settings = _settings()
    asyncio.run(_report_run(settings, uuid.UUID(run_id), output, include_payloads, out, charts))


async def _report_run(
    settings: Settings,
    run_id: uuid.UUID,
    output: ReportFormat,
    include_payloads: bool,
    out: str | None,
    charts: str | None,
) -> None:
    database = Database(settings.DATABASE_URL)
    try:
        data = await gather(database, run_id)
        if data is None:
            console.print(f"[red]no run {run_id}[/red]")
            raise typer.Exit(code=1)

        if output is ReportFormat.JSON:
            payload = {
                "run": data.run.model_dump(mode="json"),
                "coverage": {
                    "occupied": data.coverage.occupied,
                    "denominator": data.coverage.denominator,
                },
                "archive_size": data.archive_size,
                "holdout_size": data.holdout_size,
                "top_general": [
                    {
                        "attack_id": str(item.attack.id),
                        "cell_key": item.attack.cell_key,
                        "generality": item.generality,
                        "mechanism": present_payload(
                            item.attack.payload,
                            objective=(
                                item.attack.objective.value if item.attack.objective else None
                            ),
                            vector=item.attack.vector.value,
                            technique=(
                                item.attack.technique.value if item.attack.technique else None
                            ),
                            include_payloads=include_payloads,
                        ),
                    }
                    for item in data.top_general
                ],
            }
            _emit(json.dumps(payload, indent=2, sort_keys=True), out)
        else:
            _emit(render_run_report(data, include_payloads=include_payloads), out)

        if charts:
            written = _write_charts(data, Path(charts))
            for path in written:
                console.print(f"[green]wrote[/green] {path}")
    finally:
        await database.close()


def _write_charts(data: ReportData, directory: Path) -> list[Path]:
    """The three curves, the coverage grid, and the per-round strip."""
    fitness = {
        cell.cell_key: float(cell.elite_fitness)
        for cell in data.cells
        if cell.elite_fitness is not None
    }
    occupancy: dict[str, int] = {}
    for attack in data.attacks:
        if attack.cell_key:
            occupancy[attack.cell_key] = occupancy.get(attack.cell_key, 0) + 1

    per_round: list[tuple[int, dict[str, int]]] = []
    for report in data.rounds:
        upto: dict[str, int] = {}
        for attack in data.attacks:
            if attack.cell_key and attack.round_generated <= report.round_number:
                upto[attack.cell_key] = upto.get(attack.cell_key, 0) + 1
        per_round.append((report.round_number, upto))

    return [
        three_curves(data.rounds, directory / "three_curves.png"),
        coverage_grid(fitness, occupancy, directory / "coverage_grid.png", title="Final coverage"),
        coverage_strip(per_round, directory / "coverage_evolution.png"),
    ]


def _emit(text: str, out: str | None) -> None:
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        console.print(f"[green]wrote[/green] {path}")
        return
    print(text)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
