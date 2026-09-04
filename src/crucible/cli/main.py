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

from crucible.attacker.state import AttackerMode
from crucible.cli import archive as archive_cli
from crucible.cli import reclassify as reclassify_cli
from crucible.cli import seed as seed_cli
from crucible.cli.providers import Provider, attacker_on, build_factories
from crucible.config import ConfigurationError, Settings, get_settings
from crucible.db.session import Database
from crucible.defenses.config import DefenseConfig
from crucible.experiments.config import (
    ExperimentConfig,
    ExperimentKind,
    experiment_names,
    load_experiment,
    smoke_experiment,
)
from crucible.experiments.runner import (
    ExperimentContext,
    ProviderMismatch,
    cost_estimate_minutes,
    pacing_settings,
    record_result,
    run_layer_ablation,
    run_loop_experiment,
    run_model_overlap,
    run_transfer,
)
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
from crucible.reporting import findings as findings_report
from crucible.reporting.charts import coverage_grid, coverage_strip, three_curves
from crucible.reporting.data import ReportData, gather
from crucible.reporting.findings import StubbedRunRefused
from crucible.reporting.markdown import render_round_report, render_run_report
from crucible.reporting.redaction import present_payload
from crucible.repositories.configs import DefenseConfigRepository
from crucible.repositories.rounds import RunRepository
from crucible.schemas.experiments import (
    LayerAblationResult,
    LoopExperimentResult,
)
from crucible.services.cost_meter import BudgetExceeded
from crucible.services.retry import (
    AuthenticationFailed,
    ModelNotFound,
    ProviderError,
    RateLimited,
)

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
experiment_app = typer.Typer(help="Run the committed Phase 8 experiments.")
findings_app = typer.Typer(help="Regenerate the numbers in docs/findings.md.")
app.add_typer(experiment_app, name="experiment")
app.add_typer(findings_app, name="findings")


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
    except StubbedRunRefused as error:
        if _debug:
            raise
        _fail(
            str(error),
            "Run `crucible experiment run main --provider groq`, "
            "or `--smoke` first to check the providers answer.",
        )
    except ProviderMismatch as error:
        if _debug:
            raise
        _fail(str(error))
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


def _announce_provider(settings: Settings, provider: Provider) -> None:
    """Say which models will run, before spending hours or quota on them."""
    if provider is Provider.SCRIPTED:
        console.print(
            "[yellow]--provider scripted: this run uses deterministic test doubles. "
            "It will be recorded as stubbed=true, every report will carry a STUBBED RUN "
            "banner, and `crucible findings regenerate` will refuse its numbers.[/yellow]"
        )
        return
    table = Table(title="live providers, per agent")
    table.add_column("agent")
    table.add_column("provider")
    table.add_column("model")
    table.add_column("tokens/min")
    for role, agent_provider, model in settings.agents:
        tokens_per_minute, _ = settings.rate_limits_for(agent_provider)
        table.add_row(role, agent_provider.value, model, str(tokens_per_minute))
    console.print(table)
    pools = {agent_provider for _, agent_provider, _ in settings.agents}
    if len(pools) > 1:
        console.print(
            f"[cyan]{len(pools)} providers in this run: each has its own rate-limit pool "
            f"and its own recorded provenance.[/cyan]"
        )
    unpriced = settings.unpriced_models()
    if unpriced:
        console.print(
            f"[yellow]no price for {', '.join(unpriced)}: spend will record as NULL and "
            f"ROUND_BUDGET_USD cannot be enforced for it[/yellow]"
        )


def _factories(settings: Settings, provider: Provider, stack: AsyncExitStack) -> LoopFactories:
    """The model clients for one run. Live unless `--provider scripted`.

    Scripted clients are reachable only through that flag, and any run using
    them is recorded as `stubbed=true`, bannered in every report, and refused by
    `crucible findings regenerate`.
    """
    return build_factories(settings, provider, stack)


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
    provider: Annotated[
        Provider,
        typer.Option(
            "--provider",
            help=(
                "groq (live: each agent on the provider its settings name) or "
                "scripted (test doubles, marked stubbed)."
            ),
        ),
    ] = Provider.GROQ,
) -> None:
    """Start a run. D(0) is always the empty config, so the loop starts weak."""
    settings = _settings()
    _announce_provider(settings, provider)
    loop_settings = LoopSettings(
        rounds=rounds,
        mode=AttackerMode(mode),
        budget_usd=Decimal(str(budget)),
        seed=seed_value,
        **pacing_settings(settings),
    )
    report = asyncio.run(_start(settings, loop_settings, provider))
    _print_run(report)


async def _start(settings: Settings, loop_settings: LoopSettings, provider: Provider) -> RunReport:
    database = Database(settings.DATABASE_URL)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(database.close)
        checkpointer = await postgres_checkpointer(stack, settings)
        components = await build_components(
            database,
            settings=loop_settings,
            factories=_factories(settings, provider, stack),
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
    provider: Annotated[
        Provider,
        typer.Option(
            "--provider",
            help=(
                "groq (live: each agent on the provider its settings name) or "
                "scripted (test doubles, marked stubbed)."
            ),
        ),
    ] = Provider.GROQ,
) -> None:
    """Continue a checkpointed run from where it stopped."""
    settings = _settings()
    _announce_provider(settings, provider)
    report = asyncio.run(_resume(settings, uuid.UUID(run_id), provider))
    _print_run(report)


async def _resume(settings: Settings, run_id: uuid.UUID, provider: Provider) -> RunReport:
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
            factories=_factories(settings, provider, stack),
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
    banner = report.banner
    if banner is not None:
        console.print(f"[red]{banner}[/red]")
    else:
        console.print("[green]live run[/green]: " + ", ".join(report.provenance.render_lines()))
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
    provider: Annotated[
        Provider,
        typer.Option(
            "--provider",
            help=(
                "groq (live: each agent on the provider its settings name) or "
                "scripted (test doubles)."
            ),
        ),
    ] = Provider.GROQ,
) -> None:
    """Evaluate one stored config against the archive, the holdout set, or utility."""
    settings = _settings()
    _announce_provider(settings, provider)
    asyncio.run(_eval(settings, config_id, which, provider))


async def _eval(settings: Settings, config_id: str, which: EvalSet, provider: Provider) -> None:
    database = Database(settings.DATABASE_URL)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(database.close)
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
            factories=_factories(settings, provider, stack),
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
        coverage_grid(
            fitness,
            occupancy,
            directory / "coverage_grid.png",
            title="Final coverage",
            agreement=data.agreement,
        ),
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


# --------------------------------------------------------------------------- #
# Phase 8: experiments and findings
# --------------------------------------------------------------------------- #


@experiment_app.command("list")
@handled
def experiment_list() -> None:
    """Every committed experiment, with a wall-clock estimate at 6500 TPM."""
    table = Table(title="experiments")
    for column in ("name", "kind", "rounds", "seed", "ablation", "~minutes"):
        table.add_column(column)
    for name in experiment_names():
        config = load_experiment(name)
        table.add_row(
            name,
            config.kind.value,
            str(config.rounds),
            str(config.seed),
            config.ablation.value,
            str(cost_estimate_minutes(config)),
        )
    console.print(table)
    console.print(
        "[dim]Estimates assume 6500 tokens per minute and an archive of ~90 attacks.[/dim]"
    )


@experiment_app.command("run")
@handled
def experiment_run(
    name: Annotated[str, typer.Argument(help="An experiment in experiments/.")],
    config_id: Annotated[
        str | None,
        typer.Option("--config-id", help="Defense config to replay. Defaults to the latest run's."),
    ] = None,
    provider: Annotated[
        Provider,
        typer.Option(
            "--provider",
            help=(
                "groq (live: each agent on the provider its settings name) or "
                "scripted (test doubles, marked stubbed)."
            ),
        ),
    ] = Provider.GROQ,
    rounds: Annotated[
        int | None, typer.Option("--rounds", help="Override the config's round count.")
    ] = None,
    smoke: Annotated[
        bool,
        typer.Option(
            "--smoke",
            help="One round, live providers, small caps: about two minutes before a long run.",
        ),
    ] = False,
) -> None:
    """Run one committed experiment and store its result."""
    settings = _settings()
    experiment = load_experiment(name)
    if smoke:
        experiment = smoke_experiment(experiment)
        console.print(
            "[cyan]smoke run[/cyan]: 1 round, "
            f"{experiment.candidates_per_round} candidates, "
            f"{experiment.cells_per_round} cell(s), budget ${experiment.budget_usd}. "
            "Checking that live providers answer, not measuring anything."
        )
    elif rounds is not None:
        experiment = experiment.model_copy(update={"rounds": rounds})
    _announce_provider(settings, provider)
    if experiment.violates_never_cut:
        console.print(
            f"[yellow]{experiment.name} switches off a never-cut property on purpose: "
            f"{experiment.ablation.value}. Its numbers are not comparable to the main run."
            "[/yellow]"
        )
    console.print(
        f"[cyan]{experiment.name}[/cyan] ({experiment.kind.value}), seed {experiment.seed}, "
        f"estimated {cost_estimate_minutes(experiment)} minutes at 6500 TPM"
    )
    asyncio.run(_run_experiment(settings, experiment, config_id, provider))


async def _run_experiment(
    settings: Settings,
    experiment: ExperimentConfig,
    config_id: str | None,
    provider: Provider,
) -> None:
    database = Database(settings.DATABASE_URL)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(database.close)
        context = ExperimentContext(
            database=database,
            embedder=default_embedder(settings),
            factories=_factories(settings, provider, stack),
            allowlist=tuple(settings.TARGET_ALLOWLIST),
            checkpointer_settings=settings if experiment.kind is ExperimentKind.LOOP else None,
            pacing=pacing_settings(settings),
        )

        if experiment.kind is ExperimentKind.LOOP:
            report = await run_loop_experiment(experiment, context)
            await record_result(
                database,
                experiment,
                LoopExperimentResult(
                    run_id=str(report.run_id),
                    rounds_completed=report.rounds_completed,
                    status=report.status.value,
                    halt_reason=report.halt_reason.value if report.halt_reason else None,
                ),
            )
            _print_run(report)
            await _print_cost(database, report)
            return

        if experiment.kind is ExperimentKind.LAYER_ABLATION:
            rows = await run_layer_ablation(experiment, context, config_id=config_id)
            await record_result(database, experiment, LayerAblationResult(rows=rows))
            table = Table(title="layer ablation")
            for column in ("layer", "block rate", "delta", "utility", "delta"):
                table.add_column(column)
            for row in rows:
                table.add_row(
                    row.layer,
                    f"{row.archive_block_rate:.3f}",
                    f"{row.delta_block_rate:+.3f}",
                    f"{row.utility_pass_rate:.3f}",
                    f"{row.delta_utility:+.3f}",
                )
            console.print(table)
            return

        if experiment.kind is ExperimentKind.TRANSFER:
            result = await run_transfer(experiment, context, config_id=config_id)
            await record_result(database, experiment, result)
            console.print(
                f"transfer to [cyan]{result.persona}[/cyan] "
                f"({'within-family' if result.within_family else 'cross-family'}): "
                f"D(0) {result.baseline_block_rate:.3f} -> final "
                f"{result.hardened_block_rate:.3f} over {result.archive_size} attacks"
            )
            return

        # model_overlap: two live clients on different families. A model this
        # account cannot reach ends the experiment naming it.
        if provider is Provider.SCRIPTED:
            raise typer.BadParameter(
                "model_overlap compares two real model families. Scripted clients would "
                "produce an overlap of a stub with itself, which is not a measurement."
            )
        overlap = await run_model_overlap(
            experiment,
            context,
            {
                model: (lambda m=model: attacker_on(settings, m, stack))  # type: ignore[misc]
                for model in experiment.models
            },
        )
        await record_result(database, experiment, overlap)
        table = Table(title="model overlap")
        for column in ("model", "attacks", "cells"):
            table.add_column(column)
        for model in overlap.models:
            table.add_row(
                model,
                str(overlap.attacks_per_model.get(model, 0)),
                str(len(overlap.cells_per_model.get(model, []))),
            )
        console.print(table)
        console.print(
            f"cell overlap (Jaccard) `{overlap.cell_overlap:.3f}`; "
            f"mean nearest-neighbour embedding distance "
            f"`{overlap.mean_nearest_distance:.3f}`"
        )


async def _print_cost(database: Database, report: RunReport) -> None:
    """What the run actually spent, from the rounds' own recorded cost.

    A total of zero after live calls means the models in use have no price, so
    `ROUND_BUDGET_USD` is inert — worth saying out loud rather than leaving to
    be noticed.
    """
    del database
    total = sum((round_report.cost_usd for round_report in report.rounds), Decimal(0))
    console.print(f"cost: ${total:.6f}")
    if total == 0 and not report.stubbed:
        console.print(
            "[yellow]cost is $0.000000 on a live run: the models in use have no entry in "
            "MODEL_PRICING, so spend records as NULL and ROUND_BUDGET_USD cannot be "
            "enforced[/yellow]"
        )


@findings_app.command("regenerate")
@handled
def findings_regenerate(
    check: Annotated[
        bool, typer.Option("--check", help="Rewrite nothing; exit 1 if a block is stale.")
    ] = False,
    path: Annotated[Path, typer.Option("--path")] = findings_report.FINDINGS_PATH,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    force_stubbed: Annotated[
        bool,
        typer.Option(
            "--force-stubbed",
            help="Write findings from a stubbed run anyway. Never for a published document.",
        ),
    ] = False,
) -> None:
    """Rewrite every generated block in docs/findings.md from stored data."""
    settings = _settings()
    # The read and write stay out here: file IO belongs in the sync command, and
    # only the database pass is async.
    data = asyncio.run(_findings_data(settings, uuid.UUID(run_id) if run_id else None))
    if data.stubbed:
        console.print(
            "[red]the most recent run is STUBBED[/red]: "
            + (", ".join(data.provenance.render_lines()) or "provenance not recorded")
        )
    findings_report.guard_stubbed(data, force=force_stubbed)
    if force_stubbed and data.stubbed:
        console.print(
            "[yellow]--force-stubbed: writing numbers produced by test doubles. "
            "This document must not be published or cited.[/yellow]"
        )
    markdown = path.read_text(encoding="utf-8")
    if check:
        stale = findings_report.stale_blocks(markdown, data)
        if stale:
            console.print(
                f"[red]{path} is stale: {', '.join(stale)}[/red]\n"
                "[dim]Run `crucible findings regenerate` and commit the result.[/dim]"
            )
            raise typer.Exit(code=1)
        console.print(f"[green]{path} matches the stored data[/green]")
        return

    rewritten = findings_report.rewrite(markdown, data)
    if rewritten == markdown:
        console.print(f"{path} already up to date")
        return
    path.write_text(rewritten, encoding="utf-8")
    console.print(f"wrote {path}")


async def _findings_data(
    settings: Settings, run_id: uuid.UUID | None
) -> findings_report.FindingsData:
    database = Database(settings.DATABASE_URL)
    try:
        return await findings_report.gather(database, run_id=run_id)
    finally:
        await database.close()
