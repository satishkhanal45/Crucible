"""Assembling and running the loop.

`LoopRunner` owns the run record, the checkpointer, and the graph. A nine-round
run takes hours and will meet a rate limit, a network blip, or a closed laptop,
so every step is checkpointed to Postgres and `resume` picks up from the last
completed node rather than the last completed round.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from crucible.archive.classifier import ClassifierClient, TaxonomyClassifier
from crucible.archive.service import ArchiveService, BoundNoveltyGate
from crucible.archive.survey import ArchiveSurvey
from crucible.attacker.graph import Attacker
from crucible.attacker.llm import AttackerLLM, MeteredAttackerLLM
from crucible.attacker.state import AttackerMode, AttackerSettings
from crucible.config import Settings
from crucible.db.session import Database
from crucible.defender.graph import Defender
from crucible.defender.llm import DefenderLLM, MeteredDefenderLLM
from crucible.defenses.config import DefenseConfig
from crucible.evaluation.service import EvaluationService, PoolBenignRunner
from crucible.execution.egress import EgressGuard
from crucible.execution.executor import AttemptExecutor, ExecutorSettings
from crucible.execution.pool import TargetPool
from crucible.logging import get_logger
from crucible.loop.graph import CoEvolutionLoop, LoopComponents
from crucible.loop.reports import HaltReason, RoundReport, RunReport, RunStatus
from crucible.loop.state import LoopState
from crucible.oracle import Oracle
from crucible.repositories.configs import DefenseConfigRepository
from crucible.repositories.rounds import RoundRepository, RunRepository
from crucible.repositories.spend import DatabaseSpendRepository
from crucible.services.cost_meter import BudgetExceeded, CostMeter, ModelPrice
from crucible.services.embeddings import Embedder, embedder_from_settings
from crucible.target.adapter import TargetAdapter
from crucible.target.canary import CanarySet
from crucible.target.reference.llm import MeteredTargetLLM, TargetLLM
from crucible.target.reference.store import DocumentStore
from crucible.target.reference.target import ReferenceTarget

logger = get_logger(__name__)


class LoopSettings(BaseModel):
    """How one run is configured."""

    model_config = ConfigDict(frozen=True)

    rounds: int = Field(default=8, ge=1, le=50)
    mode: AttackerMode = AttackerMode.BLACK_BOX
    budget_usd: Decimal = Decimal("5.00")
    seed: int = 20260906
    concurrency: int = Field(default=2, ge=1, le=8)
    candidates_per_round: int = Field(default=4, ge=3, le=5)
    cells_per_round: int = Field(default=4, ge=1, le=16)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass
class LoopFactories:
    """Model clients, injected so tests can script them."""

    target_llm: Callable[[], TargetLLM]
    attacker_llm: Callable[[], AttackerLLM]
    defender_llm: Callable[[], DefenderLLM]
    classifier_client: Callable[[], ClassifierClient]


async def build_components(
    database: Database,
    *,
    settings: LoopSettings,
    factories: LoopFactories,
    embedder: Embedder,
    allowlist: tuple[str, ...] = ("localhost", "127.0.0.1"),
    corpus: list[Any] | None = None,
    pricing: Mapping[str, ModelPrice] | None = None,
) -> LoopComponents:
    """Wire every component one run needs."""
    cost_meter = (
        CostMeter(DatabaseSpendRepository(database), settings.budget_usd, pricing=pricing)
        if pricing is not None
        else CostMeter(DatabaseSpendRepository(database), settings.budget_usd)
    )

    async def factory(namespace: str) -> TargetAdapter:
        store = DocumentStore(database, embedder, namespace=namespace)
        target = ReferenceTarget(
            store,
            MeteredTargetLLM(factories.target_llm(), cost_meter),
            CanarySet.mint(),
        )
        await target.seed(corpus)
        return target

    pool = TargetPool(factory, size=settings.concurrency)
    executor = AttemptExecutor(
        pool,
        Oracle(),
        database,
        egress_guard=EgressGuard(allowlist),
        settings=ExecutorSettings(concurrency=settings.concurrency),
        cost_meter=cost_meter,
    )
    # Seeded from the run, so holdout assignment is reproducible: two runs with
    # the same seed must build the same archive, holdout set included.
    archive = ArchiveService(database, embedder, rng=random.Random(settings.seed))
    evaluation = EvaluationService(database, executor, PoolBenignRunner(pool))
    classifier = TaxonomyClassifier(factories.classifier_client(), cost_meter)

    attacker = Attacker(
        MeteredAttackerLLM(factories.attacker_llm(), cost_meter),
        ArchiveSurvey(database),
        BoundNoveltyGate(archive, DefenseConfig.empty(), executor),
        classifier,
        settings=AttackerSettings(mode=settings.mode, cells_per_round=settings.cells_per_round),
    )
    defender = Defender(
        MeteredDefenderLLM(factories.defender_llm(), cost_meter),
        evaluation,
        candidates=settings.candidates_per_round,
    )

    async with pool.acquire() as target:
        capabilities = target.capabilities()

    return LoopComponents(
        database=database,
        archive=archive,
        evaluation=evaluation,
        executor=executor,
        attacker=attacker,
        defender=defender,
        cost_meter=cost_meter,
        capabilities=capabilities,
        behavior=capabilities.behavior,
        budget_usd=settings.budget_usd,
    )


def checkpointer_url(database_url: str) -> str:
    """The psycopg URL LangGraph's saver wants, from our SQLAlchemy URL."""
    return database_url.replace("+asyncpg", "").replace("postgresql+psycopg", "postgresql")


class LoopRunner:
    """Starts and resumes runs."""

    def __init__(
        self,
        database: Database,
        components: LoopComponents,
        *,
        settings: LoopSettings,
        checkpointer: Any | None = None,
    ) -> None:
        self._database = database
        self._settings = settings
        self._loop = CoEvolutionLoop(components)
        self._checkpointer = checkpointer
        self._graph = self._loop.build(checkpointer)
        self._last_run_id: uuid.UUID | None = None

    @property
    def loop(self) -> CoEvolutionLoop:
        return self._loop

    async def graph_state(self, run_id: uuid.UUID | None = None) -> LoopState:
        """The checkpointed state of a run, for inspection and for `status`."""
        target = run_id or self._last_run_id
        if target is None:
            raise RuntimeError("no run has been started by this runner")
        config = RunnableConfig(configurable={"thread_id": str(target)}, recursion_limit=200)
        snapshot = await self._graph.aget_state(config)
        return cast(LoopState, snapshot.values)

    async def start(
        self,
        *,
        run_id: uuid.UUID | None = None,
        starting_config: DefenseConfig | None = None,
        interrupt_before: tuple[str, ...] = (),
    ) -> RunReport:
        """Begin a run.

        D(0) is `DefenseConfig.empty()` by decision of the technical director:
        the loop has to start weak and be shown to harden. Starting from the
        hand-written config would leave six points of headroom and no story.
        """
        run = run_id or uuid.uuid4()
        self._last_run_id = run
        config = starting_config or DefenseConfig.empty()

        try:
            baseline = await self._loop._parts.evaluation.evaluate_utility(config)
            baseline_rate = baseline.pass_rate
        except BudgetExceeded:
            # No budget even for the baseline: the run still opens, and the first
            # round halts cleanly rather than crashing here.
            logger.warning("loop.budget_exceeded", extra={"node": "baseline_utility"})
            baseline_rate = 1.0
        async with self._database.session() as session:
            await DefenseConfigRepository(session).save(config, label="D(0)")
            await RunRepository(session).create(
                run_id=run,
                attacker_mode=self._settings.mode.value,
                rounds_planned=self._settings.rounds,
                starting_config_id=config.fingerprint(),
                budget_usd=self._settings.budget_usd,
                seed=self._settings.seed,
                settings=self._settings.to_dict(),
            )

        initial: LoopState = {
            "run_id": str(run),
            "round_number": 0,
            "rounds_planned": self._settings.rounds,
            "attacker_mode": self._settings.mode.value,
            "seed": self._settings.seed,
            "current_config": config,
            "previous_config": None,
            "baseline_utility": baseline_rate,
            "events": [],
            "signals": [],
            "reports": [],
            "status": RunStatus.RUNNING.value,
        }
        logger.info(
            "loop.start",
            extra={
                "run_id": str(run),
                "rounds": self._settings.rounds,
                "starting_config": config.fingerprint(),
                "mode": self._settings.mode.value,
            },
        )
        return await self._drive(run, initial, interrupt_before=interrupt_before)

    async def resume(
        self, run_id: uuid.UUID, *, interrupt_before: tuple[str, ...] = ()
    ) -> RunReport:
        """Continue a checkpointed run from where it stopped.

        Two kinds of stop, two kinds of continuation:

        * **Interrupted mid-round** — the thread still has pending tasks, so the
          graph is resumed with no new input and picks up at the node it had not
          reached. Nothing already done is redone.
        * **Halted between rounds** (a budget cap, say) — the thread has no
          pending tasks, so the halt is cleared and the next round begins from
          the state the run already has. Completed rounds stay recorded.
        """
        if self._checkpointer is None:
            raise RuntimeError("resume needs a checkpointer: none was configured")
        logger.info("loop.resume", extra={"run_id": str(run_id)})
        self._last_run_id = run_id

        config = RunnableConfig(configurable={"thread_id": str(run_id)}, recursion_limit=200)
        snapshot = await self._graph.aget_state(config)
        if snapshot.next:
            return await self._drive(run_id, None, interrupt_before=interrupt_before)

        cleared: LoopState = {"halt_reason": None, "status": RunStatus.RUNNING.value}
        return await self._drive(run_id, cleared, interrupt_before=interrupt_before)

    async def _drive(
        self,
        run_id: uuid.UUID,
        initial: LoopState | None,
        *,
        interrupt_before: tuple[str, ...],
    ) -> RunReport:
        config = RunnableConfig(configurable={"thread_id": str(run_id)}, recursion_limit=200)
        if interrupt_before:
            # Used by the resume tests to stop the graph mid-round exactly where
            # a rate limit or a closed laptop would have.
            result = await self._graph.ainvoke(
                initial, config=config, interrupt_before=list(interrupt_before)
            )
        else:
            result = await self._graph.ainvoke(initial, config=config)

        state: LoopState = dict(result)  # type: ignore[assignment]
        return await self._finish(run_id, state)

    async def _finish(self, run_id: uuid.UUID, state: LoopState) -> RunReport:
        halt = state.get("halt_reason")
        reason = HaltReason(halt) if halt else None
        interrupted = not state.get("reports") and not state.get("round_number")
        status = (
            RunStatus.HALTED
            if reason
            else RunStatus.RUNNING
            if interrupted
            else RunStatus.COMPLETED
        )
        current = state.get("current_config") or DefenseConfig.empty()

        async with self._database.session() as session:
            rounds = await RoundRepository(session).list_for_run(run_id)
            await RunRepository(session).finish(
                run_id,
                status=status,
                halt_reason=reason,
                rounds_completed=len(rounds),
                current_config_id=current.fingerprint(),
                ended_at=datetime.now(UTC) if status is not RunStatus.RUNNING else None,
            )
            run_row = await RunRepository(session).get(run_id)

        return RunReport(
            run_id=run_id,
            status=status,
            attacker_mode=str(state.get("attacker_mode", self._settings.mode.value)),
            starting_config_id=run_row.starting_config_id if run_row else "",
            final_config_id=current.fingerprint(),
            rounds=tuple(rounds),
            halt_reason=reason,
        )


async def load_run_report(database: Database, run_id: uuid.UUID) -> RunReport | None:
    """Rebuild a run's report from the database, for `loop status` and reports."""
    async with database.session() as session:
        run = await RunRepository(session).get(run_id)
        if run is None:
            return None
        rounds: list[RoundReport] = await RoundRepository(session).list_for_run(run_id)
    return RunReport(
        run_id=run_id,
        status=RunStatus(run.status),
        attacker_mode=run.attacker_mode,
        starting_config_id=run.starting_config_id,
        final_config_id=run.current_config_id,
        rounds=tuple(rounds),
        halt_reason=HaltReason(run.halt_reason) if run.halt_reason else None,
    )


async def postgres_checkpointer(stack: AsyncExitStack, settings: Settings) -> Any:
    """An `AsyncPostgresSaver` with its tables ensured."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    saver = await stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(checkpointer_url(settings.DATABASE_URL))
    )
    await saver.setup()
    return saver


def default_embedder(settings: Settings) -> Embedder:
    return embedder_from_settings(settings)


__all__ = [
    "LoopFactories",
    "LoopRunner",
    "LoopSettings",
    "build_components",
    "checkpointer_url",
    "default_embedder",
    "load_run_report",
    "postgres_checkpointer",
]
