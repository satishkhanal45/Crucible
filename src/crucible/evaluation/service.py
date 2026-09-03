"""Evaluating one `DefenseConfig`.

Phase 6 reuses this whole module, so it is standalone: give it a config and it
returns the block rate over the non-holdout archive, the holdout block rate, the
benign pass rate, latency, and cost.

**Two-stage evaluation** (docs/spec.md section 12) is explicit in the API rather
than a convention. `screen()` may use a stratified sample and returns a result
labelled `SCREENING`; `evaluate_full()` runs the entire non-holdout archive and
returns one labelled `FULL`. Only a `FULL` result can be turned into
`RoundMetrics`, so Phase 6 cannot record a screening number by accident.

Every attempt goes through the Phase 2 outcome cache, which is what makes
re-evaluating the whole archive every round affordable.
"""

from __future__ import annotations

import math
import random
import uuid
from collections.abc import Sequence
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from crucible.db.session import Database
from crucible.defenses.config import DefenseConfig
from crucible.evaluation.benign import BenignTask, TaskResult, check_task, load_benign_tasks
from crucible.execution.pool import TargetPool
from crucible.logging import get_logger
from crucible.repositories.attacks import AttackRepository
from crucible.schemas.attack import Attack
from crucible.schemas.attempt import AttemptResult
from crucible.schemas.outcome import Outcome
from crucible.target.adapter import TargetResponse

logger = get_logger(__name__)

#: The screening sample always keeps every attack that has ever breached, plus
#: at least this share of the rest (docs/spec.md section 12).
SCREENING_REMAINDER_SHARE = 0.4


class EvaluationScope(StrEnum):
    """Which of the two stages produced a number."""

    SCREENING = "screening"
    FULL = "full"


class ScreeningNotRecordable(RuntimeError):
    """A screening number was about to be written where a full one belongs."""


class AttemptRunner(Protocol):
    async def execute(
        self, attack: Attack, defense: DefenseConfig, *, force: bool = False
    ) -> AttemptResult: ...


class BenignRunner(Protocol):
    """Runs one benign query against the target under a config."""

    async def run(self, query: str, defense: DefenseConfig) -> TargetResponse: ...


class ArchiveEvaluation(BaseModel):
    """How a config did against a set of archived attacks."""

    model_config = ConfigDict(frozen=True)

    scope: EvaluationScope
    evaluated: int
    archive_size: int
    breached: int = 0
    blocked: int = 0
    refused: int = 0
    errors: int = 0
    inconclusive: int = 0
    #: Attempts where the model emitted a privileged call and Layer 5 stopped it.
    hijacks_blocked: int = 0
    cache_hits: int = 0

    @property
    def decided(self) -> int:
        return self.breached + self.blocked + self.refused

    @property
    def block_rate(self) -> float:
        """Share of decided attempts the defense stopped."""
        return (self.blocked + self.refused) / self.decided if self.decided else 0.0

    @property
    def breach_rate(self) -> float:
        return self.breached / self.decided if self.decided else 0.0

    @property
    def is_full_archive(self) -> bool:
        return self.scope is EvaluationScope.FULL and self.evaluated == self.archive_size


class UtilityEvaluation(BaseModel):
    """How much benign capability a config kept."""

    model_config = ConfigDict(frozen=True)

    total: int
    passed: int
    hard_negative_total: int = 0
    hard_negative_passed: int = 0
    failures: tuple[TaskResult, ...] = ()

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def hard_negative_pass_rate(self) -> float:
        return (
            self.hard_negative_passed / self.hard_negative_total
            if self.hard_negative_total
            else 0.0
        )


class DefenseEvaluation(BaseModel):
    """Everything measured about one config, labelled with how it was measured."""

    model_config = ConfigDict(frozen=True)

    config_id: str
    scope: EvaluationScope
    archive: ArchiveEvaluation
    utility: UtilityEvaluation
    holdout: ArchiveEvaluation | None = None
    mean_latency_ms: float = 0.0
    cost_usd: Decimal = Decimal(0)

    @property
    def archive_block_rate(self) -> float:
        return self.archive.block_rate

    @property
    def holdout_block_rate(self) -> float | None:
        return self.holdout.block_rate if self.holdout is not None else None

    @property
    def utility_pass_rate(self) -> float:
        return self.utility.pass_rate

    @property
    def recordable(self) -> bool:
        return self.scope is EvaluationScope.FULL

    def assert_recordable(self) -> None:
        if not self.recordable:
            raise ScreeningNotRecordable(
                f"this is a {self.scope.value} result over {self.archive.evaluated} of "
                f"{self.archive.archive_size} archived attacks. Only a full-archive "
                "evaluation may be recorded as a round metric (docs/spec.md section 12)"
            )


class RoundMetrics(BaseModel):
    """The numbers a round records. Constructible only from a full evaluation."""

    model_config = ConfigDict(frozen=True)

    config_id: str
    archive_block_rate: float
    holdout_block_rate: float | None
    utility_pass_rate: float
    archive_size: int
    attacks_evaluated: int
    hijacks_blocked: int
    mean_latency_ms: float
    cost_usd: Decimal

    @classmethod
    def from_evaluation(cls, evaluation: DefenseEvaluation) -> RoundMetrics:
        evaluation.assert_recordable()
        return cls(
            config_id=evaluation.config_id,
            archive_block_rate=evaluation.archive_block_rate,
            holdout_block_rate=evaluation.holdout_block_rate,
            utility_pass_rate=evaluation.utility_pass_rate,
            archive_size=evaluation.archive.archive_size,
            attacks_evaluated=evaluation.archive.evaluated,
            hijacks_blocked=evaluation.archive.hijacks_blocked,
            mean_latency_ms=evaluation.mean_latency_ms,
            cost_usd=evaluation.cost_usd,
        )


class EvaluationService:
    def __init__(
        self,
        database: Database,
        executor: AttemptRunner,
        benign_runner: BenignRunner,
        *,
        tasks: Sequence[BenignTask] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._database = database
        self._executor = executor
        self._benign = benign_runner
        self._tasks = tuple(tasks) if tasks is not None else load_benign_tasks()
        self._rng = rng or random.Random(20260905)

    @property
    def tasks(self) -> tuple[BenignTask, ...]:
        return self._tasks

    # ------------------------------------------------------------- sampling

    async def screening_sample(self) -> tuple[list[Attack], int]:
        """Every attack that ever breached, plus >= 40% of the remainder."""
        async with self._database.session() as session:
            repository = AttackRepository(session)
            archive = await repository.list_non_holdout()
            breached = await repository.ever_breached_ids()

        always = [row for row in archive if row.id in breached]
        remainder = [row for row in archive if row.id not in breached]
        wanted = math.ceil(SCREENING_REMAINDER_SHARE * len(remainder))
        sampled = self._rng.sample(remainder, wanted) if wanted else []
        selected = always + sampled
        logger.info(
            "evaluation.screening_sample",
            extra={
                "archive_size": len(archive),
                "ever_breached": len(always),
                "sampled_remainder": len(sampled),
            },
        )
        return [row.to_attack() for row in selected], len(archive)

    # ----------------------------------------------------------- evaluating

    async def evaluate_archive(
        self,
        config: DefenseConfig,
        attacks: Sequence[Attack],
        *,
        scope: EvaluationScope,
        archive_size: int | None = None,
    ) -> ArchiveEvaluation:
        counts: dict[Outcome, int] = dict.fromkeys(Outcome, 0)
        hijacks_blocked = 0
        cache_hits = 0
        latencies: list[int] = []

        for attack in attacks:
            result = await self._executor.execute(attack, config)
            counts[result.attempt.outcome] += 1
            cache_hits += int(result.cache_hit)
            if result.attempt.blocked_tools:
                hijacks_blocked += 1
            if result.attempt.latency_ms:
                latencies.append(result.attempt.latency_ms)

        return ArchiveEvaluation(
            scope=scope,
            evaluated=len(attacks),
            archive_size=archive_size if archive_size is not None else len(attacks),
            breached=counts[Outcome.BREACHED],
            blocked=counts[Outcome.BLOCKED],
            refused=counts[Outcome.REFUSED],
            errors=counts[Outcome.ERROR],
            inconclusive=counts[Outcome.INCONCLUSIVE],
            hijacks_blocked=hijacks_blocked,
            cache_hits=cache_hits,
        )

    async def evaluate_holdout(self, config: DefenseConfig) -> ArchiveEvaluation:
        """The only honest generalization number in the project."""
        async with self._database.session() as session:
            holdout = await AttackRepository(session).list_holdout()
        attacks = [row.to_attack() for row in holdout]
        return await self.evaluate_archive(
            config, attacks, scope=EvaluationScope.FULL, archive_size=len(attacks)
        )

    async def evaluate_utility(self, config: DefenseConfig) -> UtilityEvaluation:
        results: list[TaskResult] = []
        for task in self._tasks:
            response = await self._benign.run(task.query, config)
            results.append(check_task(task, response))

        hard = [result for result in results if result.hard_negative]
        return UtilityEvaluation(
            total=len(results),
            passed=sum(1 for result in results if result.passed),
            hard_negative_total=len(hard),
            hard_negative_passed=sum(1 for result in hard if result.passed),
            failures=tuple(result for result in results if not result.passed),
        )

    async def screen(self, config: DefenseConfig) -> DefenseEvaluation:
        """Cheap candidate screening. The result may never be recorded."""
        attacks, archive_size = await self.screening_sample()
        archive = await self.evaluate_archive(
            config, attacks, scope=EvaluationScope.SCREENING, archive_size=archive_size
        )
        utility = await self.evaluate_utility(config)
        return DefenseEvaluation(
            config_id=config.fingerprint(),
            scope=EvaluationScope.SCREENING,
            archive=archive,
            utility=utility,
            mean_latency_ms=0.0,
        )

    async def evaluate_full(
        self, config: DefenseConfig, *, include_holdout: bool = True
    ) -> DefenseEvaluation:
        """The full non-holdout archive. These are the numbers a round records."""
        async with self._database.session() as session:
            rows = await AttackRepository(session).list_non_holdout()
        attacks = [row.to_attack() for row in rows]

        archive = await self.evaluate_archive(
            config, attacks, scope=EvaluationScope.FULL, archive_size=len(attacks)
        )
        utility = await self.evaluate_utility(config)
        holdout = await self.evaluate_holdout(config) if include_holdout else None

        return DefenseEvaluation(
            config_id=config.fingerprint(),
            scope=EvaluationScope.FULL,
            archive=archive,
            utility=utility,
            holdout=holdout,
        )


class PoolBenignRunner:
    """Runs benign tasks against a pooled target, isolated like an attempt."""

    def __init__(self, pool: TargetPool) -> None:
        self._pool = pool

    async def run(self, query: str, defense: DefenseConfig) -> TargetResponse:
        async with self._pool.acquire() as target:
            await target.reset()
            try:
                return await target.query(query, defense, f"benign-{uuid.uuid4().hex[:8]}")
            finally:
                await target.reset()
