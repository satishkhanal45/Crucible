"""The execution sandbox.

Per attempt, in this order (docs/spec.md section 9):

    reset -> plant fresh canaries -> apply the DefenseConfig -> deliver by
    vector -> capture response and trace -> run the oracle -> persist -> reset

The properties that make the numbers trustworthy, all enforced here:

* **Egress** is restricted to `TARGET_ALLOWLIST` plus the LLM providers. A
  target whose endpoint is off the list never runs, and an `EgressViolation`
  raised mid-attempt is recorded as `error`.
* **Timeouts** are `error`, never `blocked`. Conflating them would let a target
  that times out look like a target with excellent defenses, inflating every
  block rate in the project.
* **Isolation**: every attempt gets a pool slot with its own corpus namespace,
  its own session id, and its own freshly minted canaries.
* **Replay**: every attempt stores a trace sufficient for `replay()` to
  reproduce the outcome without calling a model.
* **The outcome cache**: `(attack_id, defense_config_id)` is unique, the target
  runs at temperature 0, and a config's id is an order-independent fingerprint,
  so a pair that has already run is not run again.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from crucible.db.session import Database
from crucible.defenses.config import DefenseConfig
from crucible.execution.egress import EgressGuard, EgressViolation
from crucible.execution.pool import TargetPool
from crucible.execution.vectors import Delivery, deliver
from crucible.logging import get_logger
from crucible.oracle import Oracle
from crucible.oracle.results import OracleVerdict
from crucible.repositories.attempts import AttemptRepository
from crucible.schemas.attack import Attack
from crucible.schemas.attempt import (
    AttemptRecord,
    AttemptResult,
    ExecutionMetrics,
    NewAttempt,
)
from crucible.schemas.outcome import Outcome, Tier
from crucible.services.cost_meter import CostMeter
from crucible.target.adapter import BehaviorSpec, TargetAdapter, TargetResponse, ToolCall
from crucible.target.canary import CanarySet

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_CONCURRENCY = 5


class ExecutorSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    concurrency: int = DEFAULT_CONCURRENCY


class ReplayResult(BaseModel):
    """What replaying a stored attempt produced."""

    model_config = ConfigDict(frozen=True)

    attempt_id: uuid.UUID
    original_outcome: Outcome
    replayed_outcome: Outcome
    verdict: OracleVerdict | None = None
    detail: str | None = None

    @property
    def matches(self) -> bool:
        return self.original_outcome is self.replayed_outcome


class AttemptExecutor:
    """Runs attacks against a target under a defense configuration."""

    def __init__(
        self,
        pool: TargetPool,
        oracle: Oracle,
        database: Database,
        *,
        egress_guard: EgressGuard,
        settings: ExecutorSettings | None = None,
        cost_meter: CostMeter | None = None,
        round_id: uuid.UUID | None = None,
    ) -> None:
        self._pool = pool
        self._oracle = oracle
        self._database = database
        self._guard = egress_guard
        self._settings = settings or ExecutorSettings()
        self._cost_meter = cost_meter
        self._round_id = round_id
        self._metrics = ExecutionMetrics()

    @property
    def metrics(self) -> ExecutionMetrics:
        return self._metrics

    @property
    def settings(self) -> ExecutorSettings:
        return self._settings

    def _count(self, **deltas: int) -> None:
        current = self._metrics.model_dump()
        for key, delta in deltas.items():
            current[key] += delta
        self._metrics = ExecutionMetrics(**current)

    # ------------------------------------------------------------- executing

    async def execute(
        self, attack: Attack, defense: DefenseConfig, *, force: bool = False
    ) -> AttemptResult:
        """Execute one attack against one defense config, or reuse the cache."""
        self._count(requested=1)
        config_id = defense.fingerprint()

        if force:
            logger.warning(
                "outcome_cache.bypassed",
                extra={
                    "attack_id": str(attack.attack_id),
                    "defense_config_id": config_id,
                    "reason": "force=True was passed; this re-runs a pair that is "
                    "already decided and costs a model call",
                },
            )
        else:
            cached = await self._cached(attack.attack_id, config_id)
            if cached is not None:
                self._count(cache_hits=1)
                logger.info(
                    "outcome_cache.hit",
                    extra={
                        "attack_id": str(attack.attack_id),
                        "defense_config_id": config_id,
                        "outcome": cached.outcome.value,
                    },
                )
                return AttemptResult(attempt=cached, cache_hit=True)

        record = await self._run(attack, defense, config_id)
        self._count(executed=1)
        return AttemptResult(attempt=record, cache_hit=False)

    async def execute_many(
        self, attacks: Sequence[Attack], defense: DefenseConfig, *, force: bool = False
    ) -> list[AttemptResult]:
        """Execute a batch. Concurrency is bounded by the target pool."""
        tasks = [
            asyncio.create_task(self.execute(attack, defense, force=force)) for attack in attacks
        ]
        return list(await asyncio.gather(*tasks))

    async def _cached(self, attack_id: uuid.UUID, config_id: str) -> AttemptRecord | None:
        async with self._database.session() as session:
            return await AttemptRepository(session).find_cached(attack_id, config_id)

    async def _run(self, attack: Attack, defense: DefenseConfig, config_id: str) -> AttemptRecord:
        session_id = f"attempt-{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()

        async with self._pool.acquire() as target:
            canaries = CanarySet.mint()
            delivery: Delivery | None = None
            failure: str | None = None
            outcome = Outcome.ERROR

            try:
                self._check_endpoint(target)
                await target.reset()
                await target.plant_canaries(canaries)
                delivery = await asyncio.wait_for(
                    deliver(target, attack, defense, session_id),
                    timeout=self._settings.timeout_seconds,
                )
            except TimeoutError:
                # A timeout is an error, never a block: see the module docstring.
                self._count(timeouts=1, errors=1)
                failure = (
                    f"attempt exceeded the {self._settings.timeout_seconds:g}s wall-clock timeout"
                )
                logger.warning(
                    "attempt.timeout",
                    extra={"attack_id": str(attack.attack_id), "session_id": session_id},
                )
            except EgressViolation as violation:
                self._count(egress_violations=1, errors=1)
                failure = str(violation)
                logger.warning(
                    "attempt.egress_violation",
                    extra={
                        "attack_id": str(attack.attack_id),
                        "attempted_host": violation.host,
                    },
                )
            except Exception as error:  # any failure of an attempt is an `error`
                self._count(errors=1)
                failure = f"{type(error).__name__}: {error}"
                logger.warning(
                    "attempt.failed",
                    extra={"attack_id": str(attack.attack_id), "error": failure},
                )

            confidential = await self._confidential_ids(target)
            verdict: OracleVerdict | None = None
            if delivery is not None:
                verdict = await self._oracle.judge_attempt(
                    delivery.response,
                    canaries,
                    attack,
                    target.capabilities().behavior,
                    confidential_doc_ids=confidential,
                )
                outcome = verdict.outcome

            try:
                await target.reset()
            except Exception as error:  # a reset failure must not mask the verdict
                logger.warning("attempt.reset_failed", extra={"error": str(error)})

        latency_ms = int((time.perf_counter() - started) * 1000)
        trace = self._build_trace(
            attack=attack,
            defense=defense,
            config_id=config_id,
            session_id=session_id,
            canaries=canaries,
            delivery=delivery,
            verdict=verdict,
            failure=failure,
            confidential=confidential,
        )
        return await self._persist(
            attack=attack,
            config_id=config_id,
            outcome=outcome,
            verdict=verdict,
            delivery=delivery,
            trace=trace,
            latency_ms=latency_ms,
        )

    def _check_endpoint(self, target: TargetAdapter) -> None:
        """A target off the allowlist never runs (docs/spec.md section 1)."""
        endpoint = target.capabilities().endpoint
        if endpoint:
            self._guard.check_url(endpoint)

    async def _confidential_ids(self, target: TargetAdapter) -> tuple[str, ...]:
        """Doc ids Tier 1 treats as confidential, read from the live corpus."""
        store = getattr(target, "store", None)
        if store is None:
            return ()
        documents = await store.documents(include_confidential=True)
        return tuple(document.doc_id for document in documents if document.confidential)

    def _build_trace(
        self,
        *,
        attack: Attack,
        defense: DefenseConfig,
        config_id: str,
        session_id: str,
        canaries: CanarySet,
        delivery: Delivery | None,
        verdict: OracleVerdict | None,
        failure: str | None,
        confidential: Sequence[str],
    ) -> dict[str, Any]:
        """Everything `replay()` needs, and nothing a report may print raw.

        The canary values for this attempt are included deliberately: without
        them the oracle cannot be re-run, and an attempt that cannot be replayed
        is not a finding. They are single-use per attempt.

        TODO(phase-7): reports redact `canaries` and `attempt.payload` unless
        `--include-payloads` is given.
        """
        trace: dict[str, Any] = {
            "schema_version": 1,
            "attempt": {
                "attack_id": str(attack.attack_id),
                "vector": attack.vector.value,
                # None when the classifier could not place the attack.
                "objective": attack.objective.value if attack.objective else None,
                "technique": attack.technique.value if attack.technique else None,
                "payload": attack.payload,
                "benign_user_input": attack.benign_user_input,
                "carrier_title": attack.carrier_title,
                "carrier_doc_id": attack.carrier_doc_id,
                "session_id": session_id,
                "defense_config_id": config_id,
                "defense": defense.model_dump(mode="json"),
                "timeout_seconds": self._settings.timeout_seconds,
            },
            "canaries": canaries.to_values(),
            "confidential_doc_ids": list(confidential),
        }
        if failure is not None:
            trace["error"] = failure
        if delivery is not None:
            response = delivery.response
            trace["delivery"] = {
                "user_input": delivery.user_input,
                "injected_doc_id": delivery.injected_doc_id,
                "carrier_retrieved": delivery.carrier_retrieved,
            }
            trace["response"] = {
                "text": response.text,
                "tool_calls": [call.model_dump(mode="json") for call in response.tool_calls],
                "retrieved_doc_ids": list(response.retrieved_doc_ids),
                "system_prompt_hash": response.system_prompt_hash,
                "latency_ms": response.latency_ms,
                "tokens": response.tokens.model_dump(mode="json"),
            }
            trace["target"] = response.raw_trace
        if verdict is not None:
            trace["oracle"] = verdict.model_dump(mode="json")
        return trace

    async def _persist(
        self,
        *,
        attack: Attack,
        config_id: str,
        outcome: Outcome,
        verdict: OracleVerdict | None,
        delivery: Delivery | None,
        trace: dict[str, Any],
        latency_ms: int,
    ) -> AttemptRecord:
        response = delivery.response if delivery is not None else None
        attempt = NewAttempt(
            attack_id=attack.attack_id,
            defense_config_id=config_id,
            round_id=self._round_id,
            vector=attack.vector,
            outcome=outcome,
            tier=verdict.tier if verdict is not None else Tier.NONE,
            canaries_hit=tuple(c.value for c in verdict.canaries_hit) if verdict else (),
            unauthorized_tools=verdict.unauthorized_tools if verdict else (),
            blocked_tools=verdict.tier1.blocked_tools if verdict else (),
            judge_score=(
                Decimal(str(verdict.tier3.score))
                if verdict is not None and verdict.tier3.score is not None
                else None
            ),
            judge_rationale=verdict.tier3.rationale if verdict is not None else None,
            response_text=response.text if response is not None else None,
            trace=trace,
            latency_ms=latency_ms,
            cost_usd=self._estimate_cost(response),
        )
        async with self._database.session() as session:
            return await AttemptRepository(session).add(attempt)

    def _estimate_cost(self, response: TargetResponse | None) -> Decimal | None:
        """What this attempt cost, priced the same way `CostMeter` prices it."""
        if response is None or self._cost_meter is None:
            return None
        provider = str(response.raw_trace.get("provider", ""))
        model = str(response.raw_trace.get("model", ""))
        if not provider or not model:
            return None
        return self._cost_meter.estimate(provider, model, response.tokens)

    # -------------------------------------------------------------- replaying

    async def replay(self, attempt_id: uuid.UUID) -> ReplayResult:
        """Re-decide a stored attempt from its trace, without calling a model.

        A breach you cannot reproduce is not a finding, so this re-runs the
        oracle over exactly what was stored and reports whether the outcome
        still holds.
        """
        async with self._database.session() as session:
            stored = await AttemptRepository(session).get(attempt_id)
        if stored is None:
            raise KeyError(f"no attempt with id {attempt_id}")

        trace = stored.trace
        if "error" in trace or "response" not in trace:
            return ReplayResult(
                attempt_id=attempt_id,
                original_outcome=stored.outcome,
                replayed_outcome=Outcome.ERROR,
                detail=str(trace.get("error", "the attempt recorded no response")),
            )

        attack = _attack_from_trace(trace)
        response = _response_from_trace(trace)
        canaries = CanarySet.restore(trace.get("canaries", {}))
        behavior = _behavior_from_trace(trace)

        verdict = await self._oracle.judge_attempt(
            response,
            canaries,
            attack,
            behavior,
            confidential_doc_ids=trace.get("confidential_doc_ids", []),
        )
        return ReplayResult(
            attempt_id=attempt_id,
            original_outcome=stored.outcome,
            replayed_outcome=verdict.outcome,
            verdict=verdict,
        )


def _attack_from_trace(trace: dict[str, Any]) -> Attack:
    recorded = trace["attempt"]
    return Attack(
        attack_id=uuid.UUID(recorded["attack_id"]),
        payload=recorded["payload"],
        vector=recorded["vector"],
        objective=recorded["objective"],
        technique=recorded["technique"],
        benign_user_input=recorded.get("benign_user_input") or "replayed",
        carrier_title=recorded.get("carrier_title"),
        carrier_doc_id=recorded.get("carrier_doc_id"),
    )


def _response_from_trace(trace: dict[str, Any]) -> TargetResponse:
    recorded = trace["response"]
    return TargetResponse(
        text=recorded["text"],
        tool_calls=[ToolCall.model_validate(call) for call in recorded["tool_calls"]],
        retrieved_doc_ids=list(recorded["retrieved_doc_ids"]),
        system_prompt_hash=recorded["system_prompt_hash"],
        latency_ms=recorded.get("latency_ms", 0),
        raw_trace=trace.get("target", {}),
    )


def _behavior_from_trace(trace: dict[str, Any]) -> BehaviorSpec:
    """The behaviour spec to re-judge against.

    TODO(phase-8): store the target's `BehaviorSpec` in the trace so a replay of
    an attempt against a *different* target still uses that target's rules.
    """
    del trace
    return BehaviorSpec()
