"""The co-evolution loop: ten steps, in this order, every round.

    1. attacker generates against D(n-1)
    2. novelty filter; survivors execute against D(n-1)
    3. oracle scores; archive, cells and elites update
    4. defender sees NON-HOLDOUT breaches from step 3
    5. candidates evaluated on the full archive, the benign tasks, latency/cost
    6. select D(n)
    7. REGRESSION CHECK
    8. GENERALIZATION CHECK on the holdout set
    9. record the RoundReport
   10. collapse detection; halt if triggered

Three of those steps are what make this an experiment rather than a demo.

Step 7 is why this is a *loop*: without it the defender fixes round 5 by
reopening round 2, the curve looks great, and the system gets worse. A config
that reopens an archived attack is named and **not promoted**.

Step 8 produces the only honest number in the project. The holdout set is
evaluated every round and its result never reaches the defender in any form.

Step 5 uses the full non-holdout archive for the selected config, never a
screening sample — `EvaluationScope` makes recording a screening number
impossible rather than merely discouraged.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from crucible.archive.service import ArchiveService
from crucible.attacker.graph import Attacker
from crucible.attacker.prompts import build_defense_summary
from crucible.attacker.state import AttackerMode, AttackerState, OutcomeSummary
from crucible.db.session import Database
from crucible.defender.graph import Defender
from crucible.defender.state import BreachSummary, DefenderState
from crucible.defenses.config import DefenseConfig
from crucible.evaluation.service import EvaluationService, RoundMetrics
from crucible.execution.executor import AttemptExecutor
from crucible.logging import get_logger
from crucible.loop.collapse import RoundSignals, detect
from crucible.loop.regression import find_regressions
from crucible.loop.reports import HaltReason, Regression, RoundReport, RunStatus
from crucible.loop.state import LoopState, event, round_id_for
from crucible.loop.statistics import Proportion
from crucible.repositories.attacks import AttackRepository
from crucible.repositories.attempts import AttemptRepository
from crucible.repositories.configs import DefenseConfigRepository
from crucible.repositories.rounds import RoundRepository
from crucible.schemas.outcome import Outcome
from crucible.services.cost_meter import BudgetExceeded, CostMeter
from crucible.target.adapter import BehaviorSpec, TargetCapabilities

logger = get_logger(__name__)


@dataclass
class LoopComponents:
    """Everything one round needs. Built once per run, scoped per round."""

    database: Database
    archive: ArchiveService
    evaluation: EvaluationService
    executor: AttemptExecutor
    attacker: Attacker
    defender: Defender
    cost_meter: CostMeter
    capabilities: TargetCapabilities
    behavior: BehaviorSpec
    budget_usd: Decimal

    def scope_round(self, round_id: uuid.UUID) -> None:
        """Point every metered client at this round's cost bucket."""
        for client in (self.attacker.llm, self.defender.llm, self.executor.llm):
            setter = getattr(client, "set_round", None)
            if callable(setter):
                setter(round_id)


class CoEvolutionLoop:
    """The round loop, as a checkpointed LangGraph graph."""

    def __init__(self, components: LoopComponents) -> None:
        self._parts = components
        self._graph: CompiledStateGraph[LoopState] | None = None

    # ------------------------------------------------------------ the steps

    async def start_round(self, state: LoopState) -> LoopState:
        """Open a round: scope the cost bucket and record where coverage began."""
        round_number = int(state.get("round_number", 0)) + 1
        run_id = str(state["run_id"])
        round_id = round_id_for(run_id, round_number)
        self._parts.scope_round(round_id)

        coverage = await self._parts.archive.coverage()
        return {
            "round_number": round_number,
            "round_id": str(round_id),
            "round_started_at": datetime.now(UTC),
            "cells_before": coverage.occupied,
            "before_outcomes": {},
            "regressions": [],
            "config_promoted": True,
            "selected_config": None,
        }

    async def attack(self, state: LoopState) -> LoopState:
        """Steps 1-3: generate, filter for novelty, execute, score, archive."""
        round_number = int(state["round_number"])
        config = state["current_config"]
        outcomes = await self._outcome_summary(config)

        attacker_state: AttackerState = {
            "round": round_number,
            "target_capabilities": self._parts.capabilities,
            "behavior_spec": self._parts.behavior,
            "current_defense_summary": build_defense_summary(
                AttackerMode(state.get("attacker_mode", AttackerMode.BLACK_BOX.value)),
                outcomes,
                defense=(
                    config if state.get("attacker_mode") == AttackerMode.WHITE_BOX.value else None
                ),
            ),
        }
        try:
            result = await self._parts.attacker.run(attacker_state)
        except BudgetExceeded as exceeded:
            logger.warning("loop.budget_exceeded", extra={"node": "attack"})
            return {
                "attacks_generated": 0,
                "attacks_rejected_novelty": 0,
                "mean_novelty": 0.0,
                "breaches_found": 0,
                "halt_reason": HaltReason.BUDGET_EXCEEDED.value,
                "events": [
                    event(round_number, 1, error=str(exceeded)),
                    event(round_number, 2, executed=0),
                    event(round_number, 3, breaches=0),
                ],
            }

        accepted = result.get("accepted", [])
        rejected = [item for item in result.get("rejected", []) if item.stage == "novelty"]
        novelties = [item.novelty for item in rejected if item.novelty is not None]
        breaches = await self._breaches_this_round(config, [a.attack_id for a in accepted])

        # Step 3 also establishes where D(n-1) stands across the *whole*
        # non-holdout archive, including the attacks this round just added. Those
        # per-attack outcomes are the baseline the regression check in step 7
        # compares against, and the outcome cache makes the repeat free.
        try:
            baseline = await self._parts.evaluation.evaluate_full(config, include_holdout=False)
        except BudgetExceeded as exceeded:
            logger.warning("loop.budget_exceeded", extra={"node": "attack_baseline"})
            return {
                "attacks_generated": len(accepted),
                "attacks_rejected_novelty": len(rejected),
                "mean_novelty": 0.0,
                "breaches_found": len(breaches),
                "halt_reason": HaltReason.BUDGET_EXCEEDED.value,
                "events": [
                    event(round_number, 1, candidates=len(result.get("candidates", []))),
                    event(round_number, 2, accepted=len(accepted), rejected=len(rejected)),
                    event(round_number, 3, error=str(exceeded)),
                ],
            }

        return {
            "before_outcomes": {
                str(attack_id): outcome.value for attack_id, outcome in baseline.archive.outcomes
            },
            "attacks_generated": len(accepted),
            "attacks_rejected_novelty": len(rejected),
            "mean_novelty": (sum(novelties) / len(novelties) if novelties and not accepted else 1.0)
            if not accepted
            else await self._mean_novelty([a.attack_id for a in accepted]),
            "breaches_found": len(breaches),
            "events": [
                event(round_number, 1, candidates=len(result.get("candidates", []))),
                event(round_number, 2, accepted=len(accepted), rejected=len(rejected)),
                event(
                    round_number,
                    3,
                    breaches=len(breaches),
                    archive_evaluated=baseline.archive.evaluated,
                ),
            ],
        }

    async def defend(self, state: LoopState) -> LoopState:
        """Steps 4-6: show the defender non-holdout breaches, screen, select."""
        round_number = int(state["round_number"])
        config = state["current_config"]
        breaches = await self._breaches_for_defender(config)

        defender_state: DefenderState = {
            "round": round_number,
            "current_config": config,
            "breaches": breaches,
            "utility_baseline": float(state.get("baseline_utility", 1.0)),
        }
        try:
            result = await self._parts.defender.run(defender_state)
        except BudgetExceeded as exceeded:
            logger.warning("loop.budget_exceeded", extra={"node": "defend"})
            return {
                "selected_config": None,
                "halt_reason": HaltReason.BUDGET_EXCEEDED.value,
                "events": [
                    event(round_number, 4, breaches=len(breaches)),
                    event(round_number, 5, error=str(exceeded)),
                    event(round_number, 6, selected=None),
                ],
            }

        chosen = result.get("chosen") or config
        return {
            "selected_config": chosen,
            "events": [
                event(round_number, 4, breaches=len(breaches), holdout=0),
                event(
                    round_number,
                    5,
                    candidates=len(result.get("validated", [])),
                    screened=len(result.get("eval_results", {})),
                ),
                event(round_number, 6, selected=chosen.fingerprint(), status=result.get("status")),
            ],
        }

    async def check_regression(self, state: LoopState) -> LoopState:
        """Step 7: full-archive evaluation of D(n), then the regression check.

        A config that reopens an archived attack is named and not promoted. That
        is the never-cut mechanism: it is what stops the defender from fixing
        this round by undoing an earlier one.
        """
        round_number = int(state["round_number"])
        current = state["current_config"]
        selected = state.get("selected_config") or current

        if selected.fingerprint() == current.fingerprint():
            try:
                evaluation = await self._parts.evaluation.evaluate_full(
                    current, include_holdout=False
                )
            except BudgetExceeded as exceeded:
                logger.warning("loop.budget_exceeded", extra={"node": "check_regression"})
                return {
                    "config_promoted": False,
                    "halt_reason": HaltReason.BUDGET_EXCEEDED.value,
                    "events": [event(round_number, 7, error=str(exceeded), promoted=False)],
                }
            metrics = RoundMetrics.from_evaluation(evaluation)
            return {
                "archive_successes": evaluation.archive.blocked + evaluation.archive.refused,
                "archive_trials": evaluation.archive.decided,
                "utility_successes": evaluation.utility.passed,
                "utility_trials": evaluation.utility.total,
                "config_promoted": False,
                "regressions": [],
                "events": [
                    event(
                        round_number,
                        7,
                        promoted=False,
                        reason="the defender proposed no improvement",
                        block_rate=metrics.archive_block_rate,
                    )
                ],
            }

        try:
            evaluation = await self._parts.evaluation.evaluate_full(selected, include_holdout=False)
        except BudgetExceeded as exceeded:
            logger.warning("loop.budget_exceeded", extra={"node": "check_regression"})
            return {
                "config_promoted": False,
                "halt_reason": HaltReason.BUDGET_EXCEEDED.value,
                "events": [event(round_number, 7, error=str(exceeded), promoted=False)],
            }
        after = evaluation.archive.outcome_map
        before = {
            uuid.UUID(key): Outcome(value)
            for key, value in state.get("before_outcomes", {}).items()
        }
        cells = await self._cells_for(list(after))
        regressions = find_regressions(before, after, cells=cells)

        promoted = not regressions
        return {
            "archive_successes": evaluation.archive.blocked + evaluation.archive.refused,
            "archive_trials": evaluation.archive.decided,
            "utility_successes": evaluation.utility.passed,
            "utility_trials": evaluation.utility.total,
            "regressions": list(regressions),
            "config_promoted": promoted,
            "events": [
                event(
                    round_number,
                    7,
                    promoted=promoted,
                    regressions=[str(item.attack_id) for item in regressions],
                )
            ],
        }

    async def generalize(self, state: LoopState) -> LoopState:
        """Step 8: the holdout set. Never fed back to the defender."""
        round_number = int(state["round_number"])
        config = self._config_after(state)
        try:
            holdout = await self._parts.evaluation.evaluate_holdout(config)
        except BudgetExceeded as exceeded:
            logger.warning("loop.budget_exceeded", extra={"node": "generalize"})
            return {
                "holdout_successes": 0,
                "holdout_trials": 0,
                "halt_reason": HaltReason.BUDGET_EXCEEDED.value,
                "events": [event(round_number, 8, error=str(exceeded))],
            }
        return {
            "holdout_successes": holdout.blocked + holdout.refused,
            "holdout_trials": holdout.decided,
            "events": [
                event(round_number, 8, holdout_attacks=holdout.evaluated),
            ],
        }

    async def record(self, state: LoopState) -> LoopState:
        """Step 9: build and persist the RoundReport."""
        round_number = int(state["round_number"])
        report = await self._build_report(state)
        after = self._config_after(state)
        async with self._parts.database.session() as session:
            await RoundRepository(session).record(report)
            # Store D(n) by id, with the config it came from. Phase 8's layer
            # ablation and cross-round comparisons address configs this way.
            await DefenseConfigRepository(session).save(
                after,
                round_number=round_number,
                run_id=uuid.UUID(str(state["run_id"])),
                parent_config_id=state["current_config"].fingerprint(),
                label=f"D({round_number})",
            )

        reports = [*state.get("reports", []), report]
        signals = [
            *state.get("signals", []),
            RoundSignals(
                round_number=round_number,
                mean_novelty=report.mean_novelty,
                new_cells=report.new_cells,
                novelty_rejection_rate=_rejection_rate(report),
                utility_pass_rate=report.utility_pass.rate,
                overfit_gap=report.overfit_gap,
                cost_usd=report.cost_usd,
            ),
        ]
        return {
            "reports": reports,
            "signals": signals,
            "current_config": self._config_after(state),
            "previous_config": state["current_config"],
            "round_cost": str(report.cost_usd),
            "events": [
                event(
                    round_number,
                    9,
                    archive_block_rate=report.archive_block.rate,
                    holdout_block_rate=report.holdout_block.rate,
                    overfit_gap=report.overfit_gap,
                )
            ],
        }

    async def assess(self, state: LoopState) -> LoopState:
        """Step 10: collapse detection. A halted run is a result, not a failure."""
        round_number = int(state["round_number"])
        existing = state.get("halt_reason")
        reason = (
            HaltReason(existing)
            if existing
            else detect(
                state.get("signals", []),
                baseline_utility=float(state.get("baseline_utility", 1.0)),
                budget_usd=self._parts.budget_usd,
            )
        )
        status = RunStatus.HALTED if reason else RunStatus.RUNNING
        return {
            "halt_reason": reason.value if reason else None,
            "status": status.value,
            "events": [event(round_number, 10, halt_reason=reason.value if reason else None)],
        }

    def should_continue(self, state: LoopState) -> str:
        if state.get("halt_reason"):
            return END
        if int(state.get("round_number", 0)) >= int(state.get("rounds_planned", 1)):
            return END
        return "start_round"

    # ------------------------------------------------------------- helpers

    def _config_after(self, state: LoopState) -> DefenseConfig:
        """D(n): the selected config, unless the regression check refused it."""
        if state.get("config_promoted") and state.get("selected_config") is not None:
            selected = state["selected_config"]
            if selected is not None:
                return selected
        return state["current_config"]

    async def _build_report(self, state: LoopState) -> RoundReport:
        coverage = await self._parts.archive.coverage()
        cost = await self._parts.cost_meter.spent(uuid.UUID(str(state["round_id"])))
        after = self._config_after(state)
        report = RoundReport(
            run_id=uuid.UUID(str(state["run_id"])),
            round_number=int(state["round_number"]),
            attacker_mode=str(state.get("attacker_mode", AttackerMode.BLACK_BOX.value)),
            defense_before=state["current_config"].fingerprint(),
            defense_after=after.fingerprint(),
            attacks_generated=int(state.get("attacks_generated", 0)),
            attacks_rejected_novelty=int(state.get("attacks_rejected_novelty", 0)),
            breaches_found=int(state.get("breaches_found", 0)),
            archive_block=Proportion(
                successes=int(state.get("archive_successes", 0)),
                trials=int(state.get("archive_trials", 0)),
            ),
            holdout_block=Proportion(
                successes=int(state.get("holdout_successes", 0)),
                trials=int(state.get("holdout_trials", 0)),
            ),
            utility_pass=Proportion(
                successes=int(state.get("utility_successes", 0)),
                trials=int(state.get("utility_trials", 0)),
            ),
            mean_novelty=float(state.get("mean_novelty", 0.0)),
            cells_occupied=coverage.occupied,
            new_cells=max(0, coverage.occupied - int(state.get("cells_before", 0))),
            regressions=tuple(state.get("regressions", [])),
            config_promoted=bool(state.get("config_promoted", True)),
            cost_usd=cost,
            halt_reason=_halt_reason(state.get("halt_reason")),
            started_at=state.get("round_started_at"),
            ended_at=datetime.now(UTC),
        )
        report.validate_intervals()
        return report

    async def _outcomes_for(self, config: DefenseConfig) -> dict[uuid.UUID, Outcome]:
        """Per-attack outcomes already recorded for a config."""
        async with self._parts.database.session() as session:
            return await AttemptRepository(session).outcomes_by_attack(config.fingerprint())

    async def _outcome_summary(self, config: DefenseConfig) -> OutcomeSummary:
        """What black-box mode may tell the attacker: its own scoreboard."""
        outcomes = await self._outcomes_for(config)
        counts = dict.fromkeys(Outcome, 0)
        for outcome in outcomes.values():
            counts[outcome] += 1
        return OutcomeSummary(
            attempts=len(outcomes),
            breached=counts[Outcome.BREACHED],
            blocked=counts[Outcome.BLOCKED],
            refused=counts[Outcome.REFUSED],
            errors=counts[Outcome.ERROR],
        )

    async def _breaches_this_round(
        self, config: DefenseConfig, attack_ids: list[uuid.UUID]
    ) -> list[uuid.UUID]:
        outcomes = await self._outcomes_for(config)
        return [
            attack_id for attack_id in attack_ids if outcomes.get(attack_id) is Outcome.BREACHED
        ]

    async def _breaches_for_defender(self, config: DefenseConfig) -> list[BreachSummary]:
        """Step 4's input. Holdout is filtered in SQL and never reaches here."""
        async with self._parts.database.session() as session:
            attacks = await AttackRepository(session).get_attacks_for_defender(
                defense_config_id=config.fingerprint()
            )
        return [
            BreachSummary(
                attack_id=row.id,
                cell_key=row.cell_key,
                objective=row.objective.value if row.objective else None,
                technique=row.technique.value if row.technique else None,
                vector=row.vector.value,
                payload=row.payload,
                is_holdout=row.is_holdout,
            )
            for row in attacks
        ]

    async def _cells_for(self, attack_ids: list[uuid.UUID]) -> dict[uuid.UUID, str | None]:
        async with self._parts.database.session() as session:
            repository = AttackRepository(session)
            cells: dict[uuid.UUID, str | None] = {}
            for attack_id in attack_ids:
                row = await repository.get(attack_id)
                if row is not None:
                    cells[attack_id] = row.cell_key
        return cells

    async def _mean_novelty(self, attack_ids: list[uuid.UUID]) -> float:
        if not attack_ids:
            return 0.0
        async with self._parts.database.session() as session:
            repository = AttackRepository(session)
            scores: list[float] = []
            for attack_id in attack_ids:
                row = await repository.get(attack_id)
                if row is not None and row.novelty_score is not None:
                    scores.append(float(row.novelty_score))
        return sum(scores) / len(scores) if scores else 0.0

    # -------------------------------------------------------------- graph

    def build(self, checkpointer: object | None = None) -> CompiledStateGraph[LoopState]:
        graph: StateGraph[LoopState] = StateGraph(LoopState)
        graph.add_node("start_round", self.start_round)
        graph.add_node("attack", self.attack)
        graph.add_node("defend", self.defend)
        graph.add_node("check_regression", self.check_regression)
        graph.add_node("generalize", self.generalize)
        graph.add_node("record", self.record)
        graph.add_node("assess", self.assess)

        graph.add_edge(START, "start_round")
        graph.add_edge("start_round", "attack")
        graph.add_edge("attack", "defend")
        graph.add_edge("defend", "check_regression")
        graph.add_edge("check_regression", "generalize")
        graph.add_edge("generalize", "record")
        graph.add_edge("record", "assess")
        graph.add_conditional_edges("assess", self.should_continue, [END, "start_round"])
        compiled = graph.compile(checkpointer=checkpointer)  # type: ignore[arg-type]
        self._graph = compiled
        return compiled

    @property
    def graph(self) -> CompiledStateGraph[LoopState]:
        if self._graph is None:
            return self.build()
        return self._graph


def _halt_reason(value: str | None) -> HaltReason | None:
    return HaltReason(value) if value else None


def _rejection_rate(report: RoundReport) -> float:
    seen = report.attacks_generated + report.attacks_rejected_novelty
    return report.attacks_rejected_novelty / seen if seen else 0.0


def as_loop_state(result: object) -> LoopState:
    return cast(LoopState, result)


__all__ = ["CoEvolutionLoop", "LoopComponents", "Regression", "as_loop_state"]
