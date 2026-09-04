"""The defender agent.

    triage -> hypothesize -> propose (fan-out) -> validate -> evaluate -> select
                                  ^                                        |
                                  +--------- no candidate improved --------+

`triage` clusters breaches deterministically; the model is only asked *why* a
cluster got through and *what configuration* might answer it. Everything the
model returns is validated against `DefenseConfig` before it can be evaluated,
so a malformed proposal costs nothing but a parse.

Selection uses the objective from docs/spec.md section 12, with the never-cut
2.0 weight on utility loss: a candidate that blocks everything loses to one that
blocks less and keeps the benign tasks working.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import ValidationError

from crucible.defender.llm import DefenderLLM
from crucible.defender.prompts import (
    assert_no_holdout,
    build_hypothesis_prompt,
    build_proposal_prompt,
)
from crucible.defender.state import (
    DEFAULT_CANDIDATES,
    MAX_PROPOSE_ROUNDS,
    BreachSummary,
    CandidateProposal,
    Cluster,
    DefenderState,
    Hypothesis,
    RejectedProposal,
)
from crucible.defenses.config import DefenseConfig, UnsafeDefense
from crucible.evaluation.objective import UTILITY_WEIGHT, ObjectiveScore, score
from crucible.evaluation.service import DefenseEvaluation
from crucible.logging import get_logger
from crucible.services.retry import ProviderError

logger = get_logger(__name__)

_JSON = re.compile(r"\{.*\}", re.DOTALL)


class CandidateScreener(Protocol):
    """What the defender needs from the evaluation service.

    Narrow on purpose: screening a candidate is the only thing the agent may do
    with the evaluation layer, and a screening number can never be recorded as a
    round metric.
    """

    async def screen(self, config: DefenseConfig) -> DefenseEvaluation: ...


STATUS_IMPROVED = "improved"
STATUS_NO_IMPROVEMENT = "no_improvement"
#: Every proposal branch failed, so there was nothing to score. Named, because a
#: round that produced no candidate at all is a different fact from a round whose
#: candidates were all worse than the incumbent, and the two must not be read as
#: the same result in the findings.
STATUS_NO_CANDIDATES = "no_candidates"


def mechanism_of(breach: BreachSummary) -> str:
    """How a breach got in, independent of its wording."""
    return f"{breach.vector}/{breach.technique or 'unclassified'}"


def cluster_breaches(breaches: Sequence[BreachSummary]) -> list[Cluster]:
    """Group breaches by cell and mechanism. Deterministic; no model involved."""
    grouped: dict[tuple[str, str], list[BreachSummary]] = {}
    for breach in breaches:
        key = (breach.cell_key or "unclassified", mechanism_of(breach))
        grouped.setdefault(key, []).append(breach)
    return [
        Cluster(
            cell_key=cell_key,
            mechanism=mechanism,
            attack_ids=tuple(item.attack_id for item in members),
        )
        for (cell_key, mechanism), members in sorted(grouped.items())
    ]


#: How much of a model reply to carry into a log line or a rejection record.
#: Long enough to see the shape of the JSON and the offending key.
RAW_REPLY_CHARS = 500


def field_errors(error: Exception) -> list[str]:
    """`field: message` for each pydantic error, or one entry for a TypeError."""
    if not isinstance(error, ValidationError):
        return [f"<root>: {error}"]
    return [
        f"{'.'.join(str(part) for part in detail['loc']) or '<root>'}: {detail['msg']}"
        for detail in error.errors()
    ]


def describe_validation_error(error: Exception) -> str:
    """A one-line summary naming the fields, for a log line and a report."""
    return "; ".join(field_errors(error))


def parse_json_object(text: str) -> dict[str, object] | None:
    match = _JSON.search(text or "")
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


class Defender:
    """Builds and runs the defender graph for one round."""

    def __init__(
        self,
        llm: DefenderLLM,
        evaluation: CandidateScreener,
        *,
        candidates: int = DEFAULT_CANDIDATES,
        max_propose_rounds: int = MAX_PROPOSE_ROUNDS,
        utility_weight: float = UTILITY_WEIGHT,
    ) -> None:
        if not 3 <= candidates <= 5:
            raise ValueError("the defender proposes three to five candidates per round")
        self._llm = llm
        self._evaluation = evaluation
        self._candidates = candidates
        self._max_rounds = max_propose_rounds
        # 2.0 everywhere except `experiments/ablation_utility.yaml`, which is
        # the one place the weight may be moved and is labelled for it.
        self._utility_weight = utility_weight
        self._graph = self._build()

    @property
    def llm(self) -> DefenderLLM:
        return self._llm

    # ------------------------------------------------------------ the nodes

    async def triage(self, state: DefenderState) -> DefenderState:
        breaches = state.get("breaches", [])
        assert_no_holdout(breaches)
        clusters = cluster_breaches(breaches)
        logger.info(
            "defender.triage",
            extra={"breaches": len(breaches), "clusters": len(clusters)},
        )
        return {"breach_clusters": clusters}

    async def hypothesize(self, state: DefenderState) -> DefenderState:
        config = state["current_config"]
        breaches = state.get("breaches", [])
        hypotheses: list[Hypothesis] = []

        for cluster in state.get("breach_clusters", []):
            try:
                reply = await self._llm.complete(build_hypothesis_prompt(cluster, breaches, config))
            except ProviderError as error:
                # One cluster's hypothesis is not worth the run. Fall through to
                # the same default statement an unparseable reply produces.
                logger.warning(
                    "defender.hypothesis_failed",
                    extra={"cluster": cluster.cell_key, "error": str(error)},
                )
                hypotheses.append(
                    Hypothesis(
                        cluster_key=cluster.cell_key,
                        mechanism=cluster.mechanism,
                        statement=f"{cluster.mechanism} is unhandled",
                    )
                )
                continue
            payload = parse_json_object(reply.text) or {}
            statement = str(payload.get("statement") or f"{cluster.mechanism} is unhandled")
            layers = payload.get("suggested_layers")
            hypotheses.append(
                Hypothesis(
                    cluster_key=cluster.cell_key,
                    mechanism=cluster.mechanism,
                    statement=statement,
                    suggested_layers=tuple(str(item) for item in layers if isinstance(layers, list))
                    if isinstance(layers, list)
                    else (),
                )
            )
        return {"hypotheses": hypotheses}

    async def propose(self, state: DefenderState) -> DefenderState:
        """A join point. The work happens in the parallel `propose_one` branches."""
        del state
        return {}

    def fan_out(self, state: DefenderState) -> list[Send]:
        """Parallel proposal: one `propose_one` per candidate."""
        return [
            Send("propose_one", {**state, "candidate_index": index})
            for index in range(self._candidates)
        ]

    async def propose_one(self, state: DefenderState) -> DefenderState:
        index = int(state.get("candidate_index", 0))
        prompt = build_proposal_prompt(
            state["current_config"],
            state.get("hypotheses", []),
            candidate_index=index,
            utility_baseline=state.get("utility_baseline", 1.0),
        )
        try:
            reply = await self._llm.complete(prompt)
        except ProviderError as error:
            # The branch fails; the round does not. A timeout that has already
            # exhausted its retries must cost one candidate, not the whole
            # experiment — the same discipline `BudgetExceeded` gets.
            logger.warning(
                "defender.proposal_failed",
                extra={"candidate_index": index, "error": str(error)},
            )
            return {"rejected": [RejectedProposal(raw="", reason=f"provider call failed: {error}")]}
        payload = parse_json_object(reply.text)

        if payload is None or "config" not in payload:
            reason = (
                "reply contained no JSON object"
                if payload is None
                else f"JSON object has no 'config' key (keys: {sorted(payload)})"
            )
            # WARNING, with the reply itself: a run that rejected every candidate
            # for six rounds and logged no reason is a run that cannot be
            # diagnosed afterwards.
            logger.warning(
                "defender.proposal_rejected",
                extra={
                    "candidate_index": index,
                    "stage": "parse",
                    "reason": reason,
                    "raw": reply.text[:RAW_REPLY_CHARS],
                },
            )
            return {"rejected": [RejectedProposal(raw=reply.text[:RAW_REPLY_CHARS], reason=reason)]}
        return self._materialise(payload, candidate_index=index)

    def _materialise(
        self, payload: dict[str, object], *, candidate_index: int = 0
    ) -> DefenderState:
        """Turn a model reply into a proposal, or into a rejection."""
        try:
            config = DefenseConfig.model_validate(payload["config"])
        except (ValidationError, TypeError) as error:
            raw = json.dumps(payload.get("config"))[:RAW_REPLY_CHARS]
            logger.warning(
                "defender.proposal_rejected",
                extra={
                    "candidate_index": candidate_index,
                    "stage": "schema",
                    "reason": describe_validation_error(error),
                    "fields": field_errors(error),
                    "raw": raw,
                },
            )
            return {
                "rejected": [
                    RejectedProposal(
                        raw=raw,
                        reason=f"schema validation failed: {describe_validation_error(error)}",
                    )
                ]
            }
        return {
            "candidate_configs": [
                CandidateProposal(config=config, rationale=str(payload.get("rationale", "")))
            ]
        }

    async def validate(self, state: DefenderState) -> DefenderState:
        """Schema and safety. A malformed candidate never reaches evaluate."""
        kept: list[CandidateProposal] = []
        rejected: list[RejectedProposal] = []
        seen: set[str] = set()

        for proposal in state.get("candidate_configs", []):
            try:
                proposal.config.assert_production_safe()
            except UnsafeDefense as error:
                logger.warning(
                    "defender.proposal_rejected",
                    extra={
                        "stage": "safety",
                        "reason": str(error),
                        "raw": json.dumps(proposal.config.to_dict())[:RAW_REPLY_CHARS],
                    },
                )
                rejected.append(RejectedProposal(raw=proposal.config_id, reason=str(error)))
                continue
            if proposal.config_id in seen:
                continue
            seen.add(proposal.config_id)
            kept.append(proposal)

        # Every reason, at WARNING, when nothing survived. `kept: 0` with no
        # explanation is exactly what made six rounds of failure undiagnosable.
        carried = list(state.get("rejected", []))
        if not kept:
            for item in carried + rejected:
                logger.warning(
                    "defender.candidate_rejected",
                    extra={"reason": item.reason, "raw": item.raw[:RAW_REPLY_CHARS]},
                )
        logger.info(
            "defender.validate",
            extra={"kept": len(kept), "rejected": len(rejected) + len(carried)},
        )
        return {"validated": kept, "rejected": rejected}

    async def evaluate(self, state: DefenderState) -> DefenderState:
        """Screen every surviving candidate. Screening numbers are never recorded."""
        results: dict[str, DefenseEvaluation] = dict(state.get("eval_results", {}))
        for proposal in state.get("validated", []):
            if proposal.config_id in results:
                continue
            results[proposal.config_id] = await self._evaluation.screen(proposal.config)
        return {"eval_results": results}

    async def select(self, state: DefenderState) -> DefenderState:
        """Pick by the objective, or report that nothing improved."""
        baseline = state.get("utility_baseline", 1.0)
        current = state["current_config"]
        results = state.get("eval_results", {})
        candidates = state.get("validated", [])

        current_result = results.get(current.fingerprint())
        current_score = (
            _score_of(current, current_result, baseline, self._utility_weight).value
            if current_result is not None
            else 0.0
        )

        scores: dict[str, float] = dict(state.get("scores", {}))
        best: tuple[float, CandidateProposal] | None = None
        for proposal in candidates:
            result = results.get(proposal.config_id)
            if result is None:
                continue
            value = _score_of(proposal.config, result, baseline, self._utility_weight).value
            scores[proposal.config_id] = value
            if best is None or value > best[0]:
                best = (value, proposal)

        rounds = int(state.get("propose_rounds", 0)) + 1
        if best is None and not candidates:
            # Nothing survived to be scored: every branch failed. The incumbent
            # stands, and the status says why rather than implying the
            # candidates were merely no better.
            logger.warning(
                "defender.no_candidates",
                extra={"rejected": len(state.get("rejected", [])), "round": rounds},
            )
            return {
                "chosen": None,
                "chosen_id": None,
                "scores": scores,
                "status": STATUS_NO_CANDIDATES,
                "propose_rounds": rounds,
            }
        if best is not None and best[0] > current_score:
            logger.info(
                "defender.selected",
                extra={"config_id": best[1].config_id, "score": round(best[0], 4)},
            )
            return {
                "chosen": best[1].config,
                "chosen_id": best[1].config_id,
                "scores": scores,
                "status": STATUS_IMPROVED,
                "propose_rounds": rounds,
            }

        return {
            "chosen": None,
            "chosen_id": None,
            "scores": scores,
            "status": STATUS_NO_IMPROVEMENT,
            "propose_rounds": rounds,
        }

    def after_select(self, state: DefenderState) -> str:
        """Retry the proposal loop, up to the cap, when nothing improved.

        A round with no candidates at all is retried too: the usual cause is a
        provider that was briefly unavailable, and the cap still bounds it.
        """
        if state.get("status") == STATUS_IMPROVED:
            return END
        if int(state.get("propose_rounds", 0)) >= self._max_rounds:
            return END
        return "propose"

    # ------------------------------------------------------------ the graph

    def _build(self) -> CompiledStateGraph[DefenderState]:
        graph: StateGraph[DefenderState] = StateGraph(DefenderState)
        graph.add_node("triage", self.triage)
        graph.add_node("hypothesize", self.hypothesize)
        graph.add_node("propose", self.propose)
        graph.add_node("propose_one", self.propose_one)
        graph.add_node("validate", self.validate)
        graph.add_node("evaluate", self.evaluate)
        graph.add_node("select", self.select)

        graph.add_edge(START, "triage")
        graph.add_edge("triage", "hypothesize")
        graph.add_edge("hypothesize", "propose")
        graph.add_conditional_edges("propose", self.fan_out, ["propose_one"])
        graph.add_edge("propose_one", "validate")
        graph.add_edge("validate", "evaluate")
        graph.add_edge("evaluate", "select")
        graph.add_conditional_edges("select", self.after_select, [END, "propose"])
        return graph.compile()

    async def run(self, state: DefenderState) -> DefenderState:
        """Run one defender round to completion."""
        initial: DefenderState = {
            "candidate_configs": [],
            "rejected": [],
            "eval_results": {},
            "scores": {},
            "propose_rounds": 0,
            "status": STATUS_NO_IMPROVEMENT,
            **state,
        }
        result = await self._graph.ainvoke(initial)
        final = cast(DefenderState, result)
        if final.get("chosen") is None:
            final["chosen"] = state["current_config"]
        return final


def _score_of(
    config: DefenseConfig,
    evaluation: DefenseEvaluation,
    baseline: float,
    weight: float = UTILITY_WEIGHT,
) -> ObjectiveScore:
    return score(
        archive_block_rate=evaluation.archive_block_rate,
        baseline_utility=baseline,
        config_utility=evaluation.utility_pass_rate,
        mean_latency_ms=evaluation.mean_latency_ms,
        config_complexity=config.complexity,
        weight=weight,
    )
