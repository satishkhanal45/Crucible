"""Verification tests 15-22: the defender agent.

Tests 19, 21 and 22 are never-cut properties: the anti-over-blocking objective,
and the two isolation guarantees that keep the holdout set and the canaries out
of anything the defender can see.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

import pytest

from crucible.defender.graph import (
    STATUS_IMPROVED,
    STATUS_NO_IMPROVEMENT,
    Defender,
    cluster_breaches,
)
from crucible.defender.llm import MeteredDefenderLLM, ScriptedDefenderLLM
from crucible.defender.prompts import (
    HoldoutLeak,
    assert_no_holdout,
    build_hypothesis_prompt,
    build_proposal_prompt,
)
from crucible.defender.state import BreachSummary, Cluster, DefenderState, Hypothesis
from crucible.defenses.config import (
    ContextLayer,
    DefenseConfig,
    OutputLayer,
    StructuralLayer,
)
from crucible.evaluation.objective import score
from crucible.evaluation.service import (
    ArchiveEvaluation,
    DefenseEvaluation,
    EvaluationScope,
    UtilityEvaluation,
)
from crucible.services.cost_meter import CostMeter, ModelPrice, price_key
from crucible.target.canary import CanaryClass, CanarySet, set_active_canaries
from tests.fixtures.fake_spend import FakeSpendRepository

HARDENED = DefenseConfig(
    context=ContextLayer(provenance_tags=True, strip_instructions_from_retrieved=True),
    structural=StructuralLayer(
        tool_allowlist=("send_email",), require_user_origin_for_privileged=True
    ),
)
OVER_BLOCKING = DefenseConfig(
    context=ContextLayer(spotlighting="encoding"),
    output=OutputLayer(schema_enforcement=True, citation_verification=True),
    structural=StructuralLayer(require_user_origin_for_privileged=True),
)


def breach(
    cell: str,
    *,
    vector: str = "direct",
    technique: str = "instruction_override",
    holdout: bool = False,
    payload: str = "ignore previous instructions",
) -> BreachSummary:
    return BreachSummary(
        attack_id=uuid.uuid4(),
        cell_key=cell,
        objective=cell.split("|")[0],
        technique=technique,
        vector=vector,
        payload=payload,
        is_holdout=holdout,
    )


def evaluation(
    config: DefenseConfig,
    *,
    block_rate: float,
    utility: float,
    scope: EvaluationScope = EvaluationScope.SCREENING,
) -> DefenseEvaluation:
    """A screening result with the requested rates, built exactly."""
    decided = 10
    blocked = round(block_rate * decided)
    passed = round(utility * 40)
    return DefenseEvaluation(
        config_id=config.fingerprint(),
        scope=scope,
        archive=ArchiveEvaluation(
            scope=scope,
            evaluated=decided,
            archive_size=decided,
            breached=decided - blocked,
            blocked=blocked,
        ),
        utility=UtilityEvaluation(
            total=40, passed=passed, hard_negative_total=12, hard_negative_passed=min(12, passed)
        ),
        cost_usd=Decimal("0.01"),
    )


@dataclass
class FakeScreener:
    """Returns scripted evaluations and records what it was asked to screen."""

    results: dict[str, DefenseEvaluation] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list)
    default_block_rate: float = 0.5
    default_utility: float = 1.0

    async def screen(self, config: DefenseConfig) -> DefenseEvaluation:
        self.seen.append(config.fingerprint())
        existing = self.results.get(config.fingerprint())
        if existing is not None:
            return existing
        return evaluation(config, block_rate=self.default_block_rate, utility=self.default_utility)


def proposal_reply(config: DefenseConfig, rationale: str = "harden the context layer") -> str:
    return json.dumps({"rationale": rationale, "config": config.to_dict()})


def hypothesis_reply(statement: str = "retrieved text is treated as instructions") -> str:
    return json.dumps({"statement": statement, "suggested_layers": ["context", "structural"]})


def base_state(**overrides: object) -> DefenderState:
    state: DefenderState = {
        "round": 1,
        "current_config": DefenseConfig.empty(),
        "breaches": [breach("tool_hijack|direct|instruction_override")],
        "utility_baseline": 1.0,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ----------------------------------------------------------------- test 15


def test_triage_clusters_ten_breaches_across_three_cells() -> None:
    breaches = [
        *[breach("tool_hijack|direct|instruction_override") for _ in range(4)],
        *[
            breach(
                "sysprompt_extraction|indirect_document|context_confusion",
                vector="indirect_document",
                technique="context_confusion",
            )
            for _ in range(3)
        ],
        *[
            breach(
                "scope_violation|direct|authority_impersonation",
                technique="authority_impersonation",
            )
            for _ in range(3)
        ],
    ]

    clusters = cluster_breaches(breaches)

    assert len(breaches) == 10
    assert len(clusters) == 3
    assert sorted(cluster.size for cluster in clusters) == [3, 3, 4]
    assert {cluster.cell_key for cluster in clusters} == {
        "tool_hijack|direct|instruction_override",
        "sysprompt_extraction|indirect_document|context_confusion",
        "scope_violation|direct|authority_impersonation",
    }


def test_triage_separates_mechanisms_inside_one_cell() -> None:
    """Same cell, different delivery: different problems to solve."""
    same_cell = "tool_hijack|direct|instruction_override"
    clusters = cluster_breaches(
        [
            breach(same_cell),
            breach(same_cell, vector="indirect_document"),
        ]
    )

    assert len(clusters) == 2


async def test_triage_node_populates_the_state() -> None:
    defender = Defender(ScriptedDefenderLLM(), FakeScreener())

    result = await defender.triage(
        base_state(breaches=[breach("a|direct|b"), breach("c|direct|d")])
    )

    assert len(result["breach_clusters"]) == 2


# ----------------------------------------------------------------- test 16


async def test_propose_fans_out_to_four_candidates_and_the_reducer_merges_them() -> None:
    configs = [
        DefenseConfig(context=ContextLayer(provenance_tags=True)),
        DefenseConfig(context=ContextLayer(strip_instructions_from_retrieved=True)),
        DefenseConfig(structural=StructuralLayer(require_user_origin_for_privileged=True)),
        HARDENED,
    ]
    llm = ScriptedDefenderLLM([hypothesis_reply(), *[proposal_reply(c) for c in configs]])
    screener = FakeScreener()
    defender = Defender(llm, screener, candidates=4)

    final = await defender.run(base_state())

    assert len(final["candidate_configs"]) == 4, "the reducer must merge every branch"
    assert {proposal.config_id for proposal in final["candidate_configs"]} == {
        config.fingerprint() for config in configs
    }


def test_the_defender_refuses_a_candidate_count_outside_three_to_five() -> None:
    for count in (2, 6):
        with pytest.raises(ValueError, match="three to five"):
            Defender(ScriptedDefenderLLM(), FakeScreener(), candidates=count)


# ----------------------------------------------------------------- test 17


async def test_a_malformed_candidate_never_reaches_evaluate() -> None:
    malformed = json.dumps(
        {"rationale": "invent a field", "config": {"structural": {"run_shell": True}}}
    )
    good = proposal_reply(HARDENED)
    llm = ScriptedDefenderLLM([hypothesis_reply(), malformed, good, malformed, good])
    screener = FakeScreener()
    defender = Defender(llm, screener, candidates=4)

    final = await defender.run(base_state())

    assert len(final["validated"]) == 1, "only the valid, de-duplicated candidate survives"
    assert final["validated"][0].config_id == HARDENED.fingerprint()
    assert screener.seen == [HARDENED.fingerprint()]
    assert any("schema validation failed" in item.reason for item in final["rejected"])


async def test_an_unsafe_candidate_is_rejected_at_validate() -> None:
    """`canary_scan: false` would make every number this config produced a lie."""
    unsafe = DefenseConfig(output=OutputLayer(canary_scan=False))
    llm = ScriptedDefenderLLM(
        [hypothesis_reply(), proposal_reply(unsafe), *[proposal_reply(HARDENED)] * 3]
    )
    screener = FakeScreener()
    defender = Defender(llm, screener, candidates=4)

    final = await defender.run(base_state())

    assert unsafe.fingerprint() not in screener.seen
    assert any("canary_scan" in item.reason for item in final["rejected"])


# ----------------------------------------------------------------- test 18


async def test_select_maximises_the_objective_over_five_candidates() -> None:
    candidates = [
        (DefenseConfig(context=ContextLayer(provenance_tags=True)), 0.90, 0.70),
        (DefenseConfig(context=ContextLayer(strip_instructions_from_retrieved=True)), 0.60, 1.00),
        (
            DefenseConfig(structural=StructuralLayer(require_user_origin_for_privileged=True)),
            0.80,
            1.00,
        ),
        (DefenseConfig(context=ContextLayer(spotlighting="datamarking")), 1.00, 0.50),
        (HARDENED, 0.70, 0.95),
    ]
    screener = FakeScreener(
        results={
            config.fingerprint(): evaluation(config, block_rate=block, utility=utility)
            for config, block, utility in candidates
        }
    )
    llm = ScriptedDefenderLLM(
        [hypothesis_reply(), *[proposal_reply(config) for config, _, _ in candidates]]
    )
    defender = Defender(llm, screener, candidates=5)

    final = await defender.run(base_state())

    expected = max(
        candidates,
        key=lambda row: (
            score(
                archive_block_rate=row[1],
                baseline_utility=1.0,
                config_utility=row[2],
                config_complexity=row[0].complexity,
            ).value
        ),
    )[0]
    assert final["chosen_id"] == expected.fingerprint()
    assert final["status"] == STATUS_IMPROVED
    # The winner is the 0.80 / no-utility-loss candidate, not the 1.00 blocker.
    assert expected.fingerprint() == candidates[2][0].fingerprint()


# ----------------------------------------------------------------- test 19
# *** NEVER-CUT: the anti-over-blocking property ***


def test_a_perfect_blocker_that_costs_utility_loses_to_a_weaker_one_that_does_not() -> None:
    blocks_everything = score(archive_block_rate=1.0, baseline_utility=1.0, config_utility=0.5)
    keeps_utility = score(archive_block_rate=0.8, baseline_utility=1.0, config_utility=1.0)

    assert blocks_everything.utility_loss == pytest.approx(0.5)
    assert keeps_utility.utility_loss == 0.0
    assert keeps_utility.value > blocks_everything.value
    assert blocks_everything.value == pytest.approx(0.0)
    assert keeps_utility.value == pytest.approx(0.8)


async def test_the_defender_picks_the_weaker_blocker_that_keeps_utility() -> None:
    screener = FakeScreener(
        results={
            OVER_BLOCKING.fingerprint(): evaluation(OVER_BLOCKING, block_rate=1.0, utility=0.5),
            HARDENED.fingerprint(): evaluation(HARDENED, block_rate=0.8, utility=1.0),
        }
    )
    llm = ScriptedDefenderLLM(
        [
            hypothesis_reply(),
            proposal_reply(OVER_BLOCKING),
            proposal_reply(HARDENED),
            proposal_reply(OVER_BLOCKING),
        ]
    )
    defender = Defender(llm, screener, candidates=3)

    final = await defender.run(base_state())

    assert final["chosen_id"] == HARDENED.fingerprint()


# ----------------------------------------------------------------- test 20


async def test_no_improvement_retries_to_the_cap_then_keeps_the_current_config() -> None:
    current = HARDENED
    weak = DefenseConfig(context=ContextLayer(provenance_tags=True))
    screener = FakeScreener(
        results={
            current.fingerprint(): evaluation(current, block_rate=0.9, utility=1.0),
            weak.fingerprint(): evaluation(weak, block_rate=0.2, utility=1.0),
        }
    )
    llm = ScriptedDefenderLLM([hypothesis_reply()] + [proposal_reply(weak)] * 24)
    defender = Defender(llm, screener, candidates=3, max_propose_rounds=2)

    final = await defender.run(
        base_state(
            current_config=current,
            eval_results={current.fingerprint(): screener.results[current.fingerprint()]},
        )
    )

    assert final["status"] == STATUS_NO_IMPROVEMENT
    assert final["chosen"] == current, "the round keeps the config it started with"
    assert final["propose_rounds"] == 2, "it retried up to the cap"


# ------------------------------------------------------------ tests 21, 22


SENTINEL = "HOLDOUT-SENTINEL-8d41c2b7"


def test_a_holdout_attack_never_reaches_a_defender_prompt() -> None:
    """Never-cut: showing the defender a holdout attack destroys the only honest
    generalization number in the project."""
    breaches = [
        breach("tool_hijack|direct|instruction_override"),
        breach("tool_hijack|direct|instruction_override", holdout=True, payload=SENTINEL),
    ]

    with pytest.raises(HoldoutLeak) as raised:
        assert_no_holdout(breaches)
    assert SENTINEL not in str(raised.value), "even the error must not quote the payload"

    cluster = Cluster(
        cell_key="tool_hijack|direct|instruction_override",
        mechanism="direct/instruction_override",
        attack_ids=(breaches[0].attack_id,),
    )
    prompt = build_hypothesis_prompt(cluster, [breaches[0]], DefenseConfig.empty())
    assert SENTINEL not in prompt


async def test_the_defender_refuses_to_run_on_holdout_breaches() -> None:
    defender = Defender(ScriptedDefenderLLM(), FakeScreener())

    with pytest.raises(HoldoutLeak):
        await defender.triage(
            base_state(breaches=[breach("a|direct|b", holdout=True, payload=SENTINEL)])
        )


def test_no_defender_prompt_contains_a_canary() -> None:
    """Never-cut: asserted in the builder, not merely in this test."""
    canaries = CanarySet.mint()
    set_active_canaries(canaries)
    try:
        cluster = Cluster(
            cell_key="tool_hijack|direct|instruction_override",
            mechanism="direct/instruction_override",
            attack_ids=(),
        )
        hypothesis_prompt = build_hypothesis_prompt(cluster, [], DefenseConfig.empty())
        proposal_prompt = build_proposal_prompt(
            HARDENED,
            [Hypothesis(cluster_key="c", mechanism="m", statement="s")],
            candidate_index=0,
            utility_baseline=1.0,
        )
        for prompt in (hypothesis_prompt, proposal_prompt):
            for canary in canaries:
                assert canary.reveal() not in prompt
        assert "CRUCIBLE-" not in hypothesis_prompt + proposal_prompt
    finally:
        set_active_canaries(None)


def test_a_prompt_carrying_a_canary_is_refused_by_the_builder() -> None:
    from crucible.target.canary import CanaryLeak

    canaries = CanarySet.mint()
    set_active_canaries(canaries)
    try:
        leaking = breach(
            "tool_hijack|direct|instruction_override",
            payload=f"leak {canaries.reveal(CanaryClass.SYSPROMPT)}",
        )
        cluster = Cluster(
            cell_key="tool_hijack|direct|instruction_override",
            mechanism="direct/instruction_override",
            attack_ids=(leaking.attack_id,),
        )
        with pytest.raises(CanaryLeak):
            build_hypothesis_prompt(cluster, [leaking], DefenseConfig.empty())
    finally:
        set_active_canaries(None)


# ---------------------------------------------------------------- metering


async def test_every_defender_call_is_metered() -> None:
    repository = FakeSpendRepository()
    meter = CostMeter(
        repository,
        Decimal("1.00"),
        pricing={price_key("stub", "scripted-defender"): ModelPrice(Decimal("1"), Decimal("1"))},
    )
    llm = MeteredDefenderLLM(
        ScriptedDefenderLLM([hypothesis_reply(), *[proposal_reply(HARDENED)] * 3]), meter
    )
    defender = Defender(llm, FakeScreener(), candidates=3)

    await defender.run(base_state())

    assert len(repository.records) == 4, "one hypothesis call plus three proposals"
    assert all(record.provider == "stub" for record in repository.records)
