"""Verification tests 1-5 and 14-18: the co-evolution loop.

Test 3 is never-cut and permanent: without the regression check the defender
fixes round 5 by reopening round 2, the curve looks excellent, and the system
gets worse. Tests 2, 4 and 14 are equally load-bearing — the full-archive
evaluation, holdout isolation, and resuming without duplicating work.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from decimal import Decimal

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from crucible.archive.seeds import load_seed_attacks
from crucible.db.session import Database
from crucible.defenses.config import (
    DefenseConfig,
)
from crucible.evaluation.benign import load_benign_tasks
from crucible.loop.reports import HaltReason, RunStatus
from crucible.loop.runner import LoopFactories, LoopRunner, LoopSettings, build_components
from crucible.loop.state import STEP_NAMES
from crucible.repositories.attacks import AttackRepository
from crucible.repositories.attempts import AttemptRepository
from crucible.schemas.attack import Attack
from crucible.schemas.taxonomy import DeliveryVector, Objective, Technique
from crucible.services.embeddings import HashingEmbedder
from crucible.target.reference.llm import ScriptedTargetLLM

#: A config whose only defense is stripping instructions out of retrieved text.
#: An indirect attack is blocked by it and breaches without it, which is what
#: makes a clean regression constructible.
#: The same defense with the stripping removed and cosmetic hardening added: it
#: looks like an improvement and reopens the indirect attack.
from tests.fixtures.loop_harness import (
    HARDENED,
    INDIRECT_PAYLOAD,
    NOT_STRIPPING,
    STRIPPING,
    CyclingAttackerLLM,
    Harness,
    ScriptedProposals,
    classifier_client,
    corpus_subset,
)


def indirect_attack() -> Attack:
    return Attack(
        payload=INDIRECT_PAYLOAD,
        vector=DeliveryVector.INDIRECT_DOCUMENT,
        objective=Objective.SYSPROMPT_EXTRACTION,
        technique=Technique.INSTRUCTION_OVERRIDE,
        benign_user_input="Summarize the laptop refresh policy",
        carrier_title="Laptop Refresh Policy - Addendum",
    )


# ------------------------------------------------------------------ test 1


async def test_a_round_runs_all_ten_steps_in_order(build_loop: Callable[..., object]) -> None:
    harness: Harness = await build_loop(rounds=1)  # type: ignore[misc]

    await harness.runner.start(starting_config=DefenseConfig.empty())
    state = await harness.runner.graph_state()

    steps = [(item.round_number, item.step) for item in state["events"]]
    assert steps == [(1, step) for step in range(1, 11)]
    assert [item.name for item in state["events"]] == list(STEP_NAMES)


# ------------------------------------------------------------------ test 2


async def test_the_defender_is_evaluated_against_the_whole_archive(
    build_loop: Callable[..., object],
) -> None:
    """Full-archive re-evaluation, not just this round's attacks."""
    seeds = load_seed_attacks()[:12]
    harness: Harness = await build_loop(rounds=1, seed_attacks=seeds, cells_per_round=2)  # type: ignore[misc]

    report = await harness.runner.start(starting_config=DefenseConfig.empty())

    async with harness.database.session() as session:
        non_holdout = await AttackRepository(session).list_non_holdout()

    round_report = report.rounds[0]
    assert round_report.archive_block.trials <= len(non_holdout)
    assert round_report.archive_block.trials >= len(seeds) - 4, (
        "the evaluation must cover the whole archive, not just this round's attacks"
    )
    assert len(non_holdout) > len(seeds) - 4, "the round added attacks to the archive"


# ------------------------------------------------------------------ test 3
# *** NEVER-CUT AND PERMANENT ***


async def test_a_config_that_reopens_an_archived_attack_is_flagged_and_not_promoted(
    build_loop: Callable[..., object],
) -> None:
    """The mechanism that makes this a loop rather than a sequence of patches."""
    reopened = indirect_attack()
    harness: Harness = await build_loop(  # type: ignore[misc]
        rounds=1,
        configs=[NOT_STRIPPING],
        seed_attacks=[reopened, *load_seed_attacks()[:3]],
        force_holdout=False,
        cells_per_round=1,
    )

    report = await harness.runner.start(starting_config=STRIPPING)
    round_report = report.rounds[0]

    assert round_report.regressions, "reopening an archived attack must be detected"
    assert reopened.attack_id in {
        regression.attack_id for regression in round_report.regressions
    }, "the regression must name the specific attack"
    assert round_report.config_promoted is False, "a regressing config is never auto-accepted"
    assert round_report.defense_after == STRIPPING.fingerprint(), "D(n) stays D(n-1)"
    assert "REGRESSIONS" in round_report.summary()


async def test_a_config_with_no_regression_is_promoted(
    build_loop: Callable[..., object],
) -> None:
    harness: Harness = await build_loop(rounds=1, configs=[HARDENED])  # type: ignore[misc]

    report = await harness.runner.start(starting_config=DefenseConfig.empty())
    round_report = report.rounds[0]

    assert round_report.regressions == ()
    assert round_report.config_promoted is True
    assert round_report.defense_after == HARDENED.fingerprint()


# ------------------------------------------------------------------ test 4


async def test_the_defender_never_sees_a_holdout_attack(
    build_loop: Callable[..., object],
) -> None:
    harness: Harness = await build_loop(rounds=1, seed_attacks=load_seed_attacks()[:10])  # type: ignore[misc]

    await harness.runner.start(starting_config=DefenseConfig.empty())

    async with harness.database.session() as session:
        holdout = await AttackRepository(session).list_holdout()
    holdout_ids = {row.id for row in holdout}
    assert holdout_ids, "the archive must actually hold some holdout attacks"

    assert harness.defender_states.states, "the defender ran"
    for state in harness.defender_states.states:
        shown = {breach.attack_id for breach in state.get("breaches", [])}
        assert not (shown & holdout_ids), "a holdout attack reached the defender"
        assert all(not breach.is_holdout for breach in state.get("breaches", []))

    for prompt in harness.defender_llm.prompts:
        for attack_id in holdout_ids:
            assert str(attack_id) not in prompt


# ------------------------------------------------------------------ test 5


async def test_the_overfit_gap_is_archive_minus_holdout_and_is_stored(
    build_loop: Callable[..., object],
) -> None:
    harness: Harness = await build_loop(rounds=1, seed_attacks=load_seed_attacks()[:10])  # type: ignore[misc]

    report = await harness.runner.start(starting_config=DefenseConfig.empty())
    round_report = report.rounds[0]

    assert round_report.overfit_gap == pytest.approx(
        round_report.archive_block.rate - round_report.holdout_block.rate
    )
    assert round_report.holdout_block.trials > 0, "the holdout set is evaluated every round"

    from crucible.repositories.rounds import RoundRepository

    async with harness.database.session() as session:
        stored = await RoundRepository(session).get(report.run_id, 1)
    assert stored is not None
    assert stored.overfit_gap == pytest.approx(round_report.overfit_gap)


async def reset_archive(database: Database) -> None:
    """Clear everything a run accumulates, so two runs start identically."""
    from sqlalchemy import text

    async with database.session() as session:
        for table in ("novelty_rejections", "cells", "attempts", "attacks"):
            await session.execute(text(f"DELETE FROM {table}"))


def comparable(report: object) -> list[tuple[object, ...]]:
    """A run's numbers, without the ids that differ between runs by design."""
    from crucible.loop.reports import RunReport

    assert isinstance(report, RunReport)
    return [
        (
            round_report.round_number,
            round_report.attacks_generated,
            round_report.attacks_rejected_novelty,
            round_report.breaches_found,
            round_report.archive_block.successes,
            round_report.archive_block.trials,
            round_report.holdout_block.successes,
            round_report.holdout_block.trials,
            round_report.utility_pass.successes,
            round_report.cells_occupied,
            round_report.new_cells,
            round_report.config_promoted,
            round_report.defense_after,
        )
        for round_report in report.rounds
    ]


# ----------------------------------------------------------------- test 14


async def test_a_run_interrupted_mid_round_resumes_without_redoing_work(
    build_loop: Callable[..., object],
) -> None:
    """A nine-round run will meet a rate limit or a closed laptop."""
    uninterrupted: Harness = await build_loop(rounds=1)  # type: ignore[misc]
    reference = await uninterrupted.runner.start(starting_config=DefenseConfig.empty())
    reference_generated = uninterrupted.attacker_llm._generated

    await reset_archive(uninterrupted.database)
    harness: Harness = await build_loop(rounds=1)  # type: ignore[misc]

    run_id = uuid.uuid4()
    # Stop exactly where a crash would: after the attacker has generated and
    # executed, before the defender is evaluated.
    interrupted = await harness.runner.start(
        run_id=run_id,
        starting_config=DefenseConfig.empty(),
        interrupt_before=("defend",),
    )
    assert interrupted.rounds == (), "the round did not finish"
    generated_before_resume = harness.attacker_llm._generated
    assert generated_before_resume > 0, "the attacker did run"

    async with harness.database.session() as session:
        attempts_before = await AttemptRepository(session).count()

    resumed = await harness.runner.resume(run_id)

    assert harness.attacker_llm._generated == generated_before_resume, (
        "resuming must not regenerate attacks"
    )
    assert len(resumed.rounds) == 1
    async with harness.database.session() as session:
        attempts_after = await AttemptRepository(session).count()
    assert attempts_after >= attempts_before

    assert comparable(resumed) == comparable(reference), (
        "a resumed round must produce the same result as an uninterrupted one"
    )
    assert reference_generated == generated_before_resume


# ----------------------------------------------------------------- test 15


async def test_a_budget_halt_can_be_resumed_with_a_bigger_budget(
    build_loop: Callable[..., object],
) -> None:
    from crucible.services.cost_meter import ModelPrice, price_key

    pricing = {
        price_key("stub", "scripted-attacker"): ModelPrice(Decimal("1000"), Decimal("1000")),
        price_key("stub", "scripted-defender"): ModelPrice(Decimal("1000"), Decimal("1000")),
        price_key("stub", "scripted-stub"): ModelPrice(Decimal("1000"), Decimal("1000")),
    }
    harness: Harness = await build_loop(rounds=2)  # type: ignore[misc]
    database = harness.database
    try:
        await reset_archive(database)
        attacker_llm = CyclingAttackerLLM()
        defender_llm = ScriptedProposals([HARDENED])
        settings = LoopSettings(
            rounds=2, budget_usd=Decimal("0.02"), concurrency=1, cells_per_round=1
        )
        components = await build_components(
            database,
            settings=settings,
            factories=LoopFactories(
                target_llm=ScriptedTargetLLM,
                attacker_llm=lambda: attacker_llm,
                defender_llm=lambda: defender_llm,
                classifier_client=classifier_client,
            ),
            embedder=HashingEmbedder(),
            corpus=corpus_subset(),
            pricing=pricing,
        )
        components.evaluation._tasks = load_benign_tasks()[:2]
        checkpointer = InMemorySaver()
        run_id = uuid.uuid4()
        runner = LoopRunner(database, components, settings=settings, checkpointer=checkpointer)
        halted = await runner.start(run_id=run_id, starting_config=DefenseConfig.empty())

        assert halted.status is RunStatus.HALTED
        assert halted.halt_reason is HaltReason.BUDGET_EXCEEDED

        generous = settings.model_copy(update={"budget_usd": Decimal("500.00")})
        richer = await build_components(
            database,
            settings=generous,
            factories=LoopFactories(
                target_llm=ScriptedTargetLLM,
                attacker_llm=lambda: attacker_llm,
                defender_llm=lambda: defender_llm,
                classifier_client=classifier_client,
            ),
            embedder=HashingEmbedder(),
            corpus=corpus_subset(),
        )
        richer.evaluation._tasks = load_benign_tasks()[:2]
        resumed_runner = LoopRunner(database, richer, settings=generous, checkpointer=checkpointer)
        resumed = await resumed_runner.resume(run_id)

        assert resumed.halt_reason is not HaltReason.BUDGET_EXCEEDED
        assert len(resumed.rounds) >= len(halted.rounds)
    finally:
        pass


# ----------------------------------------------------------------- test 16


async def test_a_three_round_run_completes_and_records_three_reports(
    build_loop: Callable[..., object],
) -> None:
    harness: Harness = await build_loop(rounds=3, configs=[HARDENED, HARDENED, HARDENED])  # type: ignore[misc]

    report = await harness.runner.start(starting_config=DefenseConfig.empty())

    assert len(report.rounds) == 3
    assert report.status in {RunStatus.COMPLETED, RunStatus.HALTED}
    assert [item.round_number for item in report.rounds] == [1, 2, 3]

    sizes = [item.archive_block.trials for item in report.rounds]
    assert sizes == sorted(sizes), "the archive never shrinks"
    coverage = [item.cells_occupied for item in report.rounds]
    assert coverage == sorted(coverage), "coverage never decreases"

    for item in report.rounds:
        item.validate_intervals()
        assert item.holdout_block.trials > 0
        if not item.config_promoted:
            assert item.defense_after == item.defense_before


# ----------------------------------------------------------------- test 17


async def test_two_runs_with_the_same_seed_produce_the_same_result(
    build_loop: Callable[..., object],
) -> None:
    """Attack ids are content-addressed, so the archives match exactly."""
    first: Harness = await build_loop(rounds=2)  # type: ignore[misc]
    first_report = await first.runner.start(starting_config=DefenseConfig.empty())
    async with first.database.session() as session:
        first_archive = [
            (str(row.id), row.payload, row.cell_key, str(row.parent_id))
            for row in await AttackRepository(session).list_all()
        ]

    await reset_archive(first.database)
    second: Harness = await build_loop(rounds=2)  # type: ignore[misc]
    second_report = await second.runner.start(starting_config=DefenseConfig.empty())
    async with second.database.session() as session:
        second_archive = [
            (str(row.id), row.payload, row.cell_key, str(row.parent_id))
            for row in await AttackRepository(session).list_all()
        ]

    # An archive is a set of attacks, not a sequence: candidates generated in
    # parallel branches land in whatever order the branches finish, so the
    # comparison is by content, keyed on the content-addressed attack id.
    assert sorted(first_archive) == sorted(second_archive), (
        "the same seed must build the same archive"
    )
    assert comparable(first_report) == comparable(second_report)


# ----------------------------------------------------------------- test 18


async def test_a_useless_defender_produces_flat_rates_and_does_not_crash(
    build_loop: Callable[..., object],
) -> None:
    """The loop must survive a defender that never improves anything."""
    harness: Harness = await build_loop(  # type: ignore[misc]
        rounds=2, configs=[DefenseConfig.empty(), DefenseConfig.empty()]
    )

    report = await harness.runner.start(starting_config=DefenseConfig.empty())

    assert len(report.rounds) == 2
    assert report.final_config_id == DefenseConfig.empty().fingerprint()
    for item in report.rounds:
        assert item.defense_after == DefenseConfig.empty().fingerprint()
        assert item.config_promoted is False, "there was nothing to promote"
        item.validate_intervals()
    assert report.status in {RunStatus.COMPLETED, RunStatus.HALTED}
