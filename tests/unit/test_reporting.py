"""Verification tests 1-7: report generation, redaction, diffs, lineage, charts.

Tests 5, 10, 11, 12, 13 and 14 of project_context.md Phase 7 are **skipped, with
reason**: they cover the React dashboard, the Playwright end-to-end journeys and
the axe-core accessibility sweep, all of which cut B1 removes from this build.
They are not xfailed, because nothing here is expected to fail — the dashboard
is absent by decision. Deferred item D1 restores it, and it depends only on the
report JSON schema, which is stable.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from crucible.defenses.config import (
    ContextLayer,
    DefenseConfig,
    HeuristicRule,
    InputLayer,
    OutputLayer,
    PromptLayer,
    RoleReassertion,
    Spotlighting,
    StructuralLayer,
)
from crucible.loop.reports import HaltReason, Regression, RoundReport, RunReport, RunStatus
from crucible.loop.statistics import Proportion
from crucible.reporting.charts import coverage_grid, coverage_strip, three_curves
from crucible.reporting.data import GeneralAttack, ReportData
from crucible.reporting.diff import ChangeKind, diff_configs
from crucible.reporting.lineage import build_lineage, render_lineage
from crucible.reporting.markdown import render_round_report, render_run_report
from crucible.reporting.redaction import payload_hash, redact_payload, redact_trace
from crucible.schemas.archive import ArchivedAttack, CellRecord
from crucible.schemas.taxonomy import DeliveryVector, Objective, Technique

RUN_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
PAYLOAD_SENTINEL = "ZEBRA-PAYLOAD-SENTINEL-4821"
CANARY_SENTINEL = "CRUCIBLE-SYSPROMPT-0123456789abcdef0123456789abcdef"
SYSTEM_PROMPT_SENTINEL = "You are the Northwind IT Assistant, the internal help assistant"

EMPTY = DefenseConfig.empty()
HARDENED = DefenseConfig(
    input=InputLayer(
        heuristic_rules=(
            HeuristicRule(
                name="instructions_in_retrieved",
                pattern_class="instruction_like",
                applies_to=("retrieved_context",),
                action="strip",
            ),
        )
    ),
    context=ContextLayer(
        spotlighting=Spotlighting.DELIMITER, strip_instructions_from_retrieved=True
    ),
    prompt=PromptLayer(precedence_statement=True, role_reassertion=RoleReassertion.BOTH),
    output=OutputLayer(canary_scan=True, citation_verification=True),
    structural=StructuralLayer(
        tool_allowlist=("send_email",), require_user_origin_for_privileged=True
    ),
)


def attack(
    *,
    attack_id: uuid.UUID | None = None,
    parent: uuid.UUID | None = None,
    payload: str = PAYLOAD_SENTINEL,
    operator: str | None = "escalate",
    round_generated: int = 1,
    breaches: int = 3,
) -> ArchivedAttack:
    return ArchivedAttack(
        id=attack_id or uuid.uuid4(),
        round_generated=round_generated,
        parent_id=parent,
        recombined_with=None,
        mutation_operator=operator,
        payload=payload,
        vector=DeliveryVector.DIRECT,
        objective=Objective.SYSPROMPT_EXTRACTION,
        technique=Technique.INSTRUCTION_OVERRIDE,
        cell_key="sysprompt_extraction|direct|instruction_override",
        novelty_score=None,
        first_breach_round=1,
        total_attempts=4,
        total_breaches=breaches,
        is_holdout=False,
        retired=False,
        benign_user_input=None,
        carrier_title=None,
        carrier_doc_id=None,
        created_at=datetime(2026, 9, 6, tzinfo=UTC),
    )


def round_report(
    number: int = 1,
    *,
    before: str = "",
    after: str = "",
    regressions: tuple[Regression, ...] = (),
    halt: HaltReason | None = None,
    promoted: bool = True,
) -> RoundReport:
    return RoundReport(
        run_id=RUN_ID,
        round_number=number,
        attacker_mode="black_box",
        defense_before=before or EMPTY.fingerprint(),
        defense_after=after or HARDENED.fingerprint(),
        attacks_generated=4,
        attacks_rejected_novelty=1,
        breaches_found=2,
        archive_block=Proportion(successes=13, trials=15),
        holdout_block=Proportion(successes=3, trials=3),
        utility_pass=Proportion(successes=40, trials=40),
        mean_novelty=0.71,
        cells_occupied=16,
        new_cells=2,
        regressions=regressions,
        config_promoted=promoted,
        halt_reason=halt,
        started_at=datetime(2026, 9, 6, tzinfo=UTC),
        ended_at=datetime(2026, 9, 6, tzinfo=UTC),
    )


def report_data(
    *,
    rounds: tuple[RoundReport, ...] | None = None,
    attacks: tuple[ArchivedAttack, ...] | None = None,
    halt: HaltReason | None = None,
) -> ReportData:
    chosen = attacks or (attack(),)
    return ReportData(
        run=RunReport(
            run_id=RUN_ID,
            status=RunStatus.HALTED if halt else RunStatus.COMPLETED,
            attacker_mode="black_box",
            starting_config_id=EMPTY.fingerprint(),
            final_config_id=HARDENED.fingerprint(),
            rounds=rounds or (round_report(),),
            halt_reason=halt,
        ),
        configs={EMPTY.fingerprint(): EMPTY, HARDENED.fingerprint(): HARDENED},
        attacks=chosen,
        cells=(
            CellRecord(
                cell_key="sysprompt_extraction|direct|instruction_override",
                elite_attack_id=chosen[0].id,
                elite_fitness=None,
                occupancy=2,
                last_updated_round=1,
            ),
        ),
        top_general=(GeneralAttack(attack=chosen[0], breached_configs=3, total_configs=4),),
        archive_size=len(chosen),
        holdout_size=2,
        unclassified=0,
    )


# ------------------------------------------------------------------ test 1


def test_the_markdown_report_is_byte_identical_across_runs() -> None:
    """A report that changes between runs cannot be reviewed in a diff."""
    data = report_data()

    first = render_run_report(data)
    second = render_run_report(data)

    assert first == second
    assert first.encode() == second.encode()


def test_the_round_report_is_also_deterministic() -> None:
    data = report_data()

    assert render_round_report(data, 1) == render_round_report(data, 1)


# ------------------------------------------------------------------ test 2


def test_every_rate_in_the_report_carries_an_interval() -> None:
    report = render_run_report(report_data())

    for rate in ("0.867", "1.000"):
        assert rate in report
    # Each rate line is followed by its interval and its n.
    assert report.count("[0.621, 0.963]") >= 1
    assert "(n=15)" in report
    assert "Wilson" in report


def test_a_report_from_data_missing_intervals_fails() -> None:
    """It fails rather than printing a bare number a reader would trust."""
    from crucible.loop.reports import BareRate

    broken = report_data().run.rounds[0].model_copy(update={"archive_block": 0.87})
    data = report_data()
    data = data.model_copy(update={"run": data.run.model_copy(update={"rounds": (broken,)})})

    with pytest.raises(BareRate):
        render_run_report(data)


# ------------------------------------------------------------------ test 3


def test_redaction_replaces_the_payload_with_a_mechanism_and_a_hash() -> None:
    redacted = redact_payload(
        PAYLOAD_SENTINEL,
        objective="sysprompt_extraction",
        vector="direct",
        technique="instruction_override",
    )

    assert PAYLOAD_SENTINEL not in redacted
    assert "sysprompt_extraction via direct using instruction_override" in redacted
    assert payload_hash(PAYLOAD_SENTINEL) in redacted


def test_the_default_report_contains_no_payload_no_canary_and_no_system_prompt() -> None:
    """Three sentinels, one for each thing docs/spec.md section 17 forbids."""
    poisoned = attack(payload=f"{PAYLOAD_SENTINEL} {CANARY_SENTINEL} {SYSTEM_PROMPT_SENTINEL}")
    report = render_run_report(report_data(attacks=(poisoned,)))

    assert PAYLOAD_SENTINEL not in report
    assert CANARY_SENTINEL not in report
    assert "CRUCIBLE-" not in report
    assert SYSTEM_PROMPT_SENTINEL not in report
    assert payload_hash(poisoned.payload) in report


def test_include_payloads_publishes_the_payload_deliberately() -> None:
    poisoned = attack(payload=PAYLOAD_SENTINEL)

    report = render_run_report(report_data(attacks=(poisoned,)), include_payloads=True)

    assert PAYLOAD_SENTINEL in report


def test_a_trace_is_stripped_of_canaries_and_payload() -> None:
    """Phase 2 traces carry that attempt's canary values so replay can work."""
    trace = {
        "canaries": {"SYSPROMPT": CANARY_SENTINEL, "TOOL": "CRUCIBLE-TOOL-dead"},
        "attempt": {
            "payload": PAYLOAD_SENTINEL,
            "objective": "sysprompt_extraction",
            "vector": "direct",
            "technique": "instruction_override",
        },
        "output": {"text": "safe"},
    }

    redacted = redact_trace(trace)

    assert CANARY_SENTINEL not in str(redacted)
    assert PAYLOAD_SENTINEL not in str(redacted)
    assert redacted["canaries"] == {"SYSPROMPT": "[redacted]", "TOOL": "[redacted]"}
    assert redacted["output"] == {"text": "safe"}


# ------------------------------------------------------------------ test 4


def test_the_report_lists_every_regression_and_every_halt_reason() -> None:
    first, second = uuid.uuid4(), uuid.uuid4()
    rounds = (
        round_report(
            1,
            regressions=(
                Regression(attack_id=first, cell_key="tool_hijack|direct|instruction_override"),
            ),
            promoted=False,
        ),
        round_report(2, regressions=(Regression(attack_id=second),), promoted=False),
        round_report(3, halt=HaltReason.SEARCH_STALLED),
    )

    report = render_run_report(report_data(rounds=rounds, halt=HaltReason.SEARCH_STALLED))

    assert str(first) in report
    assert str(second) in report
    assert "search_stalled" in report
    assert (
        "never\nauto-accepted" in report
        or "never **auto-accepted**" in report
        or "auto-accepted" in report
    )
    assert "halted run is a valid experiment" in report


def test_a_clean_run_says_so_rather_than_omitting_the_section() -> None:
    report = render_run_report(report_data())

    assert "No regression was detected" in report
    assert "No halt signal fired" in report


# ------------------------------------------------------------------ test 5


def test_the_config_diff_distinguishes_added_removed_and_changed() -> None:
    with_rule = HARDENED
    without_rule = HARDENED.model_copy(
        update={
            "input": InputLayer(heuristic_rules=()),
            "context": ContextLayer(spotlighting=Spotlighting.DATAMARKING),
        }
    )

    diff = diff_configs(without_rule, with_rule)
    rendered = diff.render()

    assert diff.of_kind(ChangeKind.ADDED), "the rule is new"
    assert diff.of_kind(ChangeKind.CHANGED), "spotlighting changed"
    assert any(line.startswith("+ ") for line in rendered.splitlines())
    assert any(line.startswith("~ ") for line in rendered.splitlines())

    reverse = diff_configs(with_rule, without_rule)
    assert reverse.of_kind(ChangeKind.REMOVED)
    assert any(line.startswith("- ") for line in reverse.render().splitlines())


def test_a_no_change_round_renders_without_crashing() -> None:
    diff = diff_configs(HARDENED, HARDENED)

    assert diff.changed is False
    assert "no change" in diff.render()

    data = report_data(
        rounds=(round_report(1, before=EMPTY.fingerprint(), after=EMPTY.fingerprint()),)
    )
    assert "no change" in render_round_report(data, 1)


def test_the_round_report_annotates_what_a_change_blocked_and_cost() -> None:
    rendered = render_round_report(report_data(), 1)

    assert "What it blocked" in rendered
    assert "What it cost" in rendered
    assert "Added" in rendered and "removed" in rendered and "changed" in rendered


# ------------------------------------------------------------------ test 6


def test_the_coverage_grid_renders_ninety_six_cells(tmp_path: Path) -> None:
    from crucible.archive.grid import all_cell_keys

    keys = all_cell_keys()
    occupancy = {key: index % 3 for index, key in enumerate(keys)}
    fitness = dict.fromkeys(keys[:10], 0.5)

    path = coverage_grid(fitness, occupancy, tmp_path / "grid.png")

    assert path.exists() and path.stat().st_size > 0
    assert len(keys) == 96


def test_an_empty_archive_renders_every_cell_empty_without_crashing(
    tmp_path: Path,
) -> None:
    path = coverage_grid({}, {}, tmp_path / "empty.png")

    assert path.exists() and path.stat().st_size > 0


def test_the_three_curve_chart_renders_bands_and_handles_no_rounds(
    tmp_path: Path,
) -> None:
    with_rounds = three_curves((round_report(1), round_report(2)), tmp_path / "curves.png")
    empty = three_curves((), tmp_path / "curves_empty.png")

    assert with_rounds.exists() and with_rounds.stat().st_size > 0
    assert empty.exists() and empty.stat().st_size > 0


def test_the_coverage_strip_renders_one_grid_per_round(tmp_path: Path) -> None:
    strip = coverage_strip(
        [(1, {"sysprompt_extraction|direct|instruction_override": 1}), (2, {})],
        tmp_path / "strip.png",
    )
    empty = coverage_strip([], tmp_path / "strip_empty.png")

    assert strip.exists() and strip.stat().st_size > 0
    assert empty.exists() and empty.stat().st_size > 0


# ------------------------------------------------------------------ test 7


def test_a_depth_four_lineage_renders_from_the_seed_down() -> None:
    seed = attack(operator=None, round_generated=0, payload="seed payload")
    second = attack(parent=seed.id, payload="second", operator="transpose_vector")
    third = attack(parent=second.id, payload="third", operator="obfuscate")
    fourth = attack(parent=third.id, payload="fourth", operator="escalate")
    by_id = {item.id: item for item in (seed, second, third, fourth)}

    nodes = build_lineage(fourth, by_id)
    rendered = render_lineage(nodes)

    assert [node.depth for node in nodes] == [0, 1, 2, 3]
    assert nodes[0].is_seed is True
    assert nodes[0].attack_id == seed.id
    assert nodes[-1].attack_id == fourth.id
    assert "[seed]" in rendered
    assert "[transpose_vector]" in rendered
    assert rendered.index("[seed]") < rendered.index("[escalate]")
    assert "      └─" in rendered, "deeper generations are indented further"


def test_a_seed_with_no_parent_renders_as_a_single_root() -> None:
    seed = attack(operator=None, parent=None, round_generated=0)

    nodes = build_lineage(seed, {seed.id: seed})
    rendered = render_lineage(nodes)

    assert len(nodes) == 1
    assert nodes[0].is_seed is True
    assert "[seed]" in rendered
    assert "└─" not in rendered


def test_a_lineage_whose_parent_is_missing_stops_cleanly() -> None:
    orphan = attack(parent=uuid.uuid4())

    nodes = build_lineage(orphan, {orphan.id: orphan})

    assert len(nodes) == 1
    assert nodes[0].attack_id == orphan.id


def test_an_empty_lineage_renders_a_message() -> None:
    assert "no lineage" in render_lineage([])
