"""The Markdown report — the document that goes in the repository.

Two rules shape everything here.

**Every rate carries its interval.** `RoundReport` already types its rates as
`Proportion`, and the renderer calls `validate_intervals()` before writing a
line, so a report generated from data missing intervals *fails* rather than
printing a bare number that a reader would take at face value.

**Nothing published is reusable as an attack.** Payloads are redacted to their
mechanism and a hash unless `--include-payloads` is given, and canary values and
the target's system prompt never appear at all (docs/spec.md section 17).
"""

from __future__ import annotations

from itertools import pairwise

from crucible.defenses.config import DefenseConfig
from crucible.loop.reports import RoundReport
from crucible.loop.statistics import Proportion, two_proportion_test
from crucible.reporting.data import ReportData
from crucible.reporting.diff import ChangeKind, diff_configs
from crucible.reporting.lineage import build_lineage, render_lineage
from crucible.reporting.redaction import present_payload
from crucible.schemas.taxonomy import GRID_DENOMINATOR

LIMITATIONS = (
    "The oracle is deterministic only: Tier 1 canary and tool-authorization "
    "checks plus Tier 2 rule-based behaviour. There is no LLM judge in this "
    "build (cut B4), so no judge-calibration figure is reported.",
    "Two delivery vectors are executable, `direct` and `indirect_document` "
    "(cut B3), so the coverage denominator is "
    f"{GRID_DENOMINATOR} cells rather than 192.",
    "The benign task set is 40 tasks with 12 hard negatives, against a spec "
    "ideal of 60 with 15 (deferred item D6).",
    "Layer 1's model-based classifier is disabled (cut B5); input inspection is "
    "heuristic rules only.",
    "Novelty k-NN is an exact scan. Approximate search becomes necessary above "
    "roughly 20k archive entries, at a measured recall cost of 0.24-0.52 at "
    "k=15 (deferred item D8).",
    "Sample sizes are small. Every rate below carries a Wilson interval, and "
    "round-over-round differences whose intervals overlap are not improvements.",
)


def _rate(proportion: Proportion) -> str:
    low, high = proportion.interval
    return f"`{proportion.rate:.3f}` [{low:.3f}, {high:.3f}] (n={proportion.trials})"


def _round_table(rounds: tuple[RoundReport, ...]) -> list[str]:
    lines = [
        "| round | archive block rate | holdout block rate | overfit gap | "
        "utility pass rate | coverage | new cells | regressions |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for report in rounds:
        lines.append(
            f"| {report.round_number} | {_rate(report.archive_block)} | "
            f"{_rate(report.holdout_block)} | `{report.overfit_gap:+.3f}` | "
            f"{_rate(report.utility_pass)} | "
            f"`{report.cells_occupied}/{GRID_DENOMINATOR}` | "
            f"`{report.new_cells}` | {len(report.regressions) or '—'} |"
        )
    return lines


def _significance(rounds: tuple[RoundReport, ...]) -> list[str]:
    """Round-over-round claims, tested rather than asserted."""
    if len(rounds) < 2:
        return ["Only one round: there is no round-over-round comparison to make."]
    lines = ["| comparison | difference | p | verdict |", "|---|---|---|---|"]
    for previous, current in pairwise(rounds):
        test = two_proportion_test(previous.archive_block, current.archive_block)
        verdict = (
            "significant" if test.significant else "**not** an improvement: the intervals overlap"
        )
        lines.append(
            f"| round {previous.round_number} -> {current.round_number} | "
            f"`{test.difference:+.3f}` | `{test.p_value:.4f}` | {verdict} |"
        )
    return lines


def _changelog(data: ReportData) -> list[str]:
    """Per round: what the defender changed, and what it measured."""
    lines: list[str] = []
    for report in data.rounds:
        before = data.configs.get(report.defense_before)
        after = data.configs.get(report.defense_after)
        lines.append(f"### Round {report.round_number}")
        lines.append("")
        if before is None or after is None:
            lines.append("_Config bodies were not stored for this round._")
            lines.append("")
            continue
        diff = diff_configs(before, after)
        lines.append("```")
        lines.append(diff.render())
        lines.append("```")
        lines.append("")
        lines.append(
            f"Measured effect: archive block rate {_rate(report.archive_block)}, "
            f"utility {_rate(report.utility_pass)}, holdout {_rate(report.holdout_block)}."
        )
        if not report.config_promoted:
            reason = (
                "the config reopened archived attacks"
                if report.regressions
                else "the defender proposed nothing better"
            )
            lines.append("")
            lines.append(f"**Not promoted**: {reason}. D(n) stayed D(n-1).")
        lines.append("")
    return lines


def _top_attacks(data: ReportData, *, include_payloads: bool) -> list[str]:
    if not data.top_general:
        return ["No attack has breached any configuration yet."]
    lines: list[str] = []
    by_id = data.by_id
    for index, item in enumerate(data.top_general, start=1):
        attack = item.attack
        lines.append(
            f"#### {index}. `{attack.id}` — generality "
            f"`{item.generality:.2f}` ({item.breached_configs}/{item.total_configs} configs)"
        )
        lines.append("")
        lines.append(f"- cell: `{attack.cell_key or 'unclassified'}`")
        lines.append(f"- breaches: `{item.breaches}` of `{item.attempts}` recorded attempts")
        lines.append(
            "- mechanism: "
            + present_payload(
                attack.payload,
                objective=attack.objective.value if attack.objective else None,
                vector=attack.vector.value,
                technique=attack.technique.value if attack.technique else None,
                include_payloads=include_payloads,
            )
        )
        lines.append("")
        lines.append("Lineage:")
        lines.append("")
        lines.append("```")
        lines.append(
            render_lineage(build_lineage(attack, by_id, include_payloads=include_payloads))
        )
        lines.append("```")
        lines.append("")
    return lines


def render_run_report(data: ReportData, *, include_payloads: bool = False) -> str:
    """The whole run, as Markdown. Deterministic for a given database state."""
    for report in data.rounds:
        report.validate_intervals()

    run = data.run
    lines: list[str] = [
        f"# Crucible run `{run.run_id}`",
        "",
        "## Methodology",
        "",
        "An attacker agent mutates archived attacks against the current defense "
        "configuration; surviving candidates are filtered for novelty before "
        "execution and scored by a deterministic oracle. A defender agent then "
        "proposes configuration changes, each evaluated against the **full "
        "non-holdout archive**, 40 benign tasks, and latency. The selected "
        "configuration is checked for regressions against the previous one and "
        "measured on a held-out attack set it never sees.",
        "",
        f"- attacker mode: `{run.attacker_mode}`",
        f"- D(0): `{run.starting_config_id}` — the empty configuration, so the "
        "loop starts weak and any hardening is visible",
        f"- final configuration: `{run.final_config_id}`",
        f"- rounds completed: `{run.rounds_completed}`",
        f"- status: `{run.status.value}`"
        + (f" (`{run.halt_reason.value}`)" if run.halt_reason else ""),
        f"- archive: `{data.archive_size}` attacks, `{data.holdout_size}` held out, "
        f"`{data.unclassified}` unclassified",
        f"- coverage: `{data.coverage}`",
        "",
        "## The three curves",
        "",
        "Archive block rate, holdout block rate, and utility pass rate, each with "
        "a 95% Wilson interval. The gap between the first two is the overfitting "
        "measure; a bare rate is not a result.",
        "",
        *_round_table(data.rounds),
        "",
        "### Round-over-round significance",
        "",
        *_significance(data.rounds),
        "",
        "## Coverage evolution",
        "",
        f"Coverage is out of **{GRID_DENOMINATOR}** cells "
        f"(6 objectives x 2 executable vectors x 8 techniques).",
        "",
        "| round | cells occupied | new cells | mean novelty of accepted attacks |",
        "|---|---|---|---|",
    ]
    for report in data.rounds:
        lines.append(
            f"| {report.round_number} | `{report.cells_occupied}/{GRID_DENOMINATOR}` | "
            f"`{report.new_cells}` | `{report.mean_novelty:.3f}` |"
        )

    lines.extend(
        [
            "",
            "## Most general attacks",
            "",
            "Ranked by the fraction of all past defense configurations each still "
            "breaches. Payloads are redacted to their mechanism and a hash unless "
            "the report was generated with `--include-payloads`.",
            "",
            *_top_attacks(data, include_payloads=include_payloads),
            "## Defense changelog",
            "",
            *_changelog(data),
            "## Regressions and halts",
            "",
        ]
    )

    regressions = [
        (report.round_number, regression)
        for report in data.rounds
        for regression in report.regressions
    ]
    if regressions:
        lines.append("| round | attack | cell | before | after |")
        lines.append("|---|---|---|---|---|")
        for round_number, regression in regressions:
            lines.append(
                f"| {round_number} | `{regression.attack_id}` | "
                f"`{regression.cell_key or 'unclassified'}` | "
                f"`{regression.previous_outcome}` | `{regression.new_outcome}` |"
            )
        lines.append("")
        lines.append(
            "A configuration that reopens an archived attack is **never "
            "auto-accepted**: the round keeps the previous configuration."
        )
    else:
        lines.append("No regression was detected: no round reopened an archived attack.")
    lines.append("")

    halts = [
        (report.round_number, report.halt_reason)
        for report in data.rounds
        if report.halt_reason is not None
    ]
    if halts or run.halt_reason:
        lines.append("Halt signals:")
        lines.append("")
        for round_number, reason in halts:
            lines.append(f"- round {round_number}: `{reason.value if reason else ''}`")
        if run.halt_reason and not halts:
            lines.append(f"- run: `{run.halt_reason.value}`")
        lines.append("")
        lines.append(
            "A halted run is a valid experiment, not a failure: it is a result "
            "about where this search stops."
        )
    else:
        lines.append("No halt signal fired.")

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in LIMITATIONS)
    lines.append("")
    return "\n".join(lines)


def render_round_report(data: ReportData, round_number: int) -> str:
    """One round, plus the structured config diff that produced it.

    A round report carries no payloads at all — the diff, the rates and the
    regression ids are the whole content — so it needs no redaction switch.
    """
    report = next((item for item in data.rounds if item.round_number == round_number), None)
    if report is None:
        return f"# Round {round_number}\n\nNo such round in run `{data.run.run_id}`.\n"
    report.validate_intervals()

    before = data.configs.get(report.defense_before)
    after = data.configs.get(report.defense_after)
    lines = [
        f"# Round {report.round_number} of run `{data.run.run_id}`",
        "",
        f"- D(n-1): `{report.defense_before}`",
        f"- D(n): `{report.defense_after}`"
        + ("" if report.config_promoted else "  _(not promoted)_"),
        f"- attacks generated: `{report.attacks_generated}` "
        f"(`{report.attacks_rejected_novelty}` rejected on novelty, "
        f"`{report.breaches_found}` breached)",
        f"- mean novelty: `{report.mean_novelty:.3f}`",
        f"- cost: `${report.cost_usd:.4f}`",
        "",
        "## Rates",
        "",
        *_round_table((report,)),
        "",
        "## Configuration diff",
        "",
    ]
    if before is None or after is None:
        lines.append("_Config bodies were not stored for this round._")
    else:
        diff = diff_configs(before, after)
        lines.append("```")
        lines.append(diff.render())
        lines.append("```")
        lines.append("")
        lines.append(
            f"Added `{len(diff.of_kind(ChangeKind.ADDED))}`, "
            f"removed `{len(diff.of_kind(ChangeKind.REMOVED))}`, "
            f"changed `{len(diff.of_kind(ChangeKind.CHANGED))}`."
        )
        lines.append("")
        lines.append(
            f"What it blocked: archive block rate {_rate(report.archive_block)}. "
            f"What it cost: utility {_rate(report.utility_pass)}."
        )
    lines.extend(["", "## Regressions", ""])
    if report.regressions:
        for regression in report.regressions:
            lines.append(f"- `{regression}`")
        lines.append("")
        lines.append("The configuration was **not** promoted.")
    else:
        lines.append("None.")
    lines.append("")
    if report.halt_reason is not None:
        lines.append(f"## Halt\n\n`{report.halt_reason.value}`\n")
    return "\n".join(lines)


def config_summary(config: DefenseConfig) -> str:
    """A one-line description, for tables."""
    return (
        f"rules={len(config.input.heuristic_rules)} "
        f"spotlight={config.context.spotlighting.value} "
        f"tools={len(config.structural.tool_allowlist)} "
        f"user_origin={config.structural.require_user_origin_for_privileged}"
    )
