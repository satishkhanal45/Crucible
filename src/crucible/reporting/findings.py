"""Regenerating every number in `docs/findings.md`.

The standing rule for that document is that **no figure is typed by hand**. This
module reads the `runs`, `rounds`, `attempts`, `defense_configs`,
`classifier_agreement` and `experiment_results` tables and rewrites the blocks
marked

    <!-- BEGIN GENERATED: <key> -->
    ...
    <!-- END GENERATED: <key> -->

leaving everything outside those markers — which is all of the interpretation —
exactly as written. `crucible findings regenerate --check` rewrites nothing and
exits non-zero if any block is stale, which is what CI runs.

Interpretation is a person's job. Nothing here writes a sentence about what a
number means.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.db.session import Database
from crucible.loop.statistics import Proportion
from crucible.repositories.agreement import ClassifierAgreementRepository
from crucible.repositories.experiments import ExperimentResultRepository
from crucible.repositories.rounds import RoundRepository, RunRepository
from crucible.schemas.agreement import (
    SOFT_AXIS_THRESHOLD,
    UNMEASURED_CAPTION,
    ClassifierAgreement,
)
from crucible.schemas.provenance import RunProvenance
from crucible.schemas.taxonomy import GRID_DENOMINATOR

FINDINGS_PATH = Path("docs/findings.md")

BEGIN = "<!-- BEGIN GENERATED: {key} -->"
END = "<!-- END GENERATED: {key} -->"

_BLOCK = re.compile(
    r"<!-- BEGIN GENERATED: (?P<key>[a-z0-9_]+) -->\n(?P<body>.*?)<!-- END GENERATED: (?P=key) -->",
    re.DOTALL,
)

#: What a block says before its experiment has ever been run. It is deliberately
#: loud: an empty table would read as a measured zero.
NOT_RUN = "_Not run yet. `crucible experiment run {name}` writes this block._"


@dataclass(frozen=True)
class FindingsData:
    """Everything the generated blocks are rendered from."""

    main_run_id: uuid.UUID | None
    rounds: tuple[Any, ...]
    agreement: ClassifierAgreement | None
    experiments: Mapping[str, dict[str, Any]]
    archive_size: int = 0
    holdout_size: int = 0
    cells_occupied: int = 0
    #: True when the run these numbers came from used scripted clients. A
    #: stubbed run's figures are shaped exactly like real ones, which is why
    #: regeneration refuses them rather than printing them with a caveat.
    stubbed: bool = False
    provenance: RunProvenance = field(default_factory=RunProvenance)


def _rate(successes: int, trials: int) -> str:
    """A rate is never reported without its interval and its denominator."""
    proportion = Proportion(successes=successes, trials=trials)
    low, high = proportion.interval
    return f"`{proportion.rate:.3f}` [{low:.3f}, {high:.3f}] (n={trials})"


def _rate_from(rate: float, trials: int) -> str:
    return _rate(round(rate * trials), trials)


def render_main_curves(data: FindingsData) -> str:
    if not data.rounds:
        return NOT_RUN.format(name="main")
    lines = [
        "| round | archive block rate | holdout block rate | overfit gap | utility pass rate |"
        " coverage | regressions |",
        "|---|---|---|---|---|---|---|",
    ]
    for report in data.rounds:
        # The report's own definition, so findings and the run report agree on
        # the sign: archive minus holdout, positive meaning overfitted.
        gap = report.overfit_gap
        lines.append(
            f"| {report.round_number} "
            f"| {_rate(report.archive_block.successes, report.archive_block.trials)} "
            f"| {_rate(report.holdout_block.successes, report.holdout_block.trials)} "
            f"| `{gap:+.3f}` "
            f"| {_rate(report.utility_pass.successes, report.utility_pass.trials)} "
            f"| `{report.cells_occupied}/{GRID_DENOMINATOR}` "
            f"| `{len(report.regressions)}` |"
        )
    return "\n".join(lines)


def render_headline(data: FindingsData) -> str:
    if not data.rounds:
        return NOT_RUN.format(name="main")
    first, last = data.rounds[0], data.rounds[-1]
    lines = [
        f"- rounds completed: `{len(data.rounds)}`",
        f"- archive block rate, round {first.round_number}: "
        f"{_rate(first.archive_block.successes, first.archive_block.trials)}",
        f"- archive block rate, round {last.round_number}: "
        f"{_rate(last.archive_block.successes, last.archive_block.trials)}",
        f"- holdout block rate, round {last.round_number}: "
        f"{_rate(last.holdout_block.successes, last.holdout_block.trials)}",
        f"- benign utility, round {last.round_number}: "
        f"{_rate(last.utility_pass.successes, last.utility_pass.trials)}",
        f"- archive: `{data.archive_size}` attacks, `{data.holdout_size}` held out",
        f"- coverage: `{data.cells_occupied}/{GRID_DENOMINATOR}`",
        f"- regressions detected across the run: "
        f"`{sum(len(report.regressions) for report in data.rounds)}`",
    ]
    return "\n".join(lines)


def render_agreement(data: FindingsData) -> str:
    agreement = data.agreement
    if agreement is None:
        return f"_{UNMEASURED_CAPTION}_"
    lines = [
        f"Measured with `{agreement.model_name}` over the "
        f"{agreement.total} hand-declared seed cells.",
        "",
        "| axis | agreement | status |",
        "|---|---|---|",
    ]
    for axis in (agreement.objective, agreement.technique, agreement.combined):
        status = "soft metric" if axis.is_soft else "measurement"
        lines.append(f"| {axis.axis} | `{axis.agreed}/{axis.total}` ({axis.rate:.0%}) | {status} |")
    lines.extend(
        [
            "",
            f"- unclassified after one retry: `{agreement.unclassified}`",
            f"- soft-metric threshold: `{SOFT_AXIS_THRESHOLD:.2f}`",
        ]
    )
    return "\n".join(lines)


def render_layer_ablation(data: FindingsData) -> str:
    payload = data.experiments.get("layer_ablation")
    if not payload:
        return NOT_RUN.format(name="layer_ablation")
    rows = payload.get("rows") or []
    lines = [
        "| layer disabled | archive block rate | delta | utility pass rate | delta | config id |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        size = int(row.get("archive_size", 0))
        utility_total = int(row.get("utility_total", 0))
        lines.append(
            f"| `{row['layer']}` "
            f"| {_rate_from(float(row['archive_block_rate']), size)} "
            f"| `{float(row.get('delta_block_rate', 0.0)):+.3f}` "
            f"| {_rate_from(float(row['utility_pass_rate']), utility_total)} "
            f"| `{float(row.get('delta_utility', 0.0)):+.3f}` "
            f"| `{str(row['config_id'])[:12]}` |"
        )
    return "\n".join(lines)


def render_transfer(data: FindingsData) -> str:
    payload = data.experiments.get("transfer")
    if not payload:
        return NOT_RUN.format(name="transfer")
    size = int(payload.get("archive_size", 0))
    family = "WITHIN-FAMILY" if payload.get("within_family", True) else "cross-family"
    lines = [
        f"Second target: `{payload.get('persona')}`. This is **{family}** transfer: the second "
        "application has its own system prompt, corpus domain and tool set, but shares "
        "Crucible's retrieval and defense-layer machinery.",
        "",
        "| configuration | archive block rate | benign utility |",
        "|---|---|---|",
        f"| D(0), empty | {_rate_from(float(payload['baseline_block_rate']), size)} "
        f"| `{float(payload['baseline_utility']):.3f}` |",
        f"| final config from the main run | "
        f"{_rate_from(float(payload['hardened_block_rate']), size)} "
        f"| `{float(payload['hardened_utility']):.3f}` |",
        "",
        f"- hardening delta on the second target: "
        f"`{float(payload['hardened_block_rate']) - float(payload['baseline_block_rate']):+.3f}`",
        f"- source config: `{str(payload.get('source_config_id', ''))[:12]}`",
    ]
    return "\n".join(lines)


def render_model_overlap(data: FindingsData) -> str:
    payload = data.experiments.get("model_overlap")
    if not payload:
        return NOT_RUN.format(name="model_overlap")
    models = payload.get("models") or []
    counts = payload.get("attacks_per_model") or {}
    lines = [
        "| model | attacks generated | cells occupied |",
        "|---|---|---|",
    ]
    cells = payload.get("cells_per_model") or {}
    for model in models:
        lines.append(f"| `{model}` | `{counts.get(model, 0)}` | `{len(cells.get(model, []))}` |")
    lines.extend(
        [
            "",
            f"- cell overlap (Jaccard): `{float(payload.get('cell_overlap', 0.0)):.3f}`",
            "- mean nearest-neighbour embedding distance: "
            f"`{float(payload.get('mean_nearest_distance', 0.0)):.3f}`",
        ]
    )
    return "\n".join(lines)


def render_ablations(data: FindingsData) -> str:
    names = ("ablation_novelty", "ablation_utility", "ablation_archive")
    lines = [
        "| ablation | run | rounds | status | halt reason |",
        "|---|---|---|---|---|",
    ]
    seen = False
    for name in names:
        payload = data.experiments.get(name)
        if not payload:
            lines.append(f"| `{name}` | _not run_ | - | - | - |")
            continue
        seen = True
        lines.append(
            f"| `{name}` | `{str(payload.get('run_id', ''))[:8]}` "
            f"| `{payload.get('rounds_completed', 0)}` "
            f"| `{payload.get('status', '')}` "
            f"| `{payload.get('halt_reason') or '-'}` |"
        )
    if not seen:
        lines.append("")
        lines.append(
            "_No ablation has been run yet. Each writes its row when "
            "`crucible experiment run <name>` completes._"
        )
    return "\n".join(lines)


#: Every generated block, by marker key.
RENDERERS: dict[str, Any] = {
    "headline": render_headline,
    "main_curves": render_main_curves,
    "ablations": render_ablations,
    "layer_ablation": render_layer_ablation,
    "transfer": render_transfer,
    "model_overlap": render_model_overlap,
    "taxonomy_agreement": render_agreement,
}


async def gather(database: Database, *, run_id: uuid.UUID | None = None) -> FindingsData:
    """Read everything the generated blocks need, in one pass."""
    async with database.session() as session:
        runs = RunRepository(session)
        row = None
        if run_id is not None:
            row = await runs.get(run_id)
        else:
            recent = await runs.list_recent()
            row = recent[0] if recent else None

        rounds: tuple[Any, ...] = ()
        if row is not None:
            rounds = tuple(await RoundRepository(session).list_for_run(row.id))

        agreement = await ClassifierAgreementRepository(session).latest()
        experiments = await ExperimentResultRepository(session).all_latest()

        from crucible.repositories.attacks import AttackRepository

        attacks = AttackRepository(session)
        all_attacks = await attacks.list_all()
        holdout = await attacks.count_holdout()

    occupied = len({attack.cell_key for attack in all_attacks if attack.cell_key})
    return FindingsData(
        main_run_id=row.id if row is not None else None,
        rounds=rounds,
        agreement=agreement,
        experiments=experiments,
        archive_size=len(all_attacks),
        holdout_size=holdout,
        cells_occupied=occupied,
        stubbed=bool(row.stubbed) if row is not None else False,
        provenance=(
            RunProvenance.model_validate(row.provenance or {})
            if row is not None
            else RunProvenance()
        ),
    )


class StubbedRunRefused(RuntimeError):
    """Regeneration was asked to publish numbers a stub produced.

    `docs/findings.md` is the deliverable, and its standing rule is that every
    figure regenerates from stored data. A figure produced by a scripted client
    satisfies that rule and means nothing, which is the one way the rule can be
    satisfied dishonestly. So it is refused rather than bannered: a banner in a
    document people quote from is not enough.
    """

    def __init__(self, provenance: RunProvenance) -> None:
        self.provenance = provenance
        agents = ", ".join(provenance.render_lines()) or "not recorded"
        super().__init__(
            "refusing to write findings from a STUBBED run: its numbers are the output "
            f"of test doubles, not of a language model ({agents}). Re-run the experiment "
            "with `--provider groq`, or pass `--force-stubbed` if you are deliberately "
            "producing an example document that will not be published."
        )


def guard_stubbed(data: FindingsData, *, force: bool = False) -> None:
    """Refuse to emit numbers from a stubbed run unless explicitly forced."""
    if data.stubbed and not force:
        raise StubbedRunRefused(data.provenance)


def generated_keys(markdown: str) -> tuple[str, ...]:
    return tuple(match.group("key") for match in _BLOCK.finditer(markdown))


def rewrite(markdown: str, data: FindingsData) -> str:
    """Replace every generated block's body. Everything else is left alone."""

    def replace(match: re.Match[str]) -> str:
        key = match.group("key")
        renderer = RENDERERS.get(key)
        if renderer is None:
            raise KeyError(
                f"docs/findings.md has a generated block {key!r} with no renderer. "
                f"Known blocks: {', '.join(sorted(RENDERERS))}"
            )
        body = renderer(data).rstrip("\n")
        return f"{BEGIN.format(key=key)}\n{body}\n{END.format(key=key)}"

    return _BLOCK.sub(replace, markdown)


def stale_blocks(markdown: str, data: FindingsData) -> tuple[str, ...]:
    """Which generated blocks differ from what the stored data produces."""
    stale: list[str] = []
    for match in _BLOCK.finditer(markdown):
        key = match.group("key")
        renderer = RENDERERS.get(key)
        if renderer is None:
            stale.append(key)
            continue
        if match.group("body").strip() != renderer(data).strip():
            stale.append(key)
    return tuple(stale)
