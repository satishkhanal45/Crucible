"""Running one committed experiment.

Four kinds, and only the first of them is a loop:

* `loop` — a co-evolution run, with whichever guarded knob its ablation label
  licenses. `main` and the three ablations.
* `layer_ablation` — no generation at all: replay the final non-holdout archive
  against the final config with each layer disabled in turn.
* `transfer` — replay the final archive against the second `TargetAdapter`,
  with **no change to the loop**. The adapter Protocol is what makes that true;
  `tests/integration/test_transfer.py` asserts it.
* `model_overlap` — one attacker round per model family, then the overlap
  between what they produced, by cell and by embedding distance.

Every result is written to `experiment_results` so the regeneration script can
read it back. Nothing here interprets a number.
"""

from __future__ import annotations

import math
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from typing import Any

from crucible.attacker.llm import AttackerLLM
from crucible.attacker.prompts import build_defense_summary
from crucible.attacker.state import OutcomeSummary
from crucible.db.session import Database
from crucible.defenses.config import (
    ContextLayer,
    DefenseConfig,
    InputLayer,
    OutputLayer,
    PromptLayer,
    StructuralLayer,
)
from crucible.experiments.config import ExperimentConfig, ExperimentKind
from crucible.logging import get_logger
from crucible.loop.reports import RunReport
from crucible.loop.runner import (
    LoopFactories,
    LoopRunner,
    LoopSettings,
    build_components,
    postgres_checkpointer,
)
from crucible.repositories.configs import DefenseConfigRepository
from crucible.repositories.experiments import ExperimentResultRepository
from crucible.repositories.rounds import RunRepository
from crucible.schemas.attack import Attack
from crucible.schemas.experiments import (
    ExperimentResult,
    LayerAblationRow,
    ModelOverlapResult,
    TransferResult,
)
from crucible.schemas.provenance import STUB_PROVIDER
from crucible.services.embeddings import Embedder
from crucible.services.retry import ProviderError

logger = get_logger(__name__)


class ProviderMismatch(RuntimeError):
    """A model the overlap experiment needs is not the one that answered.

    Raised rather than falling back: an overlap computed against one family, or
    against a stub, is not an overlap measurement, and it would be reported as
    though it were.
    """

    def __init__(self, wanted: str, got: str) -> None:
        self.wanted = wanted
        self.got = got
        super().__init__(
            f"model overlap requires {wanted!r}, but the client reports {got!r}. "
            f"Check the model is available on this account "
            f"(https://console.groq.com/docs/deprecations); the experiment will not "
            f"fall back to a single family or to a stub."
        )


#: The five layers, in the order the table reports them.
LAYERS: tuple[str, ...] = ("input", "context", "prompt", "output", "structural")

#: An empty instance of each layer, which is what "disabled" means: the layer is
#: present but configured to do nothing, exactly as D(0) has it.
_EMPTY_LAYERS: dict[str, Any] = {
    "input": InputLayer(),
    "context": ContextLayer(),
    "prompt": PromptLayer(),
    "output": OutputLayer(),
    "structural": StructuralLayer(),
}


def disable_layer(config: DefenseConfig, layer: str) -> DefenseConfig:
    """The same config with one layer reset to its empty state.

    `output.canary_scan` is measurement rather than blocking, so disabling the
    output layer still leaves the oracle able to see a leak — the empty layer
    keeps `canary_scan` at its default, which is what D(0) does too.
    """
    if layer not in _EMPTY_LAYERS:
        raise KeyError(f"unknown layer {layer!r}. Layers: {', '.join(LAYERS)}")
    return config.model_copy(update={layer: _EMPTY_LAYERS[layer]})


@dataclass
class ExperimentContext:
    """What every experiment kind needs to run."""

    database: Database
    embedder: Embedder
    factories: LoopFactories
    allowlist: tuple[str, ...]
    checkpointer_settings: Any = None
    corpus: list[Any] | None = None
    #: Free-tier pacing, as `LoopSettings` field names. Empty means unpaced,
    #: which is what an offline test wants and what a live run must never get:
    #: the CLI fills it from settings via `pacing_settings()`.
    pacing: Mapping[str, Any] = field(default_factory=dict)


def pacing_settings(settings: Any) -> dict[str, Any]:
    """The pacing fields of `LoopSettings`, read from process settings.

    An experiment that skipped these would call both providers as fast as the
    loop can generate work, which on a free tier means a 429 storm rather than a
    result. The per-provider limits travel with the pair, because the two hosts
    do not publish the same ones.
    """
    return {
        "provider_max_concurrency": settings.PROVIDER_MAX_CONCURRENCY,
        "provider_min_interval_seconds": settings.PROVIDER_MIN_INTERVAL_SECONDS,
        "provider_tokens_per_minute": settings.PROVIDER_TOKENS_PER_MINUTE,
        "provider_requests_per_minute": settings.PROVIDER_REQUESTS_PER_MINUTE,
        "provider_rate_limits": settings.provider_rate_limits,
    }


def loop_settings_for(experiment: ExperimentConfig, **overrides: Any) -> LoopSettings:
    """Translate a committed experiment into the loop's own settings.

    The guarded knobs are copied across verbatim. They were already validated
    by `ExperimentConfig`, which is the only place a never-cut property can be
    switched off and the only place that can say which experiment may do it.
    """
    values: dict[str, Any] = {
        "rounds": experiment.rounds,
        "mode": experiment.mode,
        "budget_usd": experiment.budget_usd,
        "seed": experiment.seed,
        "concurrency": experiment.concurrency,
        "candidates_per_round": experiment.candidates_per_round,
        "cells_per_round": experiment.cells_per_round,
        "min_novelty": experiment.min_novelty,
        "utility_weight": experiment.utility_weight,
        "defender_scope": experiment.defender_scope.value,
        "persona": experiment.persona,
    }
    values.update(overrides)
    return LoopSettings(**values)


async def run_loop_experiment(
    experiment: ExperimentConfig,
    context: ExperimentContext,
    *,
    starting_config: DefenseConfig | None = None,
) -> RunReport:
    """Run `main` or one of the three loop ablations."""
    settings = loop_settings_for(experiment, **context.pacing)
    async with AsyncExitStack() as stack:
        checkpointer = None
        if context.checkpointer_settings is not None:
            checkpointer = await postgres_checkpointer(stack, context.checkpointer_settings)
        components = await build_components(
            context.database,
            settings=settings,
            factories=context.factories,
            embedder=context.embedder,
            allowlist=context.allowlist,
            corpus=context.corpus,
        )
        runner = LoopRunner(
            context.database, components, settings=settings, checkpointer=checkpointer
        )
        return await runner.start(starting_config=starting_config or DefenseConfig.empty())
    raise RuntimeError("unreachable")


async def run_layer_ablation(
    experiment: ExperimentConfig,
    context: ExperimentContext,
    *,
    config_id: str | None = None,
) -> tuple[LayerAblationRow, ...]:
    """Replay the final archive against each layer-disabled config in turn.

    Configs are resolved from `defense_configs` by id, which is why Phase 7's
    table was a blocking prerequisite for this phase.
    """
    settings = loop_settings_for(experiment, **context.pacing)
    resolved = await _resolve_config(context.database, config_id, experiment.source_run_id)
    components = await build_components(
        context.database,
        settings=settings,
        factories=context.factories,
        embedder=context.embedder,
        allowlist=context.allowlist,
        corpus=context.corpus,
    )

    rows: list[LayerAblationRow] = []
    full = await components.evaluation.evaluate_full(resolved, include_holdout=False)
    rows.append(
        LayerAblationRow(
            layer="(none: full config)",
            config_id=resolved.fingerprint(),
            archive_block_rate=full.archive_block_rate,
            utility_pass_rate=full.utility.pass_rate,
            archive_size=full.archive.evaluated,
            utility_total=full.utility.total,
        )
    )
    for layer in LAYERS:
        without = disable_layer(resolved, layer)
        evaluation = await components.evaluation.evaluate_full(without, include_holdout=False)
        rows.append(
            LayerAblationRow(
                layer=layer,
                config_id=without.fingerprint(),
                archive_block_rate=evaluation.archive_block_rate,
                utility_pass_rate=evaluation.utility.pass_rate,
                archive_size=evaluation.archive.evaluated,
                utility_total=evaluation.utility.total,
                delta_block_rate=evaluation.archive_block_rate - full.archive_block_rate,
                delta_utility=evaluation.utility.pass_rate - full.utility.pass_rate,
            )
        )
    return tuple(rows)


async def run_transfer(
    experiment: ExperimentConfig,
    context: ExperimentContext,
    *,
    config_id: str | None = None,
) -> TransferResult:
    """The final archive, unchanged, against the second target.

    Nothing in the loop is modified: `build_components` is handed a settings
    object whose `persona` names the second application, and every other
    component is assembled exactly as it is for the main run.
    """
    resolved = await _resolve_config(context.database, config_id, experiment.source_run_id)
    settings = loop_settings_for(experiment, **context.pacing)
    components = await build_components(
        context.database,
        settings=settings,
        factories=context.factories,
        embedder=context.embedder,
        allowlist=context.allowlist,
        corpus=context.corpus,
    )

    baseline = await components.evaluation.evaluate_full(
        DefenseConfig.empty(), include_holdout=False
    )
    hardened = await components.evaluation.evaluate_full(resolved, include_holdout=False)
    return TransferResult(
        persona=experiment.persona,
        within_family=True,
        source_config_id=resolved.fingerprint(),
        archive_size=hardened.archive.evaluated,
        baseline_block_rate=baseline.archive_block_rate,
        hardened_block_rate=hardened.archive_block_rate,
        baseline_utility=baseline.utility.pass_rate,
        hardened_utility=hardened.utility.pass_rate,
    )


def overlap_by_cell(first: Sequence[str], second: Sequence[str]) -> float:
    """Jaccard overlap of the cells two attacker runs occupied."""
    left, right = set(first), set(second)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def mean_nearest_distance(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]
) -> float:
    """Mean cosine distance from each attack in `first` to its nearest in `second`.

    0.0 means the two model families wrote the same attacks; 1.0 means they
    explored disjoint regions. This is the number that bounds how much of the
    archive is a property of one model's priors rather than of the defense.
    """
    if not first or not second:
        return 0.0
    total = 0.0
    for vector in first:
        total += min(_cosine_distance(vector, other) for other in second)
    return total / len(first)


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = float(sum(a * b for a, b in zip(left, right, strict=False)))
    # math.sqrt rather than ** 0.5: mypy --strict widens the power operator to
    # Any, which would silently un-type the whole distance calculation.
    left_norm = math.sqrt(float(sum(a * a for a in left)))
    right_norm = math.sqrt(float(sum(b * b for b in right)))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return 1.0 - (dot / (left_norm * right_norm))


async def run_model_overlap(
    experiment: ExperimentConfig,
    context: ExperimentContext,
    attacker_factories: Mapping[str, Callable[[], AttackerLLM]],
    *,
    config: DefenseConfig | None = None,
) -> ModelOverlapResult:
    """One attacker round per model family, against an identical starting state.

    The known-hard problem this bounds: an attacker agent explores ONE model's
    priors, so an archive built by a single family may be measuring that family
    rather than the defense. High cell overlap and low embedding distance mean
    the archive is largely model-independent; low overlap means it is not, and
    that belongs in the limitations rather than being discounted.

    A model that the account cannot reach **fails the experiment loudly, naming
    the model**. Falling back to one family, or to a stub, would produce a
    number that looks like an overlap measurement and is not one.
    """
    if len(attacker_factories) < 2:
        raise ValueError(
            f"model overlap needs two model families, got "
            f"{sorted(attacker_factories)}. One family cannot overlap with itself."
        )

    settings = loop_settings_for(experiment, **context.pacing)
    # Both families face the identical starting state; `config` is accepted so a
    # caller can compare them against a hardened config rather than D(0).
    del config
    per_model_cells: dict[str, list[str]] = {}
    per_model_counts: dict[str, int] = {}
    per_model_vectors: dict[str, list[list[float]]] = {}

    for model, factory in attacker_factories.items():
        components = await build_components(
            context.database,
            settings=settings,
            factories=replace(context.factories, attacker_llm=factory),
            embedder=context.embedder,
            allowlist=context.allowlist,
            corpus=context.corpus,
        )
        actual = getattr(components.attacker.llm, "model", "")
        if actual != model:
            raise ProviderMismatch(model, str(actual))
        if getattr(components.attacker.llm, "provider", "") == STUB_PROVIDER:
            raise ProviderMismatch(model, "stub")

        try:
            result = await components.attacker.run(
                {
                    "round": 1,
                    "target_capabilities": components.capabilities,
                    "behavior_spec": components.behavior,
                    "current_defense_summary": build_defense_summary(
                        settings.mode, OutcomeSummary(), defense=None
                    ),
                }
            )
        except ProviderError as error:
            # A model the account cannot reach ends the experiment naming it,
            # rather than quietly leaving one family out of the comparison.
            raise ProviderMismatch(model, f"call failed: {error}") from error

        accepted = list(result.get("accepted", []))
        per_model_counts[model] = len(accepted)
        per_model_cells[model] = sorted(
            {cell for cell in (_cell_of(attack) for attack in accepted) if cell}
        )
        per_model_vectors[model] = [
            list(vector)
            for vector in await context.embedder.embed([attack.payload for attack in accepted])
        ]

    first, second = list(attacker_factories)
    return ModelOverlapResult(
        models=(first, second),
        attacks_per_model=per_model_counts,
        cells_per_model=per_model_cells,
        cell_overlap=overlap_by_cell(per_model_cells[first], per_model_cells[second]),
        mean_nearest_distance=mean_nearest_distance(
            per_model_vectors[first], per_model_vectors[second]
        ),
    )


def _cell_of(attack: Attack) -> str | None:
    if attack.objective is None or attack.technique is None:
        return None
    return f"{attack.objective.value}|{attack.vector.value}|{attack.technique.value}"


async def _resolve_config(
    database: Database, config_id: str | None, source_run_id: str | None
) -> DefenseConfig:
    """The config an evaluation experiment replays against.

    Explicit id wins; then the named run's final config; then the most recent
    completed run's. A missing id is an error rather than a silent fallback to
    `empty()`, which would make a layer ablation measure nothing.
    """
    async with database.session() as session:
        configs = DefenseConfigRepository(session)
        if config_id is not None:
            found = await configs.get(config_id)
            if found is None:
                raise LookupError(f"no defense config {config_id!r} in defense_configs")
            return found

        runs = RunRepository(session)
        if source_run_id is not None:
            row = await runs.get(uuid.UUID(source_run_id))
            if row is None:
                raise LookupError(f"no run {source_run_id!r}")
        else:
            recent = await runs.list_recent()
            if not recent:
                raise LookupError(
                    "no run has been recorded yet: run `crucible experiment run main` first"
                )
            row = recent[0]
        found = await configs.get(row.current_config_id)
        if found is None:
            raise LookupError(
                f"run {row.id} names config {row.current_config_id!r}, which is not stored"
            )
        return found


async def record_result(
    database: Database, experiment: ExperimentConfig, result: ExperimentResult
) -> uuid.UUID:
    """Store what an experiment produced, so findings can regenerate from it."""
    async with database.session() as session:
        return await ExperimentResultRepository(session).record(experiment.name, result)


def seeded_rng(experiment: ExperimentConfig) -> random.Random:
    """Every experiment is reproducible from its config plus this seed."""
    return random.Random(experiment.seed)


def cost_estimate_minutes(experiment: ExperimentConfig, *, archive_size: int = 90) -> int:
    """Rough wall clock at the free tier's 6500 tokens per minute.

    A full-archive evaluation of ~90 attacks is about 20 minutes, and a loop
    round costs one generation pass plus two such evaluations.
    """
    per_evaluation = max(1, round(archive_size / 4.5))
    if experiment.kind is ExperimentKind.LAYER_ABLATION:
        return per_evaluation * (len(LAYERS) + 1)
    if experiment.kind is ExperimentKind.TRANSFER:
        return per_evaluation * 2
    if experiment.kind is ExperimentKind.MODEL_OVERLAP:
        return 15 * max(1, len(experiment.models))
    return experiment.rounds * (per_evaluation * 2 + 10)


__all__ = [
    "LAYERS",
    "ExperimentContext",
    "LayerAblationRow",
    "ModelOverlapResult",
    "ProviderMismatch",
    "TransferResult",
    "cost_estimate_minutes",
    "disable_layer",
    "loop_settings_for",
    "mean_nearest_distance",
    "overlap_by_cell",
    "record_result",
    "run_layer_ablation",
    "run_loop_experiment",
    "run_model_overlap",
    "run_transfer",
    "seeded_rng",
]
