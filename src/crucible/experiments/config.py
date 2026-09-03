"""Experiment configuration, with the guards that keep an ablation an ablation.

Two of this project's never-cut properties are violated on purpose in Phase 8,
because the only way to show that a property matters is to measure what happens
without it:

* the 2.0 weight on `utility_loss` (docs/spec.md section 12), and
* the defender's view of the whole archive rather than the current round.

Each is violated **exactly once, under a name**. `ExperimentConfig` refuses to
load a config that turns one off outside its own named ablation, so a knob
cannot be left flipped in the main run by accident and quietly change every
number in `docs/findings.md`. `MIN_NOVELTY` is guarded the same way.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from crucible.archive.novelty import MIN_NOVELTY
from crucible.attacker.state import AttackerMode
from crucible.evaluation.objective import UTILITY_WEIGHT

#: Committed configs. One file per experiment, every one with an explicit seed.
EXPERIMENTS_DIR = Path("experiments")


class NeverCutViolation(Exception):
    """A config disabled a never-cut property outside its own named ablation.

    Deliberately NOT a `ValueError`. Pydantic converts `ValueError` raised in a
    validator into a `ValidationError`, which would bury the one sentence that
    says which never-cut property was touched under a wrapper. Raised from a
    model validator, this propagates as itself, so a caller can catch the
    specific thing that went wrong.
    """


class ExperimentKind(StrEnum):
    """What running this experiment actually does."""

    LOOP = "loop"
    LAYER_ABLATION = "layer_ablation"
    TRANSFER = "transfer"
    MODEL_OVERLAP = "model_overlap"


class AblationLabel(StrEnum):
    """The label that licenses one never-cut property to be switched off."""

    NONE = "none"
    NOVELTY = "ablation_novelty"
    UTILITY = "ablation_utility"
    ARCHIVE = "ablation_archive"


class DefenderScope(StrEnum):
    ARCHIVE = "archive"
    CURRENT_ROUND = "current_round"


class ExperimentConfig(BaseModel):
    """One committed experiment. Reproducible from this file plus its seed."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=64)
    kind: ExperimentKind = ExperimentKind.LOOP
    description: str = ""
    #: Why this round count and not more. Recorded as a limitation in findings.
    rounds_rationale: str = ""
    seed: int
    rounds: int = Field(default=6, ge=1, le=50)
    mode: AttackerMode = AttackerMode.BLACK_BOX
    budget_usd: Decimal = Field(default=Decimal("5.00"), gt=Decimal(0))
    concurrency: int = Field(default=1, ge=1, le=8)
    candidates_per_round: int = Field(default=4, ge=3, le=5)
    cells_per_round: int = Field(default=4, ge=1, le=16)

    #: The three guarded knobs. Defaults are the never-cut values.
    min_novelty: float = Field(default=MIN_NOVELTY, ge=0.0, le=1.0)
    utility_weight: float = Field(default=UTILITY_WEIGHT, ge=0.0, le=10.0)
    defender_scope: DefenderScope = DefenderScope.ARCHIVE
    ablation: AblationLabel = AblationLabel.NONE

    #: Which application to run against. `meridian` is the transfer target.
    persona: str = "northwind"
    #: Models to compare, for `model_overlap` only.
    models: tuple[str, ...] = ()
    #: The run whose final archive this experiment replays, when it needs one.
    source_run_id: str | None = None

    @model_validator(mode="after")
    def _guard_never_cut(self) -> Self:
        """Each never-cut property may be off only in its own named ablation."""
        if self.utility_weight != UTILITY_WEIGHT and self.ablation is not AblationLabel.UTILITY:
            raise NeverCutViolation(
                f"experiment {self.name!r} sets utility_weight={self.utility_weight} but is not "
                f"labelled {AblationLabel.UTILITY.value!r}. The 2.0 weight on utility_loss is a "
                f"never-cut property (docs/spec.md section 12); it is violated once, under that "
                f"label, to measure the consequence."
            )
        if (
            self.defender_scope is DefenderScope.CURRENT_ROUND
            and self.ablation is not AblationLabel.ARCHIVE
        ):
            raise NeverCutViolation(
                f"experiment {self.name!r} blinds the defender to the archive but is not labelled "
                f"{AblationLabel.ARCHIVE.value!r}. Full-archive re-evaluation is a never-cut "
                f"property; it is violated once, under that label, to measure the consequence."
            )
        if self.min_novelty != MIN_NOVELTY and self.ablation is not AblationLabel.NOVELTY:
            raise NeverCutViolation(
                f"experiment {self.name!r} moves MIN_NOVELTY to {self.min_novelty} but is not "
                f"labelled {AblationLabel.NOVELTY.value!r}. Novelty pressure is a never-cut "
                f"property; it is violated once, under that label, to measure the consequence."
            )
        # And the label has to be spent on the experiment that owns it: a config
        # called `main` cannot claim the utility label to unlock the knob.
        if self.ablation is not AblationLabel.NONE and self.name != self.ablation.value:
            raise NeverCutViolation(
                f"experiment {self.name!r} claims the {self.ablation.value!r} label. The label "
                f"belongs to the experiment of the same name and nowhere else."
            )
        return self

    @property
    def violates_never_cut(self) -> bool:
        return self.ablation is not AblationLabel.NONE

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


#: What `--smoke` shrinks an experiment to: enough to prove the providers
#: answer and the numbers arrive, cheap enough to run before committing hours.
SMOKE_ROUNDS = 1
SMOKE_CANDIDATES = 3
SMOKE_CELLS = 1
SMOKE_BUDGET = Decimal("0.50")


def smoke_experiment(experiment: ExperimentConfig) -> ExperimentConfig:
    """One round, small caps, everything else unchanged.

    A smoke run is a check that live providers answer and that a cost lands in
    the ledger. It is not a measurement, and its round is too small to be one.
    """
    return experiment.model_copy(
        update={
            "rounds": SMOKE_ROUNDS,
            "candidates_per_round": SMOKE_CANDIDATES,
            "cells_per_round": SMOKE_CELLS,
            "budget_usd": SMOKE_BUDGET,
        }
    )


def config_path(name: str, directory: Path = EXPERIMENTS_DIR) -> Path:
    return directory / f"{name}.yaml"


def load_experiment(name: str, directory: Path = EXPERIMENTS_DIR) -> ExperimentConfig:
    """Read one committed experiment config by name."""
    path = config_path(name, directory)
    if not path.exists():
        available = ", ".join(experiment_names(directory)) or "none"
        raise FileNotFoundError(f"no experiment config at {path}. Available: {available}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(raw)


def experiment_names(directory: Path = EXPERIMENTS_DIR) -> tuple[str, ...]:
    """Every committed experiment, in run order where one is implied."""
    if not directory.exists():
        return ()
    return tuple(sorted(path.stem for path in directory.glob("*.yaml")))


def load_all(directory: Path = EXPERIMENTS_DIR) -> tuple[ExperimentConfig, ...]:
    return tuple(load_experiment(name, directory) for name in experiment_names(directory))
