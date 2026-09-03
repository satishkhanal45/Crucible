"""What each Phase 8 experiment produces.

These are stored verbatim, because the rule for `docs/findings.md` is that every
number is regenerated from stored data. An interpretation is written by a person;
a number is read back from here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LayerAblationRow(BaseModel):
    """One row of the layer-ablation table: the config minus one layer."""

    model_config = ConfigDict(frozen=True)

    layer: str
    config_id: str
    archive_block_rate: float
    utility_pass_rate: float
    archive_size: int
    utility_total: int
    #: Against the full config. Negative means the layer was blocking something.
    delta_block_rate: float = 0.0
    delta_utility: float = 0.0


class LayerAblationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["layer_ablation"] = "layer_ablation"
    rows: tuple[LayerAblationRow, ...] = ()


class TransferResult(BaseModel):
    """The final archive against a second application."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["transfer"] = "transfer"
    persona: str
    #: True when the second target shares Crucible's own harness, which makes
    #: this a weaker claim than transfer to an independently built application.
    within_family: bool = True
    source_config_id: str
    archive_size: int
    baseline_block_rate: float
    hardened_block_rate: float
    baseline_utility: float
    hardened_utility: float

    @property
    def hardening_delta(self) -> float:
        return self.hardened_block_rate - self.baseline_block_rate


class ModelOverlapResult(BaseModel):
    """How much two model families agree about what an attack looks like."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["model_overlap"] = "model_overlap"
    models: tuple[str, ...] = ()
    attacks_per_model: dict[str, int] = Field(default_factory=dict)
    cells_per_model: dict[str, list[str]] = Field(default_factory=dict)
    #: Jaccard over occupied cells. 1.0 means the two explored the same cells.
    cell_overlap: float = 0.0
    #: Mean cosine distance to the nearest attack written by the other model.
    #: 0.0 means they wrote the same attacks; 1.0 means disjoint regions.
    mean_nearest_distance: float = 0.0


class LoopExperimentResult(BaseModel):
    """A pointer to the run a loop experiment produced. Rates live in `rounds`."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["loop"] = "loop"
    run_id: str
    rounds_completed: int
    status: str
    halt_reason: str | None = None


ExperimentResult = LayerAblationResult | TransferResult | ModelOverlapResult | LoopExperimentResult


def as_payload(result: ExperimentResult) -> dict[str, Any]:
    return result.model_dump(mode="json")
