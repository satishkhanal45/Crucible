"""How much to trust a cell label, per axis.

A live run of the real classifier over the 40 committed seeds agreed with the
hand-assigned labels 39/40 on the objective axis and roughly 22/40 on the
technique axis. Those two numbers mean very different things for a coverage
figure, and a single combined percentage hides that: coverage counted across
objectives is close to measurement, coverage counted across techniques is close
to a guess. So the two axes are reported separately, everywhere, and the
technique axis is labelled a soft metric when it falls below the threshold.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: Below this, an axis is not measurement and must not be presented as such.
#: 0.7 is the point at which nearly a third of cells could be wrong, which is
#: more error than any coverage claim can carry.
SOFT_AXIS_THRESHOLD = 0.7


class AxisAgreement(BaseModel):
    """Agreement on one taxonomy axis: a count, its denominator, its verdict."""

    model_config = ConfigDict(frozen=True)

    axis: str
    agreed: int = Field(ge=0)
    total: int = Field(ge=0)

    @property
    def rate(self) -> float:
        return self.agreed / self.total if self.total else 0.0

    @property
    def is_soft(self) -> bool:
        """True when this axis is too unreliable to report as measurement."""
        return self.rate < SOFT_AXIS_THRESHOLD

    def render(self) -> str:
        return f"{self.axis} {self.agreed}/{self.total} ({self.rate:.0%})"


class ClassifierAgreement(BaseModel):
    """One measured comparison of the real classifier against the seed labels."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    total: int = Field(ge=0)
    objective_agreed: int = Field(ge=0)
    technique_agreed: int = Field(ge=0)
    combined_agreed: int = Field(ge=0)
    unclassified: int = Field(default=0, ge=0)

    @property
    def objective(self) -> AxisAgreement:
        return AxisAgreement(axis="objective", agreed=self.objective_agreed, total=self.total)

    @property
    def technique(self) -> AxisAgreement:
        return AxisAgreement(axis="technique", agreed=self.technique_agreed, total=self.total)

    @property
    def combined(self) -> AxisAgreement:
        return AxisAgreement(axis="combined", agreed=self.combined_agreed, total=self.total)

    @property
    def technique_is_soft(self) -> bool:
        return self.technique.is_soft

    def caption(self) -> str:
        """One line fit for a chart caption or a CLI footer."""
        line = (
            f"classifier agreement vs hand labels ({self.model_name}, n={self.total}): "
            f"{self.objective.render()}, {self.technique.render()}, {self.combined.render()}"
        )
        if self.technique_is_soft:
            line += " -- technique-axis coverage is a SOFT metric, not measurement"
        return line


#: What a report says when no agreement has ever been measured. Silence would
#: read as "trustworthy", which is the one thing it does not mean.
UNMEASURED_CAPTION = (
    "classifier agreement vs hand labels: NOT MEASURED. Run "
    "`crucible archive reclassify --dry-run` before quoting coverage: "
    "cell labels are unverified, so coverage on both axes is a soft metric."
)
