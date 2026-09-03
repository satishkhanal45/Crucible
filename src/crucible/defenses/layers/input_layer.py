"""Layer 1 — input inspection.

Named heuristic rules, each drawn from the fixed detector set, applied to named
surfaces. `flag` records, `strip` removes the matching spans, `reject` refuses
the turn outright.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from crucible.defenses.config import DefenseConfig, InputSurface, RuleAction
from crucible.defenses.detectors import detect, strip_spans


class InputFinding(BaseModel):
    """One rule firing on one surface. Carries no payload text."""

    model_config = ConfigDict(frozen=True)

    rule: str
    pattern_class: str
    surface: InputSurface
    score: float
    action: RuleAction


class InputVerdict(BaseModel):
    """What Layer 1 decided about one piece of text."""

    model_config = ConfigDict(frozen=True)

    text: str
    findings: tuple[InputFinding, ...] = ()
    rejected: bool = False
    reason: str | None = None

    @property
    def modified(self) -> bool:
        return any(finding.action is RuleAction.STRIP for finding in self.findings)


def inspect_input(text: str, surface: InputSurface, config: DefenseConfig) -> InputVerdict:
    """Run every rule that applies to `surface`, in configured order."""
    findings: list[InputFinding] = []
    working = text

    for rule in config.input.heuristic_rules:
        if surface not in rule.applies_to:
            continue
        detection = detect(working, rule.pattern_class)
        if detection.score < rule.threshold or not detection.matched:
            continue

        findings.append(
            InputFinding(
                rule=rule.name,
                pattern_class=rule.pattern_class.value,
                surface=surface,
                score=detection.score,
                action=rule.action,
            )
        )
        if rule.action is RuleAction.REJECT:
            return InputVerdict(
                text=working,
                findings=tuple(findings),
                rejected=True,
                reason=f"input rule {rule.name} rejected this {surface.value}",
            )
        if rule.action is RuleAction.STRIP:
            working = strip_spans(working, detection.spans)

    return InputVerdict(text=working, findings=tuple(findings))
