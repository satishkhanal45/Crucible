"""Layer 4 — output inspection.

A note on `canary_scan`, because the reading matters.

`canary_scan` is **measurement**, not blocking: with it on, the target records
which canary classes appeared in what it emitted, so the oracle's evidence is
complete. That is why docs/spec.md section 11 requires it to be true in any
production config — a config that runs with it off produces breach numbers it
had no ability to detect.

Blocking a leaked secret is a *different* switch: `secret_patterns` plus
`on_violation`. Keeping the two apart is what lets the baseline empty config
still leak (so the loop has something to measure) while a defender that wants to
stop leaks adds a pattern and pays for it in the utility set. If `canary_scan`
itself blocked, the empty baseline would block every attack and the experiment
would measure nothing at all.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache

from pydantic import BaseModel, ConfigDict

from crucible.defenses.config import DefenseConfig, OnViolation
from crucible.target.canary import CanaryClass, CanarySet, scan_text
from crucible.target.contract import validate_output_contract


class OutputViolation(BaseModel):
    """One rule Layer 4 found broken. Never carries the offending text."""

    model_config = ConfigDict(frozen=True)

    check: str
    detail: str


class OutputVerdict(BaseModel):
    """What Layer 4 decided about one response."""

    model_config = ConfigDict(frozen=True)

    text: str
    violations: tuple[OutputViolation, ...] = ()
    #: Canary classes seen in the emitted text. Recorded, never blocked on.
    canary_hits: tuple[CanaryClass, ...] = ()
    blocked: bool = False
    regenerate: bool = False

    @property
    def clean(self) -> bool:
        return not self.violations


@lru_cache(maxsize=64)
def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


def inspect_output(
    text: str,
    config: DefenseConfig,
    *,
    retrieved_doc_ids: Sequence[str] = (),
    canaries: CanarySet | None = None,
) -> OutputVerdict:
    """Check one response against the output layer."""
    layer = config.output
    violations: list[OutputViolation] = []
    working = text

    canary_hits: tuple[CanaryClass, ...] = ()
    if layer.canary_scan and canaries is not None:
        canary_hits = tuple(sorted(scan_text(text, canaries).breached_classes))

    contract = validate_output_contract(text)
    if layer.schema_enforcement and not contract.valid:
        violations.append(
            OutputViolation(check="schema_enforcement", detail=contract.error or "invalid")
        )

    if layer.citation_verification:
        retrieved = set(retrieved_doc_ids)
        fabricated = sorted(set(contract.citations) - retrieved)
        if fabricated:
            violations.append(
                OutputViolation(
                    check="citation_verification",
                    detail=f"cited documents that were not retrieved: {', '.join(fabricated)}",
                )
            )

    matched_patterns: list[re.Pattern[str]] = []
    for pattern in _compiled(layer.secret_patterns):
        if pattern.search(text):
            matched_patterns.append(pattern)
            violations.append(
                OutputViolation(
                    check="secret_patterns", detail=f"output matched {pattern.pattern!r}"
                )
            )

    if not violations:
        return OutputVerdict(text=working, canary_hits=canary_hits)

    if layer.on_violation is OnViolation.STRIP:
        for pattern in matched_patterns:
            working = pattern.sub("[redacted]", working)
        return OutputVerdict(text=working, violations=tuple(violations), canary_hits=canary_hits)

    if layer.on_violation is OnViolation.REGENERATE:
        return OutputVerdict(
            text=working,
            violations=tuple(violations),
            canary_hits=canary_hits,
            regenerate=True,
        )

    return OutputVerdict(
        text=working, violations=tuple(violations), canary_hits=canary_hits, blocked=True
    )
