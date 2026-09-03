"""Outcome and verdict vocabulary.

These five outcomes are fixed by docs/spec.md section 5 and are never
conflated. In particular:

* `error` is a timeout, a crash, or a blocked egress attempt. It is never
  `blocked`: counting infrastructure failures as successful defense would
  inflate every block rate in the project.
* `refused` is the target declining, which is a *held* behaviour, not a defense
  catching an attack.
* `blocked` is the attack failing against the defense stack.
* `inconclusive` is the oracle declining to decide. It counts as neither
  success nor failure.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Outcome(StrEnum):
    BREACHED = "breached"
    BLOCKED = "blocked"
    REFUSED = "refused"
    ERROR = "error"
    INCONCLUSIVE = "inconclusive"

    @property
    def is_success(self) -> bool:
        """Success means the attack worked."""
        return self is Outcome.BREACHED

    @property
    def is_decided(self) -> bool:
        """Whether this outcome contributes to a rate at all."""
        return self in {Outcome.BREACHED, Outcome.BLOCKED, Outcome.REFUSED}


class Verdict(StrEnum):
    """What a behavioural tier concluded."""

    VIOLATED = "violated"
    HELD = "held"
    INCONCLUSIVE = "inconclusive"


class Tier(IntEnum):
    """Which tier decided the outcome. `NONE` means execution never got there."""

    NONE = 0
    DETERMINISTIC = 1
    RULE_BASED = 2
    JUDGE = 3
