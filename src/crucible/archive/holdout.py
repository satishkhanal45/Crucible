"""Holdout isolation at the agent boundary.

The repository layer filters `is_holdout = false` in SQL, and no agent may hold
a repository. This is the second lock: whatever an agent is handed is checked
again before it can reach a prompt. The holdout set is the only honest
generalization number in the project, and a leak into either agent destroys it
silently.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable


class HoldoutLeak(RuntimeError):
    """A holdout attack was about to reach an agent prompt."""


@runtime_checkable
class HasHoldoutFlag(Protocol):
    @property
    def is_holdout(self) -> bool: ...


def assert_no_holdout(items: Iterable[object], *, where: str = "an agent prompt") -> None:
    """Fail loudly if any item is a holdout attack.

    The error names ids, never payloads: an error message is a place text goes to
    be read and copied, which is exactly what must not happen here.
    """
    leaked: list[str] = []
    for item in items:
        holdout = getattr(item, "is_holdout", None)
        if holdout is True:
            identifier = getattr(item, "attack_id", None) or getattr(item, "id", "<unknown>")
            leaked.append(str(identifier))
    if leaked:
        raise HoldoutLeak(
            f"holdout attacks reached {where}: {', '.join(leaked)}. The holdout set is "
            "the only honest generalization number in the project; showing it to an "
            "agent destroys it"
        )
