"""Step 7 — the regression check.

This is the mechanism that makes the project a *loop* rather than a sequence of
patches. Without it the defender fixes round 5 by reopening round 2: the attack
success curve looks excellent while the system gets worse, and nothing in the
data says so.

An archived attack that `D(n-1)` blocked and `D(n)` breaches is a regression. It
is named, recorded, and the config is **not auto-accepted**.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from crucible.logging import get_logger
from crucible.loop.reports import Regression
from crucible.schemas.outcome import Outcome

logger = get_logger(__name__)

#: Outcomes that count as "the defense held" for the previous config.
HELD = frozenset({Outcome.BLOCKED, Outcome.REFUSED})


def find_regressions(
    before: Mapping[uuid.UUID, Outcome],
    after: Mapping[uuid.UUID, Outcome],
    *,
    cells: Mapping[uuid.UUID, str | None] | None = None,
) -> tuple[Regression, ...]:
    """Attacks the new config reopened.

    Only attacks evaluated under *both* configs can regress: an attack the
    previous config never saw cannot have been reopened by this one.
    """
    found: list[Regression] = []
    for attack_id, previous in before.items():
        current = after.get(attack_id)
        if current is None:
            continue
        if previous in HELD and current is Outcome.BREACHED:
            found.append(
                Regression(
                    attack_id=attack_id,
                    cell_key=(cells or {}).get(attack_id),
                    previous_outcome=previous.value,
                    new_outcome=current.value,
                )
            )
    found.sort(key=lambda regression: str(regression.attack_id))

    if found:
        logger.warning(
            "loop.regression_detected",
            extra={
                "count": len(found),
                "attack_ids": [str(regression.attack_id) for regression in found],
                "action": "the config was not promoted",
            },
        )
    return tuple(found)
