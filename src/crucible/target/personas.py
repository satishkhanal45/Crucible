"""The registry of applications the loop can be pointed at.

Two entries: the original Northwind IT desk, and the Meridian clinic used by the
transfer experiment. `experiments/*.yaml` names one by key, and nothing else in
the loop knows the difference.
"""

from __future__ import annotations

from crucible.target.clinic.persona import MERIDIAN
from crucible.target.persona import TargetPersona
from crucible.target.reference.persona import NORTHWIND

PERSONAS: dict[str, TargetPersona] = {
    NORTHWIND.key: NORTHWIND,
    MERIDIAN.key: MERIDIAN,
}


def persona_for(key: str) -> TargetPersona:
    """Look one up by key, naming the alternatives when the key is wrong."""
    persona = PERSONAS.get(key)
    if persona is None:
        raise KeyError(f"unknown target persona {key!r}. Available: {', '.join(sorted(PERSONAS))}")
    return persona
