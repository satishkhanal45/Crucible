"""Layer 5 — structural defense.

The strongest layer, and the least discussed. Three switches:

* `tool_allowlist` — which privileged tools exist at all for this config.
* `require_user_origin_for_privileged` — a privileged call must trace to the
  user's own message, not to a retrieved document. This does not try to
  recognise an attack, so payload cleverness does not help.
* `max_tool_calls_per_turn` — a hard cap. An overflow call is *blocked and
  recorded*, never silently dropped, or the trace would understate what the
  model tried to do.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from crucible.defenses.config import CallOrigin, DefenseConfig
from crucible.defenses.provenance import ProvenanceIndex


class ToolDecision(BaseModel):
    """Whether a call is permitted, whether it runs, and why."""

    model_config = ConfigDict(frozen=True)

    origin: CallOrigin
    authorized: bool
    executed: bool
    reason: str | None = None

    @property
    def blocked(self) -> bool:
        return not self.executed


def decide_tool_call(
    name: str,
    arguments: Mapping[str, object],
    *,
    privileged: bool,
    config: DefenseConfig,
    provenance: ProvenanceIndex,
    calls_this_turn: int = 0,
) -> ToolDecision:
    """Authorize one tool call under Layer 5."""
    origin = provenance.trace_call(name, arguments)

    if calls_this_turn >= config.structural.max_tool_calls_per_turn:
        return ToolDecision(
            origin=origin,
            authorized=False,
            executed=False,
            reason=(
                "blocked by the structural defense layer: max_tool_calls_per_turn "
                f"({config.structural.max_tool_calls_per_turn}) reached"
            ),
        )

    authorized = config.authorizes(tool_name=name, privileged=privileged, origin=origin)
    if authorized:
        return ToolDecision(origin=origin, authorized=True, executed=True)

    # An unconfigured structural layer records the call as unauthorized but does
    # not stop it. That is the vulnerable default the loop exists to find.
    if not config.structural.configured:
        return ToolDecision(
            origin=origin,
            authorized=False,
            executed=True,
            reason="structural layer not configured: unauthorized call was not blocked",
        )

    if privileged and name not in config.structural.tool_allowlist:
        reason = f"blocked by the structural defense layer: {name} is not on the tool allowlist"
    else:
        reason = (
            "blocked by the structural defense layer: a privileged call must trace to "
            f"user input, and this one traced to {origin.value}"
        )
    return ToolDecision(origin=origin, authorized=False, executed=False, reason=reason)
