"""The attack taxonomy from docs/spec.md section 6.

Six objectives x two executable vectors x eight techniques. The grid, its
denominator, and elite selection are Phase 3; what lives here is the vocabulary
and the rule about which vectors this build can actually execute.
"""

from __future__ import annotations

from enum import StrEnum


class Objective(StrEnum):
    """What the attack is trying to achieve."""

    SYSPROMPT_EXTRACTION = "sysprompt_extraction"
    SCOPE_VIOLATION = "scope_violation"
    TOOL_HIJACK = "tool_hijack"
    FORMAT_SUBVERSION = "format_subversion"
    ROLE_OVERRIDE = "role_override"
    CROSS_SESSION_LEAK = "cross_session_leak"


class DeliveryVector(StrEnum):
    """How the payload reaches the model.

    `multi_turn` and `indirect_tool_result` remain in the enum but are not
    executable in this build (cut B3). They fail validation at attack creation
    rather than being silently accepted or quietly downgraded.
    """

    DIRECT = "direct"
    INDIRECT_DOCUMENT = "indirect_document"
    INDIRECT_TOOL_RESULT = "indirect_tool_result"
    MULTI_TURN = "multi_turn"


class Technique(StrEnum):
    """How the payload is phrased."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    CONTEXT_CONFUSION = "context_confusion"
    ROLE_PLAY_FRAMING = "role_play_framing"
    ENCODING_OBFUSCATION = "encoding_obfuscation"
    DELIMITER_INJECTION = "delimiter_injection"
    AUTHORITY_IMPERSONATION = "authority_impersonation"
    PAYLOAD_SPLITTING = "payload_splitting"
    LANGUAGE_SWITCHING = "language_switching"


#: Vectors this build can execute. The rest are deferred item D3.
EXECUTABLE_VECTORS: frozenset[DeliveryVector] = frozenset(
    {DeliveryVector.DIRECT, DeliveryVector.INDIRECT_DOCUMENT}
)

#: TODO(phase-3): the coverage denominator (6 x 2 x 8 = 96) and the MAP-Elites
#: grid belong to the archive. Every report must print the denominator next to
#: the coverage number.
GRID_DENOMINATOR = len(Objective) * len(EXECUTABLE_VECTORS) * len(Technique)


def cell_key(objective: Objective, vector: DeliveryVector, technique: Technique) -> str:
    """`objective|vector|technique` — the MAP-Elites cell this attack lands in."""
    return f"{objective.value}|{vector.value}|{technique.value}"


def deferred_vector_message(vector: DeliveryVector) -> str:
    """Why a non-executable vector was rejected, naming the deferred item."""
    return (
        f"delivery vector {vector.value!r} is not executable in this build: it is "
        "deferred item D3 in docs/spec.md section 4 (cut B3). Only "
        f"{sorted(v.value for v in EXECUTABLE_VECTORS)} can be executed. To restore it, "
        "the executor needs multi-turn session handling and a tool-result injection "
        "hook on the adapter."
    )


class DeferredVector(ValueError):
    """A delivery vector this build deliberately cannot execute (cut B3, D3)."""


def ensure_executable(vector: DeliveryVector) -> DeliveryVector:
    """Raise if `vector` cannot be executed in this build.

    `Attack` validates the same rule, but pydantic wraps a validator's error in
    a `ValidationError`; call this at a boundary that wants the typed error.
    """
    if vector not in EXECUTABLE_VECTORS:
        raise DeferredVector(deferred_vector_message(vector))
    return vector
