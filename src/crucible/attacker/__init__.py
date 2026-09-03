"""The attacker agent: a LangGraph search over the archive's mutation pool."""

from crucible.attacker.graph import Attacker, NoveltyGate, critique
from crucible.attacker.llm import (
    AttackerLLM,
    MeteredAttackerLLM,
    ScriptedAttackerLLM,
)
from crucible.attacker.operators import (
    BINARY_OPERATORS,
    Encoding,
    MutationOperator,
    apply,
    compose,
    decode,
    encode,
    escalate,
    generalize,
    obfuscate,
    recombine,
    transpose_vector,
)
from crucible.attacker.prompts import build_defense_summary
from crucible.attacker.state import (
    STATUS_BUDGET_EXCEEDED,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    AttackerMode,
    AttackerSettings,
    AttackerState,
    OutcomeSummary,
    RejectionRecord,
)

__all__ = [
    "BINARY_OPERATORS",
    "STATUS_BUDGET_EXCEEDED",
    "STATUS_INSUFFICIENT",
    "STATUS_OK",
    "Attacker",
    "AttackerLLM",
    "AttackerMode",
    "AttackerSettings",
    "AttackerState",
    "Encoding",
    "MeteredAttackerLLM",
    "MutationOperator",
    "NoveltyGate",
    "OutcomeSummary",
    "RejectionRecord",
    "ScriptedAttackerLLM",
    "apply",
    "build_defense_summary",
    "compose",
    "critique",
    "decode",
    "encode",
    "escalate",
    "generalize",
    "obfuscate",
    "recombine",
    "transpose_vector",
]
