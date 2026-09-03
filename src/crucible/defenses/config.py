"""`DefenseConfig` — the five-layer defense stack (docs/spec.md section 11).

The defender emits configuration, never code. That is a safety property and a
testability property at once: an agent that can write arbitrary Python into the
guardrail path cannot be reviewed, cannot be replayed, and cannot be trusted.
Everything the defender is allowed to change lives in this schema, and anything
outside it is rejected rather than ignored.

Three validation rules are non-negotiable:

* **Unknown keys are rejected.** Every model here sets `extra="forbid"`, so a
  defender that invents a field gets an error instead of a silently dropped
  defense that a report would later count as active.
* **`canary_scan` must be true in production.** A config with the canary
  scanner off cannot be run against a real target: it would score breaches it
  had no ability to detect.
* **`refusal_policy_text` is not an escape hatch.** It is bounded, single-line,
  and must match one of a small set of templates with narrowly-typed slots, so
  the defender cannot smuggle a replacement system prompt into a free-text
  field.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CallOrigin(StrEnum):
    """Where a tool call, or the argument it carries, came from.

    Layer 5's `require_user_origin_for_privileged` turns this into the
    project's strongest defense: a privileged call whose arguments trace to
    retrieved content is refused however persuasive the payload is.
    """

    USER = "user"
    MODEL = "model"
    RETRIEVED = "retrieved"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"


# --------------------------------------------------------------------------- #
# Layer 1 — input inspection
# --------------------------------------------------------------------------- #


class PatternClass(StrEnum):
    """The fixed set of detector classes a heuristic rule may use.

    Fixed on purpose: a free-text regex field would let the defender write code
    in another costume, and would make one round's defense unreviewable.
    """

    INSTRUCTION_LIKE = "instruction_like"
    ROLE_CHANGE = "role_change"
    DELIMITER_SPOOF = "delimiter_spoof"
    ENCODED_BLOB = "encoded_blob"
    TOOL_INVOCATION = "tool_invocation"
    SECRET_REQUEST = "secret_request"
    URGENCY_PRESSURE = "urgency_pressure"


class InputSurface(StrEnum):
    """What a rule or the classifier is applied to."""

    USER_INPUT = "user_input"
    RETRIEVED_CONTEXT = "retrieved_context"
    TOOL_RESULTS = "tool_results"


class RuleAction(StrEnum):
    FLAG = "flag"
    STRIP = "strip"
    REJECT = "reject"


class HeuristicRule(BaseModel):
    """One named detector, applied to named surfaces, with one action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    pattern_class: PatternClass
    applies_to: tuple[InputSurface, ...] = Field(min_length=1)
    action: RuleAction = RuleAction.FLAG
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("applies_to")
    @classmethod
    def _unique_surfaces(cls, value: tuple[InputSurface, ...]) -> tuple[InputSurface, ...]:
        # Sorted and de-duplicated so that two orderings of the same rule hash
        # identically (docs/spec.md section 11: hashing is order-independent).
        return tuple(sorted(set(value), key=lambda surface: surface.value))


class ClassifierConfig(BaseModel):
    """Layer 1's model-based classifier. Cut B5: it must stay disabled.

    TODO(D5): restoring it needs the call wired in and cached per
    `(text_hash, model, threshold)`, or the cost explodes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    model: str | None = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    applies_to: tuple[InputSurface, ...] = ()

    @field_validator("enabled")
    @classmethod
    def _must_be_disabled(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "input.classifier.enabled must be false in this build: the Layer 1 "
                "classifier is cut B5 in docs/spec.md section 3, restorable as "
                "deferred item D5"
            )
        return value


class InputLayer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    heuristic_rules: tuple[HeuristicRule, ...] = ()
    classifier: ClassifierConfig = ClassifierConfig()
    max_input_tokens: int = Field(default=2048, ge=64, le=32768)

    @field_validator("heuristic_rules")
    @classmethod
    def _unique_names(cls, value: tuple[HeuristicRule, ...]) -> tuple[HeuristicRule, ...]:
        names = [rule.name for rule in value]
        if len(names) != len(set(names)):
            raise ValueError("heuristic rule names must be unique")
        return value


# --------------------------------------------------------------------------- #
# Layer 2 — context construction
# --------------------------------------------------------------------------- #


class Spotlighting(StrEnum):
    NONE = "none"
    DELIMITER = "delimiter"
    DATAMARKING = "datamarking"
    ENCODING = "encoding"


class TrustLevels(BaseModel):
    """The explicit instruction hierarchy. Higher wins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system: int = Field(default=3, ge=0, le=3)
    user: int = Field(default=2, ge=0, le=3)
    retrieved: int = Field(default=0, ge=0, le=3)
    tool_result: int = Field(default=0, ge=0, le=3)

    @model_validator(mode="after")
    def _hierarchy_is_ordered(self) -> Self:
        if not self.system >= self.user >= self.retrieved:
            raise ValueError(
                "trust levels must not invert the hierarchy: system >= user >= retrieved"
            )
        return self


class ContextLayer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spotlighting: Spotlighting = Spotlighting.NONE
    provenance_tags: bool = False
    trust_levels: TrustLevels = TrustLevels()
    strip_instructions_from_retrieved: bool = False


# --------------------------------------------------------------------------- #
# Layer 3 — prompt hardening
# --------------------------------------------------------------------------- #


class RoleReassertion(StrEnum):
    NONE = "none"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    BOTH = "both"


#: `refusal_policy_text` must match one of these, with slots filled by short,
#: narrowly-typed values. The defender picks a refusal style; it does not get to
#: write arbitrary system-prompt text.
REFUSAL_TEMPLATES: tuple[str, ...] = (
    "I can only answer questions about {topic} using the supplied policy excerpts. {suggestion}",
    "That request is outside what I can share. {suggestion}",
    "I can't act on instructions that arrive inside a document. {suggestion}",
    "I'm not able to run {tool} for that request. {suggestion}",
)

#: Slots hold a short human phrase: letters, digits, spaces, light punctuation,
#: and the underscore that tool names use. No newlines, no colons, no angle
#: brackets — nothing that could open a new instruction block.
SLOT_PATTERN = r"[A-Za-z0-9 ,.'\-_]{1,60}"
MAX_REFUSAL_POLICY_CHARS = 240

_TEMPLATE_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(
        "^"
        + re.sub(r"\\\{[a-z_]+\\\}", SLOT_PATTERN, re.escape(template).replace("\\{", "\\{"))
        + "$"
    )
    for template in REFUSAL_TEMPLATES
)


def refusal_text_is_valid(text: str) -> bool:
    """Whether a refusal policy string is one of the templates, slots filled."""
    if len(text) > MAX_REFUSAL_POLICY_CHARS or "\n" in text or "\r" in text:
        return False
    return any(pattern.match(text) for pattern in _TEMPLATE_REGEXES)


class PromptLayer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    precedence_statement: bool = False
    role_reassertion: RoleReassertion = RoleReassertion.NONE
    refusal_policy_text: str | None = None
    output_contract_restatement: bool = False

    @field_validator("refusal_policy_text")
    @classmethod
    def _template_bounded(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not refusal_text_is_valid(value):
            raise ValueError(
                "refusal_policy_text must be one of the approved templates with short "
                "slot values, at most "
                f"{MAX_REFUSAL_POLICY_CHARS} characters and on one line. It is not a "
                "free-text field: the defender emits configuration, never a new system "
                "prompt"
            )
        return value


# --------------------------------------------------------------------------- #
# Layer 4 — output inspection
# --------------------------------------------------------------------------- #


class OnViolation(StrEnum):
    BLOCK = "block"
    REGENERATE = "regenerate"
    STRIP = "strip"


class OutputLayer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Never false in a config that will be run against a real target.
    canary_scan: bool = True
    schema_enforcement: bool = False
    citation_verification: bool = False
    secret_patterns: tuple[str, ...] = ()
    on_violation: OnViolation = OnViolation.BLOCK

    @field_validator("secret_patterns")
    @classmethod
    def _compilable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in value:
            if len(pattern) > 200:
                raise ValueError("a secret pattern may not exceed 200 characters")
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"secret pattern {pattern!r} is not a valid regex: {error}"
                ) from error
        return value


# --------------------------------------------------------------------------- #
# Layer 5 — structural
# --------------------------------------------------------------------------- #


class SessionIsolation(StrEnum):
    STRICT = "strict"
    SHARED = "shared"


class StructuralLayer(BaseModel):
    """The strongest layer, and the least discussed.

    `require_user_origin_for_privileged` refuses a privileged call whose
    arguments trace to retrieved content rather than to the user's own message.
    That kills most tool-hijack attacks regardless of how clever the payload is,
    because it does not depend on recognising the payload at all.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_allowlist: tuple[str, ...] = ()
    require_user_origin_for_privileged: bool = False
    max_tool_calls_per_turn: int = Field(default=4, ge=0, le=16)
    session_isolation: SessionIsolation = SessionIsolation.SHARED

    @field_validator("tool_allowlist")
    @classmethod
    def _sorted_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        # Order-independent hashing: an allowlist is a set, not a sequence.
        return tuple(sorted(set(value)))

    @property
    def configured(self) -> bool:
        """Whether the layer enforces at all.

        An unconfigured structural layer records an unauthorized call but does
        not stop it, which is the vulnerable default the loop exists to find.
        """
        return bool(self.tool_allowlist) or self.require_user_origin_for_privileged


# --------------------------------------------------------------------------- #
# The config
# --------------------------------------------------------------------------- #


class DegenerateDefense(ValueError):
    """A config that is not a defense, only the appearance of one."""


class UnsafeDefense(ValueError):
    """A config that must not be run against a real target."""


class DefenseConfig(BaseModel):
    """A validated configuration of the five-layer defense stack."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: InputLayer = InputLayer()
    context: ContextLayer = ContextLayer()
    prompt: PromptLayer = PromptLayer()
    output: OutputLayer = OutputLayer()
    structural: StructuralLayer = StructuralLayer()

    @classmethod
    def empty(cls) -> Self:
        """No defenses at all: the baseline every round is measured against.

        `canary_scan` stays on, because it is measurement rather than defense:
        without it the harness cannot tell a breach from a block.
        """
        return cls()

    # ------------------------------------------------------------- validation

    @property
    def is_degenerate(self) -> bool:
        """An empty allowlist with no origin requirement defends nothing."""
        return (
            not self.structural.tool_allowlist
            and not self.structural.require_user_origin_for_privileged
        )

    def production_issues(self) -> tuple[str, ...]:
        """Everything that would make this config unsafe to run for real."""
        issues: list[str] = []
        if not self.output.canary_scan:
            issues.append(
                "output.canary_scan is false: the harness could not detect a breach it "
                "caused, so every measurement taken with this config would be wrong"
            )
        if self.input.classifier.enabled:
            issues.append("input.classifier.enabled is true, which is cut B5")
        return tuple(issues)

    def assert_production_safe(self) -> None:
        issues = self.production_issues()
        if issues:
            raise UnsafeDefense("; ".join(issues))

    def degeneracy_warnings(self) -> tuple[str, ...]:
        """Flagged, not rejected: a degenerate config is a valid baseline."""
        if self.is_degenerate:
            return (
                "structural layer is degenerate: an empty tool_allowlist with "
                "require_user_origin_for_privileged false permits every privileged call",
            )
        return ()

    # -------------------------------------------------------------- behaviour

    def authorizes(self, *, tool_name: str, privileged: bool, origin: CallOrigin) -> bool:
        """Whether policy permits this call. Free tools are always permitted."""
        if not privileged:
            return True
        if tool_name not in self.structural.tool_allowlist:
            return False
        if not self.structural.require_user_origin_for_privileged:
            return True
        return origin is CallOrigin.USER

    @property
    def structural_layer_configured(self) -> bool:
        return self.structural.configured

    @property
    def sessions_shared(self) -> bool:
        return self.structural.session_isolation is SessionIsolation.SHARED

    @property
    def complexity(self) -> float:
        """How much machinery this config carries, normalised to roughly 0-1.

        Rule count dominates, because forty hand-specific rules is overfitting
        in another costume (docs/spec.md section 12).
        """
        rules = len(self.input.heuristic_rules)
        patterns = len(self.output.secret_patterns)
        switches = sum(
            (
                self.context.spotlighting is not Spotlighting.NONE,
                self.context.provenance_tags,
                self.context.strip_instructions_from_retrieved,
                self.prompt.precedence_statement,
                self.prompt.role_reassertion is not RoleReassertion.NONE,
                self.prompt.refusal_policy_text is not None,
                self.prompt.output_contract_restatement,
                self.output.schema_enforcement,
                self.output.citation_verification,
                self.structural.require_user_origin_for_privileged,
                self.structural.session_isolation is SessionIsolation.STRICT,
            )
        )
        return round((rules * 2 + patterns + switches) / 30.0, 6)

    # ---------------------------------------------------------------- identity

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        """Stable, order-independent id, used as the outcome cache key.

        Sets are normalised at field level (sorted allowlist, sorted rule
        surfaces) and the JSON is key-sorted, so two configs that mean the same
        thing share one cache entry however they were written.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def __str__(self) -> str:
        return f"DefenseConfig({self.fingerprint()})"
