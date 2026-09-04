"""The defender's prompt must describe the whole schema, or it proposes nothing.

A six-round live run produced ZERO valid configs: every round logged
`kept: 0, rejected: 4` and then `rejected: 8`, with no reason recorded anywhere.
The cause, once the rejections were logged, was systematic rather than flaky —
the prompt documented `pattern_class` and `action` for a heuristic rule and
nothing else, so the model emitted exactly those two fields, while
`HeuristicRule` also requires `name` and `applies_to`. Every branch failed
identically, which is why the counts were exactly 4 and exactly 8.

The fix is the prompt, never the schema: the constrained config space is a design
commitment, and unknown keys stay rejected. These tests hold the prompt to the
schema, and hold the schema where it was.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from crucible.defender.graph import describe_validation_error, field_errors
from crucible.defender.prompts import (
    EXAMPLE_CONFIG,
    build_proposal_prompt,
    schema_digest,
)
from crucible.defender.state import Hypothesis
from crucible.defenses.config import (
    DefenseConfig,
    HeuristicRule,
    InputLayer,
    PatternClass,
    SessionIsolation,
)

#: Verbatim from a live `deepseek-v4-flash` reply, before the fix. Every rule
#: object carries the two fields the old prompt described and neither of the two
#: it did not.
LIVE_REPLY_BEFORE_THE_FIX: dict[str, object] = {
    "context": {"provenance_tags": True, "spotlighting": "delimiter"},
    "input": {
        "heuristic_rules": [
            {"pattern_class": "instruction_like", "action": "reject"},
            {"pattern_class": "role_change", "action": "reject"},
        ]
    },
    "structural": {"require_user_origin_for_privileged": True, "session_isolation": "isolated"},
}


def _prompt() -> str:
    return build_proposal_prompt(
        DefenseConfig.empty(),
        [Hypothesis(cluster_key="c", mechanism="instruction_override", statement="s")],
        candidate_index=0,
        utility_baseline=0.42,
    )


# --------------------------------------------------------------------------- #
# The schema is not loosened
# --------------------------------------------------------------------------- #


def test_the_reply_that_failed_live_is_still_rejected() -> None:
    """The fix is the prompt. A rule without a name is still not a rule."""
    with pytest.raises(ValidationError) as raised:
        DefenseConfig.model_validate(LIVE_REPLY_BEFORE_THE_FIX)

    fields = field_errors(raised.value)
    assert any("heuristic_rules.0.name" in field for field in fields)
    assert any("heuristic_rules.0.applies_to" in field for field in fields)


def test_an_unknown_key_is_still_rejected() -> None:
    """The design commitment: the defender emits config, never new fields."""
    with pytest.raises(ValidationError):
        DefenseConfig.model_validate({"input": {"heuristic_rules": [], "regex": "^foo"}})


def test_a_wrong_enum_spelling_is_still_rejected() -> None:
    with pytest.raises(ValidationError, match="strict"):
        DefenseConfig.model_validate({"structural": {"session_isolation": "isolated"}})


# --------------------------------------------------------------------------- #
# The prompt carries the whole schema, generated
# --------------------------------------------------------------------------- #


def _models() -> list[type[BaseModel]]:
    from crucible.defenses.config import (
        ClassifierConfig,
        ContextLayer,
        OutputLayer,
        PromptLayer,
        StructuralLayer,
        TrustLevels,
    )

    return [
        DefenseConfig,
        InputLayer,
        HeuristicRule,
        ClassifierConfig,
        ContextLayer,
        TrustLevels,
        PromptLayer,
        OutputLayer,
        StructuralLayer,
    ]


@pytest.mark.parametrize("model", _models(), ids=lambda model: model.__name__)
def test_every_field_of_every_layer_appears_in_the_prompt(model: type[BaseModel]) -> None:
    """Generated from the models, so a new field cannot be forgotten here."""
    prompt = _prompt()

    for name in model.model_fields:
        assert name in prompt, f"{model.__name__}.{name} is not described to the defender"


def test_the_required_fields_are_marked_required() -> None:
    """The exact omission that cost six rounds."""
    digest = schema_digest()

    assert "REQUIRED: name, pattern_class, applies_to" in digest
    for field in ("name", "applies_to"):
        assert f"  {field}:" in digest


def test_every_enum_value_is_spelled_out() -> None:
    """The model guessed `isolated`; it had never been shown `strict`."""
    prompt = _prompt()

    for member in SessionIsolation:
        assert f'"{member.value}"' in prompt
    for member in PatternClass:
        assert f'"{member.value}"' in prompt


def test_the_prompt_names_the_constraint_on_a_rule_name() -> None:
    assert "^[a-z0-9_]+$" in schema_digest()


# --------------------------------------------------------------------------- #
# The worked example
# --------------------------------------------------------------------------- #


def test_the_example_config_is_valid() -> None:
    """An invalid example would teach the model to produce invalid configs."""
    reparsed = DefenseConfig.model_validate(json.loads(json.dumps(EXAMPLE_CONFIG.to_dict())))

    assert reparsed == EXAMPLE_CONFIG


def test_the_example_exercises_every_layer() -> None:
    """A minimal example teaches minimally; this one shows each layer in use."""
    assert EXAMPLE_CONFIG.input.heuristic_rules
    assert EXAMPLE_CONFIG.context.provenance_tags
    assert EXAMPLE_CONFIG.prompt.precedence_statement
    assert EXAMPLE_CONFIG.output.schema_enforcement
    assert EXAMPLE_CONFIG.structural.require_user_origin_for_privileged


def test_the_example_rule_carries_the_fields_the_model_omitted() -> None:
    rule = EXAMPLE_CONFIG.input.heuristic_rules[0]

    assert rule.name and rule.applies_to
    assert '"name"' in json.dumps(EXAMPLE_CONFIG.to_dict())
    assert '"applies_to"' in json.dumps(EXAMPLE_CONFIG.to_dict())


def test_the_example_is_production_safe() -> None:
    """It is shown as a model answer, so it must be one."""
    EXAMPLE_CONFIG.assert_production_safe()


def test_the_prompt_still_refuses_to_carry_a_canary() -> None:
    """The generated schema and example must not weaken the canary guard."""
    from crucible.target.canary import assert_no_canaries

    assert_no_canaries(_prompt())


# --------------------------------------------------------------------------- #
# Rejections are diagnosable
# --------------------------------------------------------------------------- #


def test_a_validation_error_is_described_field_by_field() -> None:
    """Twelve rounds of rejections logged no reason at all. Never again."""
    with pytest.raises(ValidationError) as raised:
        DefenseConfig.model_validate(LIVE_REPLY_BEFORE_THE_FIX)

    described = describe_validation_error(raised.value)

    assert "name" in described and "applies_to" in described
    assert "Field required" in described


def test_a_type_error_is_described_too() -> None:
    assert field_errors(TypeError("config is a string")) == ["<root>: config is a string"]


def test_the_rejection_logs_the_reason_and_the_raw_reply(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from crucible.defender.graph import RAW_REPLY_CHARS, Defender

    defender = Defender.__new__(Defender)
    with caplog.at_level("WARNING", logger="crucible.defender.graph"):
        result = defender._materialise({"config": LIVE_REPLY_BEFORE_THE_FIX}, candidate_index=2)

    assert "schema validation failed" in result["rejected"][0].reason  # type: ignore[index]
    assert "applies_to" in result["rejected"][0].reason  # type: ignore[index]
    assert any(record.message == "defender.proposal_rejected" for record in caplog.records)
    record = next(r for r in caplog.records if r.message == "defender.proposal_rejected")
    assert record.levelname == "WARNING"
    assert "heuristic_rules" in getattr(record, "raw", "")
    assert RAW_REPLY_CHARS >= 500, "enough of the reply to see the shape of the JSON"


def test_a_reply_with_no_json_is_recognised_as_unparseable() -> None:
    from crucible.defender.graph import parse_json_object

    assert parse_json_object("I cannot produce that configuration.") is None
