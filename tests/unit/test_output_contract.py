"""Verification test 10: the output contract validator."""

from __future__ import annotations

import pytest

from crucible.target.adapter import OutputContract
from crucible.target.contract import validate_output_contract


def test_accepts_a_well_formed_response() -> None:
    check = validate_output_contract(
        '{"answer": "Laptops refresh every 36 months.", "citations": ["IT-POL-001"]}'
    )

    assert check.valid is True
    assert check.answer == "Laptops refresh every 36 months."
    assert check.citations == ("IT-POL-001",)
    assert check.error is None


def test_accepts_an_empty_citation_list() -> None:
    """Zero citations is contract-valid; whether it is *useful* is Tier 2's call."""
    check = validate_output_contract('{"answer": "I could not find that.", "citations": []}')
    assert check.valid is True
    assert check.citations == ()


def test_accepts_a_fenced_json_block() -> None:
    check = validate_output_contract('```json\n{"answer": "yes", "citations": ["IT-POL-002"]}\n```')
    assert check.valid is True


@pytest.mark.parametrize(
    "text",
    [
        "Laptops are refreshed every 36 months.",
        "",
        "   ",
        "{not json at all",
        "answer: yes\ncitations: none",
        "<answer>yes</answer>",
    ],
)
def test_rejects_non_json(text: str) -> None:
    check = validate_output_contract(text)

    assert check.valid is False
    assert check.error is not None


@pytest.mark.parametrize("text", ["[1, 2, 3]", '"a string"', "42", "null", "true"])
def test_rejects_json_that_is_not_an_object(text: str) -> None:
    check = validate_output_contract(text)

    assert check.valid is False
    assert check.error == "response is not a JSON object"


def test_rejects_json_missing_citations() -> None:
    check = validate_output_contract('{"answer": "Every 36 months."}')

    assert check.valid is False
    assert check.error is not None
    assert "citations" in check.error


def test_rejects_json_missing_the_answer() -> None:
    check = validate_output_contract('{"citations": ["IT-POL-001"]}')

    assert check.valid is False
    assert check.error is not None
    assert "answer" in check.error


@pytest.mark.parametrize("citations", ['"IT-POL-001"', "[1, 2]", '{"0": "IT-POL-001"}', "null"])
def test_rejects_citations_that_are_not_a_list_of_strings(citations: str) -> None:
    check = validate_output_contract(f'{{"answer": "x", "citations": {citations}}}')

    assert check.valid is False
    assert check.error == "'citations' must be a list of strings"


def test_rejects_a_non_string_answer() -> None:
    check = validate_output_contract('{"answer": {"text": "x"}, "citations": []}')

    assert check.valid is False
    assert check.error == "'answer' must be a string"


def test_a_custom_contract_can_require_more_keys() -> None:
    contract = OutputContract(required_keys=("answer", "citations", "confidence"))

    assert validate_output_contract('{"answer": "x", "citations": []}', contract).valid is False
    assert (
        validate_output_contract(
            '{"answer": "x", "citations": [], "confidence": "high"}', contract
        ).valid
        is True
    )
