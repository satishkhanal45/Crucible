"""Verification test 9 (unit half): authorization of privileged tool calls.

The full path — a query that produces a `ToolCall` in the trace with
`authorized=false` — is exercised in `tests/integration/test_target_harness.py`,
because it needs real retrieval. What is checked here is the policy itself.
"""

from __future__ import annotations

from typing import cast

import pytest

from crucible.defenses.config import CallOrigin, DefenseConfig, StructuralLayer
from crucible.target.canary import CanaryClass, CanarySet, scan_text
from crucible.target.reference.store import DocumentStore
from crucible.target.reference.tools import (
    DELETE_DOCUMENT,
    GET_USER_PROFILE,
    SEARCH_DOCS,
    SEND_EMAIL,
    TOOL_SPECS,
    ToolRuntime,
)


@pytest.fixture
def canaries() -> CanarySet:
    return CanarySet.mint()


@pytest.fixture
def runtime(canaries: CanarySet) -> ToolRuntime:
    """The store is only reached by the two document tools, which this file
    does not exercise; retrieval and deletion are covered in the integration
    tests against a real database."""
    return ToolRuntime(store=cast(DocumentStore, None), canaries=canaries)


def test_the_target_exposes_two_free_and_two_privileged_tools() -> None:
    names = {spec.name for spec in TOOL_SPECS}
    privileged = {spec.name for spec in TOOL_SPECS if spec.privileged}

    assert names == {SEARCH_DOCS, GET_USER_PROFILE, SEND_EMAIL, DELETE_DOCUMENT}
    assert privileged == {SEND_EMAIL, DELETE_DOCUMENT}


@pytest.mark.parametrize("tool", [SEARCH_DOCS, GET_USER_PROFILE])
def test_free_tools_are_always_authorized(tool: str) -> None:
    empty = DefenseConfig.empty()
    assert empty.authorizes(tool_name=tool, privileged=False, origin=CallOrigin.MODEL) is True


@pytest.mark.parametrize("tool", [SEND_EMAIL, DELETE_DOCUMENT])
@pytest.mark.parametrize("origin", list(CallOrigin))
def test_an_empty_config_authorizes_no_privileged_tool(tool: str, origin: CallOrigin) -> None:
    empty = DefenseConfig.empty()
    assert empty.authorizes(tool_name=tool, privileged=True, origin=origin) is False


def test_an_empty_config_does_not_enforce_so_the_call_still_runs() -> None:
    """An app with no structural defenses is the vulnerable case, on purpose."""
    assert DefenseConfig.empty().structural_layer_configured is False


def test_an_allowlisted_privileged_tool_is_authorized() -> None:
    config = DefenseConfig(structural=StructuralLayer(tool_allowlist=(SEND_EMAIL,)))

    assert config.authorizes(tool_name=SEND_EMAIL, privileged=True, origin=CallOrigin.MODEL) is True
    assert (
        config.authorizes(tool_name=DELETE_DOCUMENT, privileged=True, origin=CallOrigin.MODEL)
        is False
    )
    assert config.structural_layer_configured is True


def test_requiring_user_origin_rejects_a_model_originated_call() -> None:
    config = DefenseConfig(
        structural=StructuralLayer(
            tool_allowlist=(SEND_EMAIL,), require_user_origin_for_privileged=True
        )
    )

    assert config.authorizes(tool_name=SEND_EMAIL, privileged=True, origin=CallOrigin.USER) is True
    for origin in (CallOrigin.MODEL, CallOrigin.RETRIEVED, CallOrigin.TOOL_RESULT):
        assert config.authorizes(tool_name=SEND_EMAIL, privileged=True, origin=origin) is False


def test_unknown_configuration_keys_are_rejected_never_ignored() -> None:
    with pytest.raises(ValueError, match="canary_scam"):
        DefenseConfig.model_validate({"canary_scam": True})
    with pytest.raises(ValueError, match=r"extra_forbidden|canary_scam"):
        DefenseConfig.model_validate({"structural": {"canary_scam": True}})


def test_the_fingerprint_is_stable_and_order_independent() -> None:
    first = DefenseConfig(
        structural=StructuralLayer(
            tool_allowlist=(SEND_EMAIL, "delete_document"),
            require_user_origin_for_privileged=True,
        )
    )
    second = DefenseConfig(
        structural=StructuralLayer(
            require_user_origin_for_privileged=True,
            tool_allowlist=("delete_document", SEND_EMAIL),
        )
    )

    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != DefenseConfig.empty().fingerprint()


async def test_only_privileged_tools_return_the_tool_canary(
    runtime: ToolRuntime, canaries: CanarySet
) -> None:
    profile = await runtime.execute(GET_USER_PROFILE, {"user_id": "me"})
    receipt = await runtime.execute(
        SEND_EMAIL, {"to": "ops@example.test", "body": "please action this"}
    )

    assert scan_text(profile, canaries).matches == ()
    assert CanaryClass.TOOL in scan_text(receipt, canaries).breached_classes
    assert runtime.sent_emails == [{"to": "ops@example.test", "body": "please action this"}]
