"""The OpenAI tool-calling message contract, enforced on the way into a client.

A live DeepSeek smoke run died with

    400 ... messages[5]: missing field `tool_call_id`

after 20-odd successful calls. The reference target was appending tool results
without the id of the call they answered, and was not replaying the assistant
turn that requested them at all. Groq accepted that; DeepSeek validates it.

The bug survived 1008 tests because `ScriptedTargetLLM` accepted whatever it was
handed. That is the same failure mode as the four bugs before it, so the tests
here come in two halves:

  * the target assembles a correctly paired conversation, over more than one hop;
  * the scripted client REJECTS a malformed one, the way a provider does, so the
    next shape bug fails in CI instead of 20 calls into a live run.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from crucible.config import LLMProvider
from crucible.target.adapter import ToolSpec
from crucible.target.reference.llm import (
    ChatCompletionsLLM,
    LLMMessage,
    MalformedConversation,
    RequestedToolCall,
    ScriptedTargetLLM,
    validate_conversation,
    with_call_ids,
)
from crucible.target.reference.tools import SEND_EMAIL, TOOL_SPECS

MODEL = "test-model"


def _assistant(*calls: RequestedToolCall) -> LLMMessage:
    return LLMMessage(role="assistant", content="", tool_calls=calls)


def _call(name: str, call_id: str) -> RequestedToolCall:
    return RequestedToolCall(name=name, arguments={"to": "a@b.test"}, call_id=call_id)


def _tool(call_id: str | None, *, name: str = SEND_EMAIL) -> LLMMessage:
    return LLMMessage(role="tool", content="ok", name=name, tool_call_id=call_id)


def _conversation() -> list[LLMMessage]:
    """A well-formed two-hop tool conversation."""
    return [
        LLMMessage(role="system", content="you are a helpful assistant"),
        LLMMessage(role="user", content="send it"),
        _assistant(_call(SEND_EMAIL, "call_0")),
        _tool("call_0"),
        _assistant(_call(SEND_EMAIL, "call_1")),
        _tool("call_1"),
    ]


# --------------------------------------------------------------------------- #
# The validator: every rule a provider enforces
# --------------------------------------------------------------------------- #


def test_a_well_formed_two_hop_conversation_validates() -> None:
    validate_conversation(_conversation())


def test_a_tool_message_without_an_id_is_rejected() -> None:
    """The exact live failure, reproduced as a unit test."""
    messages = _conversation()
    messages[3] = _tool(None)

    with pytest.raises(MalformedConversation, match="missing field `tool_call_id`"):
        validate_conversation(messages)


def test_the_rejection_names_the_offending_index_like_a_provider_does() -> None:
    messages = _conversation()
    messages[5] = _tool(None)

    with pytest.raises(MalformedConversation, match=r"messages\[5\]"):
        validate_conversation(messages)


def test_an_id_that_matches_no_assistant_call_is_rejected() -> None:
    messages = _conversation()
    messages[3] = _tool("call_99")

    with pytest.raises(MalformedConversation, match="not requested by any preceding"):
        validate_conversation(messages)


def test_a_tool_result_before_the_call_it_answers_is_rejected() -> None:
    """`preceding` is the load-bearing word: order is part of the contract."""
    messages = [
        LLMMessage(role="user", content="send it"),
        _tool("call_0"),
        _assistant(_call(SEND_EMAIL, "call_0")),
    ]

    with pytest.raises(MalformedConversation, match="not requested by any preceding"):
        validate_conversation(messages)


def test_a_tool_result_with_no_assistant_turn_at_all_is_rejected() -> None:
    """What the reference target actually built: results answering nothing."""
    messages = [
        LLMMessage(role="system", content="s"),
        LLMMessage(role="user", content="u"),
        LLMMessage(role="tool", content="ok", name=SEND_EMAIL),
    ]

    with pytest.raises(MalformedConversation, match=r"messages\[2\]"):
        validate_conversation(messages)


def test_an_assistant_call_without_an_id_is_rejected() -> None:
    messages = [
        LLMMessage(role="user", content="u"),
        _assistant(RequestedToolCall(name=SEND_EMAIL)),
    ]

    with pytest.raises(MalformedConversation, match="has no id"):
        validate_conversation(messages)


def test_answering_the_same_call_twice_is_rejected() -> None:
    messages = [*_conversation(), _tool("call_0")]

    with pytest.raises(MalformedConversation, match="already answered"):
        validate_conversation(messages)


def test_requesting_the_same_id_twice_is_rejected() -> None:
    messages = [
        LLMMessage(role="user", content="u"),
        _assistant(_call(SEND_EMAIL, "call_0")),
        _tool("call_0"),
        _assistant(_call(SEND_EMAIL, "call_0")),
    ]

    with pytest.raises(MalformedConversation, match="already requested"):
        validate_conversation(messages)


def test_only_a_tool_message_may_carry_a_tool_call_id() -> None:
    messages = [LLMMessage(role="user", content="u", tool_call_id="call_0")]

    with pytest.raises(MalformedConversation, match="only a tool message"):
        validate_conversation(messages)


def test_only_an_assistant_turn_may_carry_tool_calls() -> None:
    messages = [LLMMessage(role="user", content="u", tool_calls=(_call(SEND_EMAIL, "call_0"),))]

    with pytest.raises(MalformedConversation, match="only an assistant turn"):
        validate_conversation(messages)


# --------------------------------------------------------------------------- #
# Id assignment across hops
# --------------------------------------------------------------------------- #


def test_ids_are_assigned_when_a_client_supplies_none() -> None:
    assigned = with_call_ids([RequestedToolCall(name=SEND_EMAIL)] * 2, start=0)

    assert [call.call_id for call in assigned] == ["call_0", "call_1"]


def test_a_providers_own_ids_are_kept() -> None:
    """They are what the provider will look for in the next request."""
    assigned = with_call_ids([RequestedToolCall(name=SEND_EMAIL, call_id="abc123")], start=7)

    assert assigned[0].call_id == "abc123"


def test_a_second_hop_does_not_reuse_a_first_hops_ids() -> None:
    """Numbering that restarted per turn would collide on the second hop."""
    first = with_call_ids([RequestedToolCall(name=SEND_EMAIL)] * 2, start=0)
    second = with_call_ids([RequestedToolCall(name=SEND_EMAIL)], start=len(first))

    assert {call.call_id for call in first}.isdisjoint(call.call_id for call in second)


def test_assignment_is_deterministic() -> None:
    """The outcome cache requires an identical exchange for identical inputs."""
    once = with_call_ids([RequestedToolCall(name=SEND_EMAIL)] * 3, start=0)
    twice = with_call_ids([RequestedToolCall(name=SEND_EMAIL)] * 3, start=0)

    assert [call.call_id for call in once] == [call.call_id for call in twice]


# --------------------------------------------------------------------------- #
# The scripted client validates, like a provider
# --------------------------------------------------------------------------- #


async def test_the_scripted_client_rejects_a_missing_tool_call_id() -> None:
    """The important half: without this the next shape bug reaches a live run."""
    messages = [
        LLMMessage(role="system", content="s"),
        LLMMessage(role="user", content="u"),
        LLMMessage(role="tool", content="ok", name=SEND_EMAIL),
    ]

    with pytest.raises(MalformedConversation, match="missing field `tool_call_id`"):
        await ScriptedTargetLLM().complete(messages, TOOL_SPECS)


async def test_the_scripted_client_rejects_a_mismatched_id() -> None:
    messages = [
        LLMMessage(role="user", content="u"),
        _assistant(_call(SEND_EMAIL, "call_0")),
        _tool("call_7"),
    ]

    with pytest.raises(MalformedConversation, match="not requested"):
        await ScriptedTargetLLM().complete(messages, TOOL_SPECS)


async def test_the_scripted_client_accepts_a_well_formed_conversation() -> None:
    reply = await ScriptedTargetLLM().complete(_conversation(), TOOL_SPECS)

    assert reply.text


# --------------------------------------------------------------------------- #
# The wire format
# --------------------------------------------------------------------------- #


async def test_the_live_client_sends_the_tool_calling_fields() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await ChatCompletionsLLM("key", client, model=MODEL, provider=LLMProvider.GROQ).complete(
            _conversation(), TOOL_SPECS
        )

    sent = captured["body"]["messages"]
    assert sent[2]["tool_calls"][0]["id"] == "call_0"
    assert sent[2]["tool_calls"][0]["function"]["name"] == SEND_EMAIL
    assert json.loads(sent[2]["tool_calls"][0]["function"]["arguments"]) == {"to": "a@b.test"}
    assert sent[3]["tool_call_id"] == "call_0"
    assert sent[5]["tool_call_id"] == "call_1"


async def test_the_live_client_omits_the_fields_where_they_do_not_apply() -> None:
    """A strict provider rejects a tool_call_id on a user turn just as readily."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await ChatCompletionsLLM(
            "key", client, model=MODEL, provider=LLMProvider.DEEPSEEK
        ).complete([LLMMessage(role="user", content="hello")], TOOL_SPECS)

    user_turn = captured["body"]["messages"][0]
    assert set(user_turn) == {"role", "content"}


async def test_the_live_client_refuses_to_send_a_malformed_conversation() -> None:
    """Our error, before the request, rather than a provider 400 after it."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        del request
        raise AssertionError("a malformed conversation must not reach the provider")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MalformedConversation):
            await ChatCompletionsLLM(
                "key", client, model=MODEL, provider=LLMProvider.DEEPSEEK
            ).complete([LLMMessage(role="tool", content="ok")], TOOL_SPECS)


def test_a_providers_call_id_is_parsed_back_out() -> None:
    """Dropping it on parse is what left the results unpairable."""
    llm = ChatCompletionsLLM("key", httpx.AsyncClient(), model=MODEL, provider=LLMProvider.GROQ)

    reply = llm._parse(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "function": {"name": SEND_EMAIL, "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )

    assert reply.tool_calls[0].call_id == "call_xyz"


def test_the_tool_specs_are_available_for_these_tests() -> None:
    """Guards the import: the specs are the tools a target really exposes."""
    assert any(isinstance(spec, ToolSpec) and spec.name == SEND_EMAIL for spec in TOOL_SPECS)
