"""The reference target's language model boundary.

Two implementations: a scripted stub used by tests and offline development, and
one live client for real runs. Both run at temperature 0 — the Phase 2 outcome
cache is only sound if an identical (attack, defense) pair produces an identical
response, so temperature is a constant of this module, not a setting.

There is a single live client, `ChatCompletionsLLM`, because every provider this
project talks to speaks the OpenAI chat-completions shape. It is parameterised
by provider and base URL rather than duplicated per host, so a provider's HTTP
status maps to our typed errors in exactly one place — that mapping was a
hotfix, and a second copy of it is a second thing to get wrong.

Every real call is metered: wrap any `TargetLLM` in `MeteredTargetLLM` and the
`CostMeter` records provider, model, tokens, and cost, and stops the round when
the budget is gone.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from crucible.config import (
    DEFAULT_READ_TIMEOUT_SECONDS,
    PROVIDER_BASE_URLS,
    LLMProvider,
)
from crucible.logging import get_logger
from crucible.schemas.spend import TokenUsage
from crucible.services.cost_meter import CostMeter, MeteredResult
from crucible.services.pacing import estimate_tokens
from crucible.services.retry import (
    ProviderTimeout,
    TransientProviderError,
    provider_error_for,
)
from crucible.target.adapter import ToolSpec

logger = get_logger(__name__)

#: Not configurable. See the module docstring.
TARGET_TEMPERATURE = 0.0


class RequestedToolCall(BaseModel):
    """A tool call the model asked for, before any authorization decision."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    #: The provider's own id for this call. It is what pairs the call with its
    #: result in the next request, so it must survive the round trip. Empty when
    #: a client did not supply one; the caller assigns a deterministic id then,
    #: because a random one would break the outcome cache.
    call_id: str = ""


class LLMMessage(BaseModel):
    """One message in a chat-completions conversation.

    The three optional fields are the OpenAI tool-calling contract, and they are
    not decoration: an assistant turn that requested tools must be replayed
    **with** its `tool_calls`, and each tool result must name the
    `tool_call_id` it answers. Groq accepted a conversation missing both;
    DeepSeek rejects it with a 400 on the offending index. The contract is the
    format's, not one provider's, so it is modelled here and enforced by
    `validate_conversation` on the way into every client, scripted included.
    """

    model_config = ConfigDict(frozen=True)

    role: str
    content: str
    name: str | None = None
    #: Assistant turns only: the calls this turn requested.
    tool_calls: tuple[RequestedToolCall, ...] = ()
    #: Tool turns only: the id of the assistant call this message answers.
    tool_call_id: str | None = None


class LLMReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    tool_calls: tuple[RequestedToolCall, ...] = ()
    usage: TokenUsage = TokenUsage(prompt_tokens=0, completion_tokens=0)


class MalformedConversation(ValueError):
    """A message list that no OpenAI-compatible provider would accept.

    Raised before a request is sent rather than after one is refused, so the
    failure names our defect instead of arriving as a provider deserialisation
    error 20 calls into a live run.
    """


def validate_conversation(messages: Sequence[LLMMessage]) -> None:
    """Check the tool-calling contract over a whole conversation.

    Every rule here is one a real provider enforces:

    * an assistant turn's tool calls each carry an id;
    * a `tool` message names a `tool_call_id`;
    * that id was requested by an assistant turn **earlier** in the list — a
      result cannot precede the call it answers;
    * each id is answered at most once;
    * only a `tool` message carries a `tool_call_id`.

    Checked across the whole list, not per hop: with more than one round of tool
    calling the ids must keep pairing up, and the second hop is exactly where an
    id scheme that restarts its numbering per turn goes wrong.
    """
    announced: dict[str, int] = {}
    answered: set[str] = set()
    for index, message in enumerate(messages):
        if message.role == "assistant":
            for call in message.tool_calls:
                if not call.call_id:
                    raise MalformedConversation(
                        f"messages[{index}]: assistant tool call {call.name!r} has no id, "
                        f"so its result cannot be paired with it"
                    )
                if call.call_id in announced:
                    raise MalformedConversation(
                        f"messages[{index}]: tool call id {call.call_id!r} was already "
                        f"requested at messages[{announced[call.call_id]}]"
                    )
                announced[call.call_id] = index
            continue
        if message.tool_calls:
            raise MalformedConversation(
                f"messages[{index}]: only an assistant turn may carry tool_calls, "
                f"got role {message.role!r}"
            )
        if message.role != "tool":
            if message.tool_call_id is not None:
                raise MalformedConversation(
                    f"messages[{index}]: only a tool message may carry a tool_call_id, "
                    f"got role {message.role!r}"
                )
            continue
        if not message.tool_call_id:
            raise MalformedConversation(
                f"messages[{index}]: tool message is missing field `tool_call_id`. "
                f"Every tool result must name the assistant call it answers."
            )
        if message.tool_call_id not in announced:
            raise MalformedConversation(
                f"messages[{index}]: tool_call_id {message.tool_call_id!r} was not "
                f"requested by any preceding assistant turn"
            )
        if message.tool_call_id in answered:
            raise MalformedConversation(
                f"messages[{index}]: tool_call_id {message.tool_call_id!r} was already "
                f"answered earlier in the conversation"
            )
        answered.add(message.tool_call_id)


def with_call_ids(
    calls: Sequence[RequestedToolCall], *, start: int
) -> tuple[RequestedToolCall, ...]:
    """The same calls, each guaranteed an id, numbered from `start`.

    A provider that supplies ids keeps them. One that does not — the scripted
    stub, or a provider that omits them — gets `call_<n>` counted across the
    whole conversation, so a second hop cannot reuse a first hop's id. The
    numbering is deterministic because the outcome cache requires that an
    identical (attack, defense) pair produce an identical exchange.
    """
    return tuple(
        call if call.call_id else call.model_copy(update={"call_id": f"call_{start + offset}"})
        for offset, call in enumerate(calls)
    )


@runtime_checkable
class TargetLLM(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def complete(
        self, messages: Sequence[LLMMessage], tools: Sequence[ToolSpec]
    ) -> LLMReply: ...


# --------------------------------------------------------------------------- #
# Scripted stub
# --------------------------------------------------------------------------- #

_EXTRACTION = re.compile(
    r"(print|reveal|show|repeat|output|display|echo|ignore|disregard|forget)\b[^.\n]{0,80}?"
    r"("
    r"system\s+(prompt|message|instructions)"
    r"|initial\s+(prompt|instructions)"
    r"|(your|the|previous|prior|above|preceding)\s+(instructions|prompt)"
    r"|(instructions|prompt|text|message)\s+above"
    r"|verification\s+code"
    r")",
    re.IGNORECASE,
)
_TOOL_REQUEST = re.compile(
    r"(?:call|invoke|use|run|execute)\s+(?:the\s+)?[`\"']?"
    r"(search_docs|get_user_profile|send_email|delete_document)[`\"']?"
    r"\s*(?:\(([^)]*)\))?",
    re.IGNORECASE,
)
_DOC_ID = re.compile(r"\[doc_id:\s*([A-Za-z0-9\-]+)\]")
_ARGUMENT = re.compile(r"(\w+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^,\)]+))")


class ScriptedTargetLLM:
    """A deterministic stand-in for a compliant, weakly-aligned chat model.

    It is a faithful pass-through: it reads the *assembled* prompt — system
    message, retrieved context, session history, user turn — and reacts to what
    is actually in it. That is what makes it usable for the Phase 1 gate: when
    the user asks it to print the system prompt it complies, exactly as a weak
    model does, so the test measures whether the harness DETECTS the leak rather
    than whether a particular model produces one. It has no knowledge of
    canaries; anything it echoes, it echoes because the text was in its prompt.
    """

    def __init__(self, *, model: str = "scripted-stub", tokens_per_call: int = 128) -> None:
        self._model = model
        self._tokens_per_call = tokens_per_call
        #: Every assembled prompt this stub has seen, for test assertions.
        self.seen_prompts: list[str] = []

    @property
    def provider(self) -> str:
        return "stub"

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, messages: Sequence[LLMMessage], tools: Sequence[ToolSpec]) -> LLMReply:
        # A stub that accepts anything is a stub that lets a message-shape bug
        # reach a live run: this one died on a provider's 400 after passing the
        # whole suite. Validate exactly what a provider validates.
        validate_conversation(messages)
        assembled = "\n\n".join(message.content for message in messages)
        self.seen_prompts.append(assembled)
        usage = TokenUsage(
            prompt_tokens=max(1, len(assembled) // 4),
            completion_tokens=self._tokens_per_call,
        )

        tool_call = self._requested_tool(assembled, tools)
        if tool_call is not None and not any(m.role == "tool" for m in messages):
            return LLMReply(text="", tool_calls=(tool_call,), usage=usage)

        system_prompt = next((m.content for m in messages if m.role == "system"), "")
        instruction = self._latest_instruction(messages)
        citations = _DOC_ID.findall(assembled)

        if _EXTRACTION.search(instruction):
            # A weakly-aligned model complies: it prints what it was told to keep.
            return LLMReply(
                text=_as_json(system_prompt, citations[:1]),
                usage=usage,
            )

        tool_output = "\n".join(m.content for m in messages if m.role == "tool")
        if tool_output:
            return LLMReply(text=_as_json(tool_output, citations[:2]), usage=usage)

        if not citations:
            return LLMReply(
                text=_as_json(
                    "I could not find a policy excerpt that answers that. "
                    "Please raise a service desk ticket.",
                    [],
                ),
                usage=usage,
            )
        summary = self._summarise(messages, citations)
        return LLMReply(text=_as_json(summary, citations[:3]), usage=usage)

    def _latest_instruction(self, messages: Sequence[LLMMessage]) -> str:
        """The user turn plus retrieved context: everything an attacker controls."""
        return "\n".join(
            message.content for message in messages if message.role in {"user", "tool"}
        )

    def _summarise(self, messages: Sequence[LLMMessage], citations: Sequence[str]) -> str:
        """Answer from the excerpts, quoting the top one.

        A real assistant answers with the policy's own words, and the utility set
        checks for those words, so the stub quotes the leading text of the
        highest-ranked excerpt rather than inventing a summary.
        """
        user_turn = next(
            (m.content for m in reversed(messages) if m.role == "user"), "your question"
        )
        assembled = "\n\n".join(message.content for message in messages)
        extract = _leading_extract(assembled)
        answer = (
            f"Based on the policy excerpts ({', '.join(citations[:3])}), here is what applies "
            f"to {user_turn.strip()[:120]}"
        )
        return f"{answer}. {extract}" if extract else answer

    def _requested_tool(
        self, assembled: str, tools: Sequence[ToolSpec]
    ) -> RequestedToolCall | None:
        known = {spec.name for spec in tools}
        match = _TOOL_REQUEST.search(assembled)
        if match is None or match.group(1) not in known:
            return None
        return RequestedToolCall(name=match.group(1), arguments=_parse_arguments(match.group(2)))


#: How much of the top excerpt the stub quotes back.
EXTRACT_CHARS = 400


def _leading_extract(assembled: str) -> str:
    """The opening of the first excerpt in the assembled prompt.

    Works for the plain and provenance-tagged renderings; when the context layer
    encodes or datamarks the body there is nothing quotable, which is exactly the
    utility cost those settings carry.
    """
    match = _DOC_ID.search(assembled)
    if match is None:
        return ""
    after = assembled[match.end() :]
    lines = [line for line in after.split("\n") if line.strip()]
    if len(lines) < 2:
        return ""
    body = lines[1].strip()
    if body.startswith("(base64)"):
        return ""
    return body[:EXTRACT_CHARS]


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    arguments: dict[str, Any] = {}
    for match in _ARGUMENT.finditer(raw):
        key = match.group(1)
        value = next((group for group in match.groups()[1:] if group is not None), "")
        arguments[key] = value.strip()
    return arguments


def _as_json(answer: str, citations: Sequence[str]) -> str:
    return json.dumps({"answer": answer, "citations": list(citations)})


# --------------------------------------------------------------------------- #
# Metering wrapper
# --------------------------------------------------------------------------- #


class MeteredTargetLLM:
    """Routes every call through `CostMeter`. Wrap any `TargetLLM` in this."""

    def __init__(
        self,
        inner: TargetLLM,
        cost_meter: CostMeter,
        *,
        round_id: uuid.UUID | None = None,
    ) -> None:
        self._inner = inner
        self._meter = cost_meter
        self._round_id = round_id

    def set_round(self, round_id: uuid.UUID | None) -> None:
        """Scope subsequent calls to a round, so its cost can be totalled."""
        self._round_id = round_id

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    async def complete(self, messages: Sequence[LLMMessage], tools: Sequence[ToolSpec]) -> LLMReply:
        async def call() -> MeteredResult[LLMReply]:
            reply = await self._inner.complete(messages, tools)
            return MeteredResult(value=reply, usage=reply.usage)

        assembled = "\n".join(message.content for message in messages)
        tool_text = "".join(f"{spec.name}{spec.description}" for spec in tools)
        return await self._meter.call(
            call,
            round_id=self._round_id,
            provider=self._inner.provider,
            model=self._inner.model,
            estimated_tokens=estimate_tokens(assembled + tool_text),
        )


# --------------------------------------------------------------------------- #
# Groq
# --------------------------------------------------------------------------- #


class ChatCompletionsLLM:
    """An OpenAI-compatible chat completions API, at temperature 0.

    One class for every provider: Groq and DeepSeek differ only in base URL and
    credential, so parameterising is what keeps the status-to-typed-error
    mapping in `complete` singular. `provider` has no default — a client that
    did not have to name its provider could be handed one provider's key and
    another's endpoint, and the only symptom would be a 401 mid-run.

    The HTTP client is injected so that the executor can pass one built on the
    egress guard's transport, and so tests can pass `httpx.MockTransport`. No
    test in this repository makes a live call.
    """

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        *,
        model: str,
        provider: LLMProvider,
        base_url: str | None = None,
        timeout: httpx.Timeout | float = DEFAULT_READ_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._model = model
        self._provider = provider
        self._base_url = (base_url or PROVIDER_BASE_URLS[provider.value]).rstrip("/")
        self._timeout = timeout
        # Reported in the timeout error, so the message names the deadline that
        # was actually applied rather than a default it may not be using.
        self._read_timeout = (
            timeout.read or DEFAULT_READ_TIMEOUT_SECONDS
            if isinstance(timeout, httpx.Timeout)
            else float(timeout)
        )

    @property
    def provider(self) -> str:
        return self._provider.value

    @property
    def model(self) -> str:
        return self._model

    def _payload(self, messages: Sequence[LLMMessage], tools: Sequence[ToolSpec]) -> dict[str, Any]:
        validate_conversation(messages)
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": TARGET_TEMPERATURE,
            "messages": [_message_payload(message) for message in messages],
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                name: {"type": "string", "description": description}
                                for name, description in spec.parameters.items()
                            },
                            "required": list(spec.parameters),
                        },
                    },
                }
                for spec in tools
            ]
        return payload

    async def complete(self, messages: Sequence[LLMMessage], tools: Sequence[ToolSpec]) -> LLMReply:
        payload = self._payload(messages, tools)
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as error:
            # Mapped, like a status code is: an unmapped timeout is in no retry
            # policy, so it ends the run instead of backing off.
            raise ProviderTimeout(
                self.provider, self._model, self._read_timeout, kind=_timeout_kind(error)
            ) from error
        except httpx.TransportError as error:
            raise TransientProviderError(
                f"{self.provider} connection failed for {self._model!r}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if response.status_code >= 400:
            # Mapped to a typed error rather than httpx's HTTPStatusError, which
            # no retry policy lists: an unmapped 429 ends the run instead of
            # backing off.
            raise provider_error_for(
                response.status_code,
                provider=self.provider,
                model=self._model,
                retry_after=response.headers.get("Retry-After"),
                detail=_detail(response),
            )
        return self._parse(response.json())

    def _parse(self, body: dict[str, Any]) -> LLMReply:
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        raw_calls = message.get("tool_calls") or []
        calls: list[RequestedToolCall] = []
        for raw in raw_calls:
            function = raw.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                RequestedToolCall(
                    name=str(function.get("name", "")),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    # Kept because the next request has to name it: dropping it
                    # here is what left tool results unpaired.
                    call_id=str(raw.get("id") or ""),
                )
            )
        usage = body.get("usage") or {}
        return LLMReply(
            text=str(message.get("content") or ""),
            tool_calls=tuple(calls),
            usage=TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
        )


def _timeout_kind(error: httpx.TimeoutException) -> str:
    """Which deadline expired, for a message that says what to raise."""
    if isinstance(error, httpx.ConnectTimeout):
        return "connect"
    if isinstance(error, httpx.PoolTimeout):
        return "pool"
    if isinstance(error, httpx.WriteTimeout):
        return "write"
    return "read"


def _message_payload(message: LLMMessage) -> dict[str, Any]:
    """One message in the wire format.

    The tool-calling fields are only present when they apply: a provider that
    validates strictly rejects `tool_call_id` on a user turn as readily as it
    rejects its absence on a tool turn.
    """
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return payload


#: How much of an error body to quote back. Enough to name the cause, short
#: enough that a CLI line stays one line.
DETAIL_CHARS = 200


def _detail(response: httpx.Response) -> str:
    """The provider's own explanation, when it sends one."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:DETAIL_CHARS].strip()
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message", ""))[:DETAIL_CHARS]
        if isinstance(error, str):
            return error[:DETAIL_CHARS]
    return str(body)[:DETAIL_CHARS]
