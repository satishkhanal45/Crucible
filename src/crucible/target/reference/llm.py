"""The reference target's language model boundary.

Two implementations: a scripted stub used by tests and offline development, and
a Groq client for real runs. Both run at temperature 0 — the Phase 2 outcome
cache is only sound if an identical (attack, defense) pair produces an identical
response, so temperature is a constant of this module, not a setting.

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

from crucible.logging import get_logger
from crucible.schemas.spend import TokenUsage
from crucible.services.cost_meter import CostMeter, MeteredResult
from crucible.services.pacing import estimate_tokens
from crucible.services.retry import provider_error_for
from crucible.target.adapter import ToolSpec

logger = get_logger(__name__)

#: Not configurable. See the module docstring.
TARGET_TEMPERATURE = 0.0

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    content: str
    name: str | None = None


class RequestedToolCall(BaseModel):
    """A tool call the model asked for, before any authorization decision."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    tool_calls: tuple[RequestedToolCall, ...] = ()
    usage: TokenUsage = TokenUsage(prompt_tokens=0, completion_tokens=0)


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


class GroqTargetLLM:
    """Groq's OpenAI-compatible chat completions API, at temperature 0.

    The client is injected so that the executor can pass one built on the egress
    guard's transport, and so tests can pass `httpx.MockTransport`. No test in
    this repository makes a live call.
    """

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        *,
        model: str,
        base_url: str = GROQ_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def provider(self) -> str:
        return "groq"

    @property
    def model(self) -> str:
        return self._model

    def _payload(self, messages: Sequence[LLMMessage], tools: Sequence[ToolSpec]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": TARGET_TEMPERATURE,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
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
        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=self._payload(messages, tools),
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
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
