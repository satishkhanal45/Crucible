"""The defender's model boundary.

Every call goes through `CostMeter`, like every other LLM call in the project.
The scripted client is what the tests use; no test makes a live call.
"""

from __future__ import annotations

import json
import uuid
from typing import Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict

from crucible.config import DEFAULT_READ_TIMEOUT_SECONDS, LLMProvider
from crucible.schemas.spend import TokenUsage
from crucible.services.cost_meter import CostMeter, MeteredResult
from crucible.services.pacing import estimate_tokens
from crucible.target.adapter import ToolSpec
from crucible.target.reference.llm import ChatCompletionsLLM, LLMMessage


class DefenderReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    usage: TokenUsage = TokenUsage(prompt_tokens=0, completion_tokens=0)


@runtime_checkable
class DefenderLLM(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def complete(self, prompt: str) -> DefenderReply: ...


class ScriptedDefenderLLM:
    """Deterministic stand-in. Replies are handed to it in order."""

    def __init__(self, replies: list[str] | None = None, *, model: str = "scripted-defender"):
        self._replies = list(replies or [])
        self._model = model
        #: Every prompt this client has seen, for the isolation assertions.
        self.prompts: list[str] = []

    @property
    def provider(self) -> str:
        return "stub"

    @property
    def model(self) -> str:
        return self._model

    def queue(self, reply: str) -> None:
        self._replies.append(reply)

    async def complete(self, prompt: str) -> DefenderReply:
        self.prompts.append(prompt)
        text = self._replies.pop(0) if self._replies else json.dumps({"rationale": "no change"})
        return DefenderReply(
            text=text,
            usage=TokenUsage(prompt_tokens=max(1, len(prompt) // 4), completion_tokens=64),
        )


class ChatDefenderLLM:
    """Live chat completions, reusing the one OpenAI-shaped client. Temperature 0."""

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        *,
        model: str,
        provider: LLMProvider,
        timeout: httpx.Timeout | float = DEFAULT_READ_TIMEOUT_SECONDS,
    ) -> None:
        self._inner = ChatCompletionsLLM(
            api_key, client, model=model, provider=provider, timeout=timeout
        )

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    async def complete(self, prompt: str) -> DefenderReply:
        reply = await self._inner.complete([LLMMessage(role="user", content=prompt)], ())
        return DefenderReply(text=reply.text, usage=reply.usage)


class MeteredDefenderLLM:
    """Routes every defender call through `CostMeter`."""

    def __init__(
        self, inner: DefenderLLM, cost_meter: CostMeter, *, round_id: uuid.UUID | None = None
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

    @property
    def inner(self) -> DefenderLLM:
        return self._inner

    async def complete(self, prompt: str) -> DefenderReply:
        async def call() -> MeteredResult[DefenderReply]:
            reply = await self._inner.complete(prompt)
            return MeteredResult(value=reply, usage=reply.usage)

        return await self._meter.call(
            call,
            round_id=self._round_id,
            provider=self._inner.provider,
            model=self._inner.model,
            estimated_tokens=estimate_tokens(prompt),
        )


__all__ = [
    "ChatDefenderLLM",
    "DefenderLLM",
    "DefenderReply",
    "MeteredDefenderLLM",
    "ScriptedDefenderLLM",
    "ToolSpec",
]
