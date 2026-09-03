"""Hotfix verification: HTTP failures map to the right retry decision.

A live `crucible archive reclassify` run died 23 calls into 40 because a 429
surfaced as `httpx.HTTPStatusError`, which no retry policy lists. These tests
drive the real client over `httpx.MockTransport`, so they exercise the same
code path a live call takes without making one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from decimal import Decimal

import httpx
import pytest

from crucible.schemas.spend import TokenUsage
from crucible.services.cost_meter import BudgetExceeded, CostMeter, MeteredResult
from crucible.services.retry import (
    AuthenticationFailed,
    ModelNotFound,
    ProviderConfigurationError,
    RateLimited,
    RetryPolicy,
    TransientProviderError,
    parse_retry_after,
    retry_async,
)
from crucible.target.reference.llm import GroqTargetLLM, LLMMessage
from crucible.target.reference.tools import TOOL_SPECS
from tests.fixtures.fake_spend import FakeSpendRepository

MODEL = "test-model"

OK_BODY = {
    "choices": [{"message": {"content": '{"answer": "hi", "citations": []}'}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


@asynccontextmanager
async def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        yield client


def _messages() -> list[LLMMessage]:
    return [LLMMessage(role="user", content="hello")]


async def _complete(handler: Callable[[httpx.Request], httpx.Response]) -> object:
    async with _client(handler) as client:
        return await GroqTargetLLM("key", client, model=MODEL).complete(_messages(), TOOL_SPECS)


def _responder(*responses: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    """Serve the given responses in order, repeating the last one forever."""
    remaining = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return handler


def _policy(attempts: int = 3) -> RetryPolicy:
    return RetryPolicy(attempts=attempts, base_delay=0.01, max_delay=0.05)


async def _retried(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    attempts: int = 3,
    slept: list[float] | None = None,
) -> object:
    recorded = slept if slept is not None else []

    async def sleep(seconds: float) -> None:
        recorded.append(seconds)

    async with _client(handler) as client:
        llm = GroqTargetLLM("key", client, model=MODEL)

        async def call() -> object:
            return await llm.complete(_messages(), TOOL_SPECS)

        return await retry_async(call, _policy(attempts), sleep=sleep, jitter=lambda: 0.0)


# --------------------------------------------------------------------------- #
# 429
# --------------------------------------------------------------------------- #


async def test_a_429_then_a_success_is_retried_and_succeeds() -> None:
    slept: list[float] = []
    reply = await _retried(
        _responder(
            httpx.Response(429, json={"error": {"message": "slow down"}}),
            httpx.Response(200, json=OK_BODY),
        ),
        slept=slept,
    )

    assert getattr(reply, "text", "") != ""
    assert len(slept) == 1, "one backoff between the refusal and the success"


async def test_a_429_that_never_clears_raises_after_the_attempt_cap() -> None:
    slept: list[float] = []
    with pytest.raises(RateLimited):
        await _retried(
            _responder(httpx.Response(429, json={"error": {"message": "slow down"}})),
            attempts=3,
            slept=slept,
        )

    assert len(slept) == 2, "three attempts means two waits, not three"


async def test_retry_after_is_honoured_over_the_computed_backoff() -> None:
    slept: list[float] = []
    await _retried(
        _responder(
            httpx.Response(429, headers={"Retry-After": "7"}, json={}),
            httpx.Response(200, json=OK_BODY),
        ),
        slept=slept,
    )

    assert slept == [7.0], "the provider's own number wins over exponential backoff"


@pytest.mark.parametrize(
    ("header", "expected"),
    [("5", 5.0), ("2.5", 2.5), ("0", None), ("-1", None), ("", None), ("nonsense", None)],
)
def test_retry_after_parsing(header: str, expected: float | None) -> None:
    assert parse_retry_after(header) == expected


def test_retry_after_accepts_an_http_date() -> None:
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    future = format_datetime(datetime.now(UTC) + timedelta(seconds=30))
    parsed = parse_retry_after(future)

    assert parsed is not None
    assert 20.0 < parsed <= 30.0


async def test_a_503_is_retried() -> None:
    slept: list[float] = []
    reply = await _retried(
        _responder(httpx.Response(503, text="upstream"), httpx.Response(200, json=OK_BODY)),
        slept=slept,
    )

    assert getattr(reply, "text", "") != ""
    assert len(slept) == 1


# --------------------------------------------------------------------------- #
# Configuration errors, which retrying only wastes quota on
# --------------------------------------------------------------------------- #


async def test_a_401_is_not_retried_and_names_the_key() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

    with pytest.raises(AuthenticationFailed) as raised:
        await _retried(handler)

    assert calls == 1, "an expired key does not become valid on the second try"
    assert "GROQ_API_KEY" in str(raised.value)
    assert "Invalid API Key" in str(raised.value)


async def test_a_404_is_not_retried_and_names_the_model() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        nonlocal calls
        calls += 1
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    with pytest.raises(ModelNotFound) as raised:
        await _retried(handler)

    assert calls == 1
    assert raised.value.model == MODEL
    assert MODEL in str(raised.value)
    assert "decommissioned" in str(raised.value)


def test_configuration_errors_share_a_base_the_cli_can_catch() -> None:
    assert issubclass(AuthenticationFailed, ProviderConfigurationError)
    assert issubclass(ModelNotFound, ProviderConfigurationError)
    assert not issubclass(RateLimited, ProviderConfigurationError)
    assert not issubclass(TransientProviderError, ProviderConfigurationError)


def test_configuration_errors_are_outside_the_default_retry_policy() -> None:
    """The policy is the contract; assert it rather than only the behaviour."""
    policy = RetryPolicy()

    assert not issubclass(AuthenticationFailed, policy.retry_on)
    assert not issubclass(ModelNotFound, policy.retry_on)
    assert issubclass(RateLimited, policy.retry_on)
    assert issubclass(TransientProviderError, policy.retry_on)


async def test_budget_exceeded_is_never_retried() -> None:
    calls = 0

    async def call() -> None:
        nonlocal calls
        calls += 1
        raise BudgetExceeded(None, Decimal("6.00"), Decimal("5.00"))

    with pytest.raises(BudgetExceeded):
        await retry_async(call, _policy())

    assert calls == 1


async def test_the_meter_retries_a_429_and_records_one_spend() -> None:
    """Retries are attempts at one call: the ledger records the call, once."""
    repository = FakeSpendRepository()
    meter = CostMeter(repository, Decimal("1.00"), retry_policy=_policy())
    attempts = 0

    async def call() -> MeteredResult[str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimited("429", retry_after=0.0)
        return MeteredResult(value="ok", usage=TokenUsage(prompt_tokens=10, completion_tokens=5))

    value = await meter.call(call, round_id=None, provider="groq", model=MODEL)

    assert value == "ok"
    assert attempts == 2
    assert len(repository.records) == 1
