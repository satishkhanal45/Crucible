"""Timeouts must not end a run, and pacing must fit the account's tier.

Run df8dfe53 cleared every real blocker — attacks admitted and scored, the
defender triaging breaches into clusters, zero 429s across ~90 calls — and then
died on an `httpx.ReadTimeout` in `propose_one`. A client-side deadline, not a
model failure, and it escaped the retry policy exactly as an unmapped 429 once
did.

Two properties, then:

  1. A timeout is transient: retried with backoff, and when the retries are gone
     it costs one branch, not the experiment. Zero surviving candidates is a
     named status.
  2. The pacer stays — it is what keeps a burst from earning a 429 — but its
     window has to be sized for the account's tier, per provider, from settings.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from decimal import Decimal

import httpx
import pytest

from crucible.config import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ROLE_READ_TIMEOUT_SECONDS,
    ConfigurationError,
    LLMProvider,
    Settings,
    load_settings,
)
from crucible.defender.graph import STATUS_NO_CANDIDATES
from crucible.schemas.spend import TokenUsage
from crucible.services.cost_meter import CostMeter, MeteredResult
from crucible.services.pacing import ProviderPacer
from crucible.services.retry import (
    ProviderTimeout,
    RetryPolicy,
    TransientProviderError,
    retry_async,
)
from crucible.target.reference.llm import ChatCompletionsLLM, LLMMessage
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


def _llm(client: httpx.AsyncClient, *, timeout: httpx.Timeout | float = 60.0) -> ChatCompletionsLLM:
    return ChatCompletionsLLM(
        "key", client, model=MODEL, provider=LLMProvider.DEEPSEEK, timeout=timeout
    )


# --------------------------------------------------------------------------- #
# 1. A timeout is a typed, transient error
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raised", "kind"),
    [
        (httpx.ReadTimeout("too slow"), "read"),
        (httpx.ConnectTimeout("no handshake"), "connect"),
        (httpx.PoolTimeout("no connection free"), "pool"),
    ],
)
async def test_every_timeout_becomes_a_transient_provider_error(
    raised: Exception, kind: str
) -> None:
    """`httpx.ReadTimeout` is in no retry policy; `ProviderTimeout` is."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise raised

    async with _client(handler) as client:
        with pytest.raises(ProviderTimeout) as error:
            await _llm(client).complete(_messages(), TOOL_SPECS)

    assert isinstance(error.value, TransientProviderError)
    assert error.value.kind == kind
    assert MODEL in str(error.value)


async def test_the_timeout_error_names_the_deadline_that_applied() -> None:
    """So the operator raises the right setting, not a guessed one."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ReadTimeout("too slow")

    async with _client(handler) as client:
        with pytest.raises(ProviderTimeout, match="180s"):
            await _llm(client, timeout=httpx.Timeout(180.0, connect=10.0)).complete(
                _messages(), TOOL_SPECS
            )


async def test_a_read_timeout_then_a_success_is_retried() -> None:
    """The property the run needed: back off, do not die."""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("too slow")
        return httpx.Response(200, json=OK_BODY)

    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    async with _client(handler) as client:
        llm = _llm(client)

        async def call() -> object:
            return await llm.complete(_messages(), TOOL_SPECS)

        reply = await retry_async(
            call, RetryPolicy(attempts=3, base_delay=0.01), sleep=sleep, jitter=lambda: 0.0
        )

    assert len(attempts) == 2
    assert slept, "a retried timeout backs off"
    assert getattr(reply, "text", "")


async def test_a_timeout_is_retried_by_the_default_policy() -> None:
    """Not just by a policy a test constructs: by the one the meter uses."""
    assert any(issubclass(ProviderTimeout, kind) for kind in RetryPolicy().retry_on)


async def test_a_transport_failure_is_also_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ConnectError("connection refused")

    async with _client(handler) as client:
        with pytest.raises(TransientProviderError):
            await _llm(client).complete(_messages(), TOOL_SPECS)


async def test_the_cost_meter_retries_a_timeout_and_records_the_call() -> None:
    """Every provider call goes through the meter, so this is the real path."""
    calls: list[int] = []

    async def provider_call() -> MeteredResult[str]:
        calls.append(1)
        if len(calls) == 1:
            raise ProviderTimeout("deepseek", MODEL, 180.0)
        return MeteredResult(value="ok", usage=TokenUsage(prompt_tokens=10, completion_tokens=5))

    repository = FakeSpendRepository()
    meter = CostMeter(
        repository, Decimal("5.00"), retry_policy=RetryPolicy(attempts=3, base_delay=0.0)
    )

    result = await meter.call(
        provider_call, round_id=None, provider="deepseek", model=MODEL, estimated_tokens=10
    )

    assert result == "ok"
    assert len(calls) == 2
    assert len(repository.records) == 1, "the successful attempt is metered once"


# --------------------------------------------------------------------------- #
# 2. Timeouts are configured, never a literal
# --------------------------------------------------------------------------- #


def test_the_defender_gets_the_most_generous_read_timeout(settings: Settings) -> None:
    """Its proposal prompt is the longest one the loop sends."""
    _, defender = settings.timeout_for(LLMProvider.DEEPSEEK, "defender")
    _, target = settings.timeout_for(LLMProvider.DEEPSEEK, "target")

    assert defender >= 180.0
    assert defender > target


def test_the_connect_timeout_stays_short(settings: Settings) -> None:
    """A host that has not shaken hands in 10s is unreachable, not slow."""
    for role in ("target", "attacker", "defender", "classifier"):
        connect, _ = settings.timeout_for(LLMProvider.GROQ, role)
        assert connect == DEFAULT_CONNECT_TIMEOUT_SECONDS <= 10.0


def test_a_read_timeout_is_overridable_per_provider_and_role(env: dict[str, str]) -> None:
    del env  # the fixture's effect is on the environment
    settings = load_settings(
        _env_file=None, PROVIDER_READ_TIMEOUTS="deepseek:defender=300,classifier=45"
    )

    assert settings.read_timeout_for(LLMProvider.DEEPSEEK, "defender") == 300.0
    assert settings.read_timeout_for(LLMProvider.GROQ, "classifier") == 45.0
    assert (
        settings.read_timeout_for(LLMProvider.GROQ, "defender")
        == (ROLE_READ_TIMEOUT_SECONDS["defender"])
    ), "a provider-specific override must not leak to the other provider"


def test_a_malformed_timeout_override_is_refused(env: dict[str, str]) -> None:
    del env
    with pytest.raises(ConfigurationError, match="PROVIDER_READ_TIMEOUTS"):
        load_settings(_env_file=None, PROVIDER_READ_TIMEOUTS="deepseek:defender=soon")


def test_the_built_client_carries_its_roles_timeout(settings: Settings) -> None:
    """Wired through, not merely configured."""
    import asyncio
    from contextlib import AsyncExitStack

    from crucible.cli.providers import Provider, build_factories

    async def _timeouts() -> dict[str, object]:
        async with AsyncExitStack() as stack:
            factories = build_factories(settings, Provider.GROQ, stack)
            return {
                "defender": factories.defender_llm()._inner._timeout,
                "target": factories.target_llm()._timeout,
            }
        raise AssertionError("unreachable")

    timeouts = asyncio.run(_timeouts())
    defender = timeouts["defender"]
    target = timeouts["target"]
    assert isinstance(defender, httpx.Timeout) and isinstance(target, httpx.Timeout)
    assert defender.read == ROLE_READ_TIMEOUT_SECONDS["defender"]
    assert target.read == ROLE_READ_TIMEOUT_SECONDS["target"]
    assert defender.connect == DEFAULT_CONNECT_TIMEOUT_SECONDS


# --------------------------------------------------------------------------- #
# 3. Pacing sized per provider, from settings
# --------------------------------------------------------------------------- #


def test_the_two_providers_have_different_default_limits(settings: Settings) -> None:
    """Groq free tier, DeepSeek paid: one number for both fits neither."""
    groq = settings.rate_limits_for(LLMProvider.GROQ)
    deepseek = settings.rate_limits_for(LLMProvider.DEEPSEEK)

    assert deepseek[0] > groq[0] * 10, "a free-tier window on a paid key wastes the run"
    assert groq == (6500, 25), "the Groq key is still free tier"


def test_limits_are_overridable_from_the_environment(env: dict[str, str]) -> None:
    """Tuning a tier must never need a code change."""
    del env
    settings = load_settings(_env_file=None, PROVIDER_RATE_LIMITS="deepseek=250000/600")

    assert settings.rate_limits_for(LLMProvider.DEEPSEEK) == (250_000, 600)


def test_concurrency_is_per_provider_and_overridable(env: dict[str, str]) -> None:
    del env
    settings = load_settings(_env_file=None, PROVIDER_CONCURRENCY="deepseek=8,groq=1")

    assert settings.concurrency_for(LLMProvider.DEEPSEEK) == 8
    assert settings.concurrency_for(LLMProvider.GROQ) == 1
    assert settings.provider_concurrency == {"groq": 1, "deepseek": 8}


def test_a_paid_provider_may_exceed_the_global_concurrency(settings: Settings) -> None:
    """PROVIDER_MAX_CONCURRENCY is a fallback, not a ceiling on every provider."""
    assert settings.PROVIDER_MAX_CONCURRENCY == 1
    assert settings.concurrency_for(LLMProvider.DEEPSEEK) > 1


def test_the_pacer_gives_each_provider_its_own_concurrency() -> None:
    pacer = ProviderPacer(max_concurrency=1, concurrency={"deepseek": 4})

    assert pacer.concurrency_for("deepseek:deepseek-v4-flash") == 4
    assert pacer.concurrency_for("groq:openai-model") == 1


async def test_two_providers_with_different_limits_do_not_share_a_window() -> None:
    """The paid provider must not be throttled by the free one's window."""
    pacer = ProviderPacer(
        tokens_per_minute=6500,
        requests_per_minute=25,
        limits={"groq": (1000, 10), "deepseek": (200_000, 600)},
    )

    async with pacer.slot(model="groq:m", estimated_tokens=900):
        pass
    async with pacer.slot(model="deepseek:m", estimated_tokens=900):
        pass

    assert pacer.window_for("groq:m").tokens_per_minute == 1000
    assert pacer.window_for("deepseek:m").tokens_per_minute == 200_000
    assert pacer.tokens_used("groq:m") == pacer.tokens_used("deepseek:m") == 900


async def test_a_slow_provider_does_not_space_a_fast_ones_calls() -> None:
    """The minimum interval is per provider, like the window."""
    slept: list[float] = []
    now = [0.0]

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    pacer = ProviderPacer(
        min_interval_seconds=2.0, max_concurrency=1, sleep=sleep, clock=lambda: now[0]
    )

    async with pacer.slot(model="groq:m"):
        pass
    async with pacer.slot(model="deepseek:m"):
        pass

    assert slept == [], "two providers keep two schedules"


async def test_parallel_calls_cannot_all_pass_the_check_before_one_commits() -> None:
    """The reservation is taken under a lock, or the window means nothing."""
    pacer = ProviderPacer(tokens_per_minute=1000, requests_per_minute=100)

    async def one() -> None:
        async with pacer.slot(model="deepseek:m", estimated_tokens=400):
            await asyncio.sleep(0)

    await asyncio.gather(*(one() for _ in range(2)))

    assert pacer.tokens_used("deepseek:m") == 800, "both reservations were booked"


async def test_a_retry_books_its_own_reservation() -> None:
    """A retried call spends tokens again; riding the first booking hides them."""
    pacer = ProviderPacer(tokens_per_minute=100_000, requests_per_minute=1000)
    calls: list[int] = []

    async def provider_call() -> MeteredResult[str]:
        calls.append(1)
        if len(calls) == 1:
            raise ProviderTimeout("deepseek", MODEL, 180.0)
        return MeteredResult(value="ok", usage=TokenUsage(prompt_tokens=50, completion_tokens=10))

    meter = CostMeter(
        FakeSpendRepository(),
        Decimal("5.00"),
        retry_policy=RetryPolicy(attempts=3, base_delay=0.0),
        pacer=pacer,
    )

    await meter.call(
        provider_call, round_id=None, provider="deepseek", model=MODEL, estimated_tokens=500
    )

    assert pacer.window_for("deepseek:test-model").requests_used == 2, (
        "the retry reserved its own slot rather than riding the first attempt's"
    )


def test_the_named_status_for_a_round_with_no_candidates_exists() -> None:
    """A round where every branch failed is not a round where none improved."""
    assert STATUS_NO_CANDIDATES == "no_candidates"
