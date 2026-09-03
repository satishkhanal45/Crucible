"""Hotfix verification: the budget is enforceable, and calls are paced.

BUG 2 was that no configured model had a price, so every call recorded a NULL
cost, `ROUND_BUDGET_USD` was inert, and the `budget_exceeded` halt reason was
unreachable. BUG's companion was the absence of any pacing: a live run met its
rate limit 23 calls into a 40-call loop.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal

import pytest

from crucible.config import (
    DEFAULT_MODEL_PRICING,
    Settings,
    load_settings,
    price_key,
    validate_model_pricing,
)
from crucible.schemas.spend import TokenUsage
from crucible.services.cost_meter import BudgetExceeded, CostMeter, MeteredResult, pricing_from
from crucible.services.pacing import ProviderPacer
from tests.fixtures.fake_spend import FakeSpendRepository

# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #


def test_every_configured_model_has_a_price(settings: Settings) -> None:
    """The condition BUG 2 violated, asserted against the shipped defaults."""
    assert settings.unpriced_models() == ()


def test_the_models_the_live_run_used_are_priced() -> None:
    """`openai/gpt-oss-120b` is the id that worked after the decommissioning."""
    assert price_key("groq", "openai/gpt-oss-120b") in DEFAULT_MODEL_PRICING
    assert price_key("groq", "llama-3.1-8b-instant") in DEFAULT_MODEL_PRICING


def test_a_priced_model_produces_a_non_null_cost(settings: Settings) -> None:
    meter = CostMeter(
        FakeSpendRepository(), Decimal("1.00"), pricing=pricing_from(settings.model_pricing)
    )
    cost = meter.estimate(
        "groq", "openai/gpt-oss-120b", TokenUsage(prompt_tokens=1_000_000, completion_tokens=0)
    )

    assert cost == Decimal("0.15000000")


async def test_a_priced_model_can_actually_exhaust_the_budget(settings: Settings) -> None:
    """Without a price this is unreachable, which is what made the budget inert."""
    repository = FakeSpendRepository()
    meter = CostMeter(repository, Decimal("0.10"), pricing=pricing_from(settings.model_pricing))

    async def call() -> MeteredResult[str]:
        return MeteredResult(
            value="ok", usage=TokenUsage(prompt_tokens=1_000_000, completion_tokens=0)
        )

    with pytest.raises(BudgetExceeded):
        await meter.call(call, round_id=None, provider="groq", model="openai/gpt-oss-120b")

    assert repository.records[0].estimated_cost_usd == Decimal("0.15000000")


def test_pricing_can_be_overridden_from_the_environment(env: dict[str, str]) -> None:
    del env
    settings = load_settings(
        _env_file=None, MODEL_PRICING="groq:openai/gpt-oss-120b=1.25/2.50,groq:brand-new=0.01/0.02"
    )

    assert settings.model_pricing["groq:openai/gpt-oss-120b"] == (Decimal("1.25"), Decimal("2.50"))
    assert settings.model_pricing["groq:brand-new"] == (Decimal("0.01"), Decimal("0.02"))
    # Untouched entries survive the merge.
    assert "groq:llama-3.1-8b-instant" in settings.model_pricing


@pytest.mark.parametrize("value", ["nonsense", "groq:model=1.0", "groq:model=a/b"])
def test_a_malformed_pricing_override_is_rejected(env: dict[str, str], value: str) -> None:
    del env
    from crucible.config import ConfigurationError

    with pytest.raises(ConfigurationError):
        load_settings(_env_file=None, MODEL_PRICING=value)


# --------------------------------------------------------------------------- #
# The startup warning
# --------------------------------------------------------------------------- #


def test_an_unpriced_model_warns_once_and_still_runs(
    env: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """An unpriced model must not fail startup: it just cannot be budgeted."""
    del env
    settings = load_settings(_env_file=None, CLASSIFIER_MODEL="groq-model-with-no-price")

    with caplog.at_level(logging.WARNING, logger="crucible.config"):
        unpriced = validate_model_pricing(settings)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, "exactly one warning per unpriced model"
    assert warnings[0].message == "config.model_not_priced"
    assert getattr(warnings[0], "model", "") == "groq:groq-model-with-no-price"
    assert "ROUND_BUDGET_USD" in getattr(warnings[0], "detail", "")
    assert unpriced == ("groq:groq-model-with-no-price",)


async def test_a_call_to_an_unpriced_model_still_succeeds(env: dict[str, str]) -> None:
    del env
    repository = FakeSpendRepository()
    meter = CostMeter(repository, Decimal("1.00"))

    async def call() -> MeteredResult[str]:
        return MeteredResult(value="ok", usage=TokenUsage(prompt_tokens=10, completion_tokens=5))

    value = await meter.call(call, round_id=None, provider="groq", model="no-such-price")

    assert value == "ok"
    assert repository.records[0].estimated_cost_usd is None


def test_no_warning_when_every_model_is_priced(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="crucible.config"):
        assert validate_model_pricing(settings) == ()

    assert [record for record in caplog.records if record.levelno == logging.WARNING] == []


# --------------------------------------------------------------------------- #
# Pacing
# --------------------------------------------------------------------------- #


class FakeClock:
    """A monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


async def test_calls_are_spaced_by_the_minimum_interval() -> None:
    clock = FakeClock()
    pacer = ProviderPacer(
        min_interval_seconds=2.0, max_concurrency=1, sleep=clock.sleep, clock=clock
    )

    for _ in range(3):
        async with pacer.slot():
            pass

    assert clock.slept == [2.0, 2.0], "the first call goes straight through, then spacing"


async def test_a_slow_call_does_not_add_its_own_duration_to_the_next_wait() -> None:
    clock = FakeClock()
    pacer = ProviderPacer(
        min_interval_seconds=2.0, max_concurrency=1, sleep=clock.sleep, clock=clock
    )

    async with pacer.slot():
        clock.now += 5.0  # a call slower than the interval
    async with pacer.slot():
        pass

    assert clock.slept == [], "the interval had already elapsed while the call ran"


async def test_concurrency_one_serialises_provider_calls() -> None:
    pacer = ProviderPacer(min_interval_seconds=0.0, max_concurrency=1)
    active = 0
    peak = 0

    async def call() -> None:
        nonlocal active, peak
        async with pacer.slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

    await asyncio.gather(*(call() for _ in range(5)))

    assert peak == 1


async def test_concurrency_is_configurable_above_one() -> None:
    pacer = ProviderPacer(min_interval_seconds=0.0, max_concurrency=3)
    active = 0
    peak = 0
    release = asyncio.Event()

    async def call() -> None:
        nonlocal active, peak
        async with pacer.slot():
            active += 1
            peak = max(peak, active)
            await release.wait()
            active -= 1

    tasks = [asyncio.create_task(call()) for _ in range(6)]
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*tasks)

    assert peak == 3


async def test_the_meter_paces_every_provider_call() -> None:
    """Pacing lives behind `CostMeter.call`, which every LLM call routes through."""
    clock = FakeClock()
    meter = CostMeter(
        FakeSpendRepository(),
        Decimal("1.00"),
        pacer=ProviderPacer(
            min_interval_seconds=1.5, max_concurrency=1, sleep=clock.sleep, clock=clock
        ),
    )

    async def call() -> MeteredResult[str]:
        return MeteredResult(value="ok", usage=TokenUsage(prompt_tokens=1, completion_tokens=1))

    for _ in range(3):
        await meter.call(call, round_id=None, provider="groq", model="openai/gpt-oss-120b")

    assert clock.slept == [1.5, 1.5]


async def test_pacing_applies_to_each_retry_attempt() -> None:
    """A retry is another request, so it waits its turn like any other."""
    from crucible.services.retry import RateLimited, RetryPolicy

    clock = FakeClock()
    meter = CostMeter(
        FakeSpendRepository(),
        Decimal("1.00"),
        retry_policy=RetryPolicy(attempts=3, base_delay=0.0, max_delay=0.0, jitter=0.0),
        pacer=ProviderPacer(
            min_interval_seconds=1.0, max_concurrency=1, sleep=clock.sleep, clock=clock
        ),
    )
    attempts = 0

    async def call() -> MeteredResult[str]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RateLimited("429", retry_after=0.0)
        return MeteredResult(value="ok", usage=TokenUsage(prompt_tokens=1, completion_tokens=1))

    await meter.call(call, round_id=None, provider="groq", model="openai/gpt-oss-120b")

    assert attempts == 3
    assert clock.slept.count(1.0) == 2, "the two retries each waited their pacing turn"


def test_the_unlimited_pacer_is_the_offline_default() -> None:
    """Stubbed runs must not sleep: the whole suite would pay for it."""
    pacer = ProviderPacer.unlimited()

    assert pacer.min_interval_seconds == 0.0
    assert pacer.max_concurrency > 1


@pytest.mark.parametrize(("interval", "concurrency"), [(-1.0, 1), (0.0, 0), (0.0, -3)])
def test_a_nonsense_pacer_is_rejected(interval: float, concurrency: int) -> None:
    with pytest.raises(ValueError):
        ProviderPacer(min_interval_seconds=interval, max_concurrency=concurrency)
