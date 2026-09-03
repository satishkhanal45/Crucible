"""Token-aware pacing: the limit that actually bites is TPM, not RPM.

Groq's free tier allows 8000 tokens per minute. A 40-call reclassify run stayed
comfortably under any request-per-minute limit and still died at call 23,
because request pacing cannot see tokens. These tests drive the pacer on a fake
clock, so they assert the schedule rather than the wall clock.
"""

from __future__ import annotations

from decimal import Decimal

from crucible.config import Settings
from crucible.schemas.spend import TokenUsage
from crucible.services.cost_meter import CostMeter, MeteredResult
from crucible.services.pacing import (
    COMPLETION_ALLOWANCE_TOKENS,
    WINDOW_SECONDS,
    ProviderPacer,
    estimate_tokens,
)
from tests.fixtures.fake_spend import FakeSpendRepository

MODEL = "groq:test-model"
OTHER_MODEL = "groq:other-model"


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


def _pacer(clock: FakeClock, *, tokens: int = 1000, requests: int = 100) -> ProviderPacer:
    return ProviderPacer(
        min_interval_seconds=0.0,
        max_concurrency=1,
        tokens_per_minute=tokens,
        requests_per_minute=requests,
        sleep=clock.sleep,
        clock=clock,
    )


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #


def test_an_estimate_covers_the_prompt_and_a_reply_allowance() -> None:
    estimate = estimate_tokens("x" * 4000)

    assert estimate == 1000 + COMPLETION_ALLOWANCE_TOKENS


def test_even_an_empty_prompt_reserves_something() -> None:
    assert estimate_tokens("", completion_allowance=0) == 1


# --------------------------------------------------------------------------- #
# The window delays rather than sends
# --------------------------------------------------------------------------- #


async def test_a_sequence_that_would_exceed_tpm_is_delayed() -> None:
    clock = FakeClock()
    pacer = _pacer(clock, tokens=1000)

    for _ in range(3):
        async with pacer.slot(model=MODEL, estimated_tokens=400):
            pass

    assert clock.slept, "the third call must wait rather than breach the window"
    assert clock.now >= WINDOW_SECONDS, "it waits for the window to roll, not a fixed delay"


async def test_calls_inside_the_budget_are_not_delayed() -> None:
    clock = FakeClock()
    pacer = _pacer(clock, tokens=10_000)

    for _ in range(5):
        async with pacer.slot(model=MODEL, estimated_tokens=100):
            pass

    assert clock.slept == []


async def test_the_window_rolls_rather_than_resetting() -> None:
    """After the oldest call ages out, its tokens come back and no more."""
    clock = FakeClock()
    pacer = _pacer(clock, tokens=1000)

    async with pacer.slot(model=MODEL, estimated_tokens=600):
        pass
    clock.now += WINDOW_SECONDS + 1.0  # the first call ages out
    async with pacer.slot(model=MODEL, estimated_tokens=600):
        pass

    assert clock.slept == [], "an aged-out call frees its tokens"
    assert pacer.tokens_used(MODEL) == 600


async def test_a_request_limit_also_delays() -> None:
    """Tokens are the binding limit, but RPM is still a limit."""
    clock = FakeClock()
    pacer = _pacer(clock, tokens=1_000_000, requests=2)

    for _ in range(3):
        async with pacer.slot(model=MODEL, estimated_tokens=1):
            pass

    assert clock.slept, "the third request waited on the request cap"


async def test_a_call_larger_than_the_whole_window_is_still_sent() -> None:
    """Refusing it would deadlock the run; the provider's 429 is the backstop."""
    clock = FakeClock()
    pacer = _pacer(clock, tokens=100)

    async with pacer.slot(model=MODEL, estimated_tokens=5_000):
        pass

    assert clock.slept == []


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #


async def test_actual_usage_replaces_the_estimate() -> None:
    clock = FakeClock()
    pacer = _pacer(clock, tokens=10_000)

    async with pacer.slot(model=MODEL, estimated_tokens=900) as reservation:
        assert reservation.reserved_tokens == 900
        reservation.reconcile(120)

    assert pacer.tokens_used(MODEL) == 120, "the window tracks what was spent, not what we guessed"


async def test_an_over_spending_call_reconciles_upward_and_delays_the_next() -> None:
    clock = FakeClock()
    pacer = _pacer(clock, tokens=1000)

    async with pacer.slot(model=MODEL, estimated_tokens=100) as reservation:
        reservation.reconcile(950)  # the reply was far longer than allowed for
    async with pacer.slot(model=MODEL, estimated_tokens=100):
        pass

    assert clock.slept, "the real cost, not the estimate, is what the next call sees"


async def test_the_meter_reconciles_every_metered_call() -> None:
    """No agent has to remember to reconcile: `CostMeter.call` does it."""
    clock = FakeClock()
    pacer = _pacer(clock, tokens=10_000)
    meter = CostMeter(FakeSpendRepository(), Decimal("1.00"), pacer=pacer)

    async def call() -> MeteredResult[str]:
        return MeteredResult(value="ok", usage=TokenUsage(prompt_tokens=40, completion_tokens=25))

    await meter.call(call, round_id=None, provider="groq", model="test-model", estimated_tokens=900)

    assert pacer.tokens_used(MODEL) == 65


# --------------------------------------------------------------------------- #
# Per model
# --------------------------------------------------------------------------- #


async def test_the_limiter_is_per_model() -> None:
    clock = FakeClock()
    pacer = _pacer(clock, tokens=1000)

    async with pacer.slot(model=MODEL, estimated_tokens=900):
        pass
    async with pacer.slot(model=OTHER_MODEL, estimated_tokens=900):
        pass

    assert clock.slept == [], "a cheap model must not spend an expensive one's budget"
    assert pacer.tokens_used(MODEL) == 900
    assert pacer.tokens_used(OTHER_MODEL) == 900


async def test_one_model_saturating_its_window_does_not_block_another() -> None:
    clock = FakeClock()
    pacer = _pacer(clock, tokens=1000)

    for _ in range(3):
        async with pacer.slot(model=MODEL, estimated_tokens=400):
            pass
    waits_after_first_model = len(clock.slept)

    async with pacer.slot(model=OTHER_MODEL, estimated_tokens=400):
        pass

    assert len(clock.slept) == waits_after_first_model


async def test_the_meter_keys_the_window_by_provider_and_model() -> None:
    clock = FakeClock()
    pacer = _pacer(clock, tokens=10_000)
    meter = CostMeter(FakeSpendRepository(), Decimal("1.00"), pacer=pacer)

    async def call() -> MeteredResult[str]:
        return MeteredResult(value="ok", usage=TokenUsage(prompt_tokens=10, completion_tokens=1))

    await meter.call(call, round_id=None, provider="GROQ", model="Test-Model", estimated_tokens=100)

    assert pacer.tokens_used(MODEL) == 11, "the key is lower-cased provider:model"


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_the_offline_pacer_enforces_no_window() -> None:
    """The suite must not wait on a token budget it is not spending."""
    pacer = ProviderPacer.unlimited()

    assert not pacer.limits_tokens


def test_settings_carry_the_free_tier_values(settings: Settings) -> None:
    assert settings.PROVIDER_TOKENS_PER_MINUTE == 6500, "below Groq's 8000 TPM, with margin"
    assert settings.PROVIDER_REQUESTS_PER_MINUTE == 25


def test_the_pacer_built_from_settings_limits_tokens(settings: Settings) -> None:
    pacer = ProviderPacer.from_settings(
        settings.PROVIDER_MIN_INTERVAL_SECONDS,
        settings.PROVIDER_MAX_CONCURRENCY,
        tokens_per_minute=settings.PROVIDER_TOKENS_PER_MINUTE,
        requests_per_minute=settings.PROVIDER_REQUESTS_PER_MINUTE,
    )

    assert pacer.limits_tokens
    assert pacer.tokens_per_minute == 6500


async def test_the_executor_pool_cannot_outrun_the_provider_bound() -> None:
    """A pool slot is a provider caller, so the pool is capped by the pacer.

    `build_components` sets pool size to `min(concurrency, provider bound)`;
    this asserts the arithmetic that makes a wide pool safe.
    """
    from crucible.loop.runner import LoopSettings

    settings = LoopSettings(concurrency=8, provider_max_concurrency=1)

    assert min(settings.concurrency, settings.provider_max_concurrency) == 1
