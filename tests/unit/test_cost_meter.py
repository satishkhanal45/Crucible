"""Verification tests 3 and 4: the cost meter accumulates, caps, and warns."""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal

import pytest

from crucible.schemas.spend import TokenUsage
from crucible.services.cost_meter import (
    BudgetExceeded,
    CostMeter,
    MeteredResult,
    ModelPrice,
    price_key,
)
from crucible.services.retry import RateLimited, RetryPolicy
from tests.fixtures.fake_spend import FakeSpendRepository

PROVIDER = "groq"
MODEL = "test-model"
# One cent per call: 1000 prompt tokens at $10 per million.
PRICING = {price_key(PROVIDER, MODEL): ModelPrice(Decimal("10"), Decimal("20"))}
CALL_USAGE = TokenUsage(prompt_tokens=1000, completion_tokens=0)
CALL_COST = Decimal("0.01")


@pytest.fixture
def repository() -> FakeSpendRepository:
    return FakeSpendRepository()


@pytest.fixture
def meter(repository: FakeSpendRepository, budget: Decimal) -> CostMeter:
    return CostMeter(repository, budget, pricing=PRICING)


@pytest.fixture
def round_id() -> uuid.UUID:
    return uuid.uuid4()


async def test_estimate_uses_prompt_and_completion_prices(meter: CostMeter) -> None:
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
    assert meter.estimate(PROVIDER, MODEL, usage) == Decimal("20.00000000")


async def test_spend_accumulates_across_calls(
    meter: CostMeter, repository: FakeSpendRepository, round_id: uuid.UUID
) -> None:
    for _ in range(10):
        await meter.record(round_id=round_id, provider=PROVIDER, model=MODEL, usage=CALL_USAGE)

    assert len(repository.records) == 10
    assert await meter.spent(round_id) == CALL_COST * 10


async def test_spend_is_tracked_per_round(meter: CostMeter) -> None:
    first, second = uuid.uuid4(), uuid.uuid4()
    await meter.record(round_id=first, provider=PROVIDER, model=MODEL, usage=CALL_USAGE)
    await meter.record(round_id=second, provider=PROVIDER, model=MODEL, usage=CALL_USAGE)

    assert await meter.spent(first) == CALL_COST
    assert await meter.spent(second) == CALL_COST


async def test_budget_trips_at_the_cap_and_not_before(
    meter: CostMeter, repository: FakeSpendRepository, budget: Decimal, round_id: uuid.UUID
) -> None:
    """A simulated 1000-call sequence stops at the cap, and only at the cap."""
    calls_within_budget = int(budget / CALL_COST)  # 100 at $1.00 and $0.01 per call
    tripped_on: int | None = None

    for call_number in range(1, 1001):
        try:
            await meter.record(round_id=round_id, provider=PROVIDER, model=MODEL, usage=CALL_USAGE)
        except BudgetExceeded as error:
            tripped_on = call_number
            assert str(round_id) in str(error)
            assert "1.010000" in str(error)
            assert f"{budget:.6f}" in str(error)
            assert error.round_id == round_id
            assert error.spent == budget + CALL_COST
            assert error.budget == budget
            break

    assert tripped_on == calls_within_budget + 1
    # The spend that broke the budget is still on the ledger.
    assert len(repository.records) == calls_within_budget + 1
    assert await meter.spent(round_id) == budget + CALL_COST


async def test_spend_exactly_at_the_cap_does_not_raise(
    meter: CostMeter, budget: Decimal, round_id: uuid.UUID
) -> None:
    for _ in range(int(budget / CALL_COST)):
        await meter.record(round_id=round_id, provider=PROVIDER, model=MODEL, usage=CALL_USAGE)
    assert await meter.spent(round_id) == budget


async def test_unknown_model_records_null_cost_and_warns(
    meter: CostMeter,
    repository: FakeSpendRepository,
    round_id: uuid.UUID,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="crucible.services.cost_meter"):
        record = await meter.record(
            round_id=round_id, provider=PROVIDER, model="not-in-the-price-table", usage=CALL_USAGE
        )

    assert record.estimated_cost_usd is None
    assert repository.records[0].estimated_cost_usd is None
    assert await meter.spent(round_id) == Decimal(0)

    warnings = [r for r in caplog.records if r.message == "cost_meter.unknown_model"]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert warnings[0].model == "not-in-the-price-table"


async def test_unknown_model_never_raises_even_over_budget(
    meter: CostMeter, round_id: uuid.UUID
) -> None:
    for _ in range(200):
        await meter.record(round_id=round_id, provider=PROVIDER, model="unpriced", usage=CALL_USAGE)


async def test_call_meters_a_stubbed_provider_call(
    meter: CostMeter, repository: FakeSpendRepository, round_id: uuid.UUID
) -> None:
    async def stubbed() -> MeteredResult[str]:
        return MeteredResult(value="scripted response", usage=CALL_USAGE)

    value = await meter.call(stubbed, round_id=round_id, provider=PROVIDER, model=MODEL)

    assert value == "scripted response"
    assert len(repository.records) == 1
    assert repository.records[0].estimated_cost_usd == CALL_COST


async def test_call_retries_a_rate_limit_then_meters_once(
    meter: CostMeter, repository: FakeSpendRepository, round_id: uuid.UUID
) -> None:
    attempts = 0

    async def flaky() -> MeteredResult[str]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            # retry_after=0 keeps the test instant without patching the clock.
            raise RateLimited("429 from the provider", retry_after=0.0)
        return MeteredResult(value="ok", usage=CALL_USAGE)

    value = await meter.call(
        flaky,
        round_id=round_id,
        provider=PROVIDER,
        model=MODEL,
        retry_policy=RetryPolicy(attempts=5, base_delay=0.01),
    )

    assert value == "ok"
    assert attempts == 3
    # Retries are not billed twice: only the successful call is metered.
    assert len(repository.records) == 1


async def test_call_refuses_to_start_when_the_round_is_already_over_budget(
    meter: CostMeter, round_id: uuid.UUID
) -> None:
    for _ in range(101):
        try:
            await meter.record(round_id=round_id, provider=PROVIDER, model=MODEL, usage=CALL_USAGE)
        except BudgetExceeded:
            break

    async def never_called() -> MeteredResult[str]:
        raise AssertionError("the provider must not be called once the budget is blown")

    with pytest.raises(BudgetExceeded):
        await meter.call(never_called, round_id=round_id, provider=PROVIDER, model=MODEL)


def test_meter_rejects_a_non_positive_budget(repository: FakeSpendRepository) -> None:
    with pytest.raises(ValueError, match="budget_usd"):
        CostMeter(repository, Decimal(0))
