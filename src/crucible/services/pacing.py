"""Pacing for provider calls.

Free-tier rate limits are the binding constraint on this project, and the limit
that actually bites is **tokens per minute, not requests per minute**: Groq's
free tier allows 8000 TPM, which one long attacker prompt plus its reply can
consume several times over inside a minute of otherwise modest traffic. Pacing
by request count alone therefore stays under the request limit while sailing
straight through the token limit, which is what ended a live 40-call reclassify
run 23 calls in.

So the pacer keeps a rolling one-minute window per **provider and model**, of
both requests and tokens. A call reserves an estimate before it is sent and
reconciles the reservation against the provider's reported usage after, so the
window tracks what was really spent rather than what we guessed. Backoff in
`crucible.services.retry` still handles the 429 that slips through; this exists
to make that the exception rather than the plan.

The key is `provider:model`, and the limits are looked up by the provider half
of it. Two providers are two separate pools — that is the reason for running on
two — so `groq:X` and `deepseek:X` must never share a window, and a window sized
for one provider's published limit must not be applied to the other's.

The pacer is owned by `CostMeter`, because "every LLM call routes through
`CostMeter`" is already a project invariant and that makes this one too.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from crucible.logging import get_logger

logger = get_logger(__name__)

SleepFn = Callable[[float], Awaitable[None]]
ClockFn = Callable[[], float]

#: The provider's window. Not configurable: it is the provider's, not ours.
WINDOW_SECONDS = 60.0

#: Rough English-text ratio, used only to reserve before a call. The reservation
#: is replaced by the provider's own count the moment the reply lands, so an
#: inaccurate estimate costs at most one call's worth of over- or under-booking.
CHARS_PER_TOKEN = 4

#: Reserved for a reply whose length is unknown when the request goes out.
#: Reconciled away as soon as the call returns.
COMPLETION_ALLOWANCE_TOKENS = 512

#: Effectively no bound, while keeping the semaphore's type honest.
_UNBOUNDED = 1024


def estimate_tokens(text: str, *, completion_allowance: int = COMPLETION_ALLOWANCE_TOKENS) -> int:
    """Tokens one call is likely to spend, prompt plus a reply allowance."""
    return max(1, len(text) // CHARS_PER_TOKEN) + completion_allowance


@dataclass
class _Entry:
    """One call's footprint in the rolling window."""

    at: float
    tokens: int


class Reservation:
    """A claim on the window, to be reconciled with what the call really cost."""

    def __init__(self, entry: _Entry | None = None) -> None:
        self._entry = entry
        self.reconciled = False

    @property
    def reserved_tokens(self) -> int:
        return self._entry.tokens if self._entry is not None else 0

    def reconcile(self, actual_tokens: int) -> None:
        """Replace the estimate with the provider's own count.

        Called after every metered call, so a window built from estimates
        converges on the truth within one call rather than drifting all run.
        """
        self.reconciled = True
        if self._entry is None:
            return
        self._entry.tokens = max(0, actual_tokens)


@dataclass
class ModelWindow:
    """A rolling one-minute window of requests and tokens for one model."""

    tokens_per_minute: int
    requests_per_minute: int
    entries: list[_Entry] = field(default_factory=list)

    def prune(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        self.entries = [entry for entry in self.entries if entry.at > cutoff]

    @property
    def tokens_used(self) -> int:
        return sum(entry.tokens for entry in self.entries)

    @property
    def requests_used(self) -> int:
        return len(self.entries)

    def wait_for(self, now: float, tokens: int) -> float:
        """Seconds until this call fits, or 0.0 when it fits already.

        A call larger than the whole window can never fit and is let through
        once the window is empty: refusing it would deadlock the run, and the
        provider's own 429 handling is the backstop.
        """
        self.prune(now)
        fits_tokens = self.tokens_used + tokens <= self.tokens_per_minute
        fits_requests = self.requests_used < self.requests_per_minute
        if fits_tokens and fits_requests:
            return 0.0
        if not self.entries:
            # Nothing to wait for: the call is bigger than the window itself.
            logger.warning(
                "provider.call_exceeds_window",
                extra={"tokens": tokens, "tokens_per_minute": self.tokens_per_minute},
            )
            return 0.0
        oldest = min(entry.at for entry in self.entries)
        return max(0.0, oldest + WINDOW_SECONDS - now)

    def record(self, now: float, tokens: int) -> _Entry:
        entry = _Entry(at=now, tokens=tokens)
        self.entries.append(entry)
        return entry


class ProviderPacer:
    """Holds provider calls to a request rate and a per-model token rate.

    The interval is measured between call *starts*, so a slow call does not add
    its own duration to the next one's wait.
    """

    def __init__(
        self,
        *,
        min_interval_seconds: float = 0.0,
        max_concurrency: int = 1,
        tokens_per_minute: int = 0,
        requests_per_minute: int = 0,
        limits: Mapping[str, tuple[int, int]] | None = None,
        sleep: SleepFn = asyncio.sleep,
        clock: ClockFn = time.monotonic,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds cannot be negative")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        if tokens_per_minute < 0 or requests_per_minute < 0:
            raise ValueError("rate limits cannot be negative")
        self._interval = min_interval_seconds
        self._max_concurrency = max_concurrency
        self._tokens_per_minute = tokens_per_minute
        self._requests_per_minute = requests_per_minute
        #: Per-provider overrides of the pair above, keyed by provider name.
        #: Two providers publish different limits and never share a pool.
        self._limits = dict(limits or {})
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._gate = asyncio.Lock()
        self._sleep = sleep
        self._clock = clock
        self._next_allowed: float | None = None
        self._windows: dict[str, ModelWindow] = {}
        #: Total seconds spent waiting, for tests and for the round log.
        self.waited_seconds = 0.0

    @classmethod
    def unlimited(cls) -> ProviderPacer:
        """No spacing, no bound, no window: the default for offline runs."""
        return cls(min_interval_seconds=0.0, max_concurrency=_UNBOUNDED)

    @classmethod
    def from_settings(
        cls,
        min_interval_seconds: float,
        max_concurrency: int,
        *,
        tokens_per_minute: int = 0,
        requests_per_minute: int = 0,
        limits: Mapping[str, tuple[int, int]] | None = None,
    ) -> ProviderPacer:
        return cls(
            min_interval_seconds=min_interval_seconds,
            max_concurrency=max_concurrency,
            tokens_per_minute=tokens_per_minute,
            requests_per_minute=requests_per_minute,
            limits=limits,
        )

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def min_interval_seconds(self) -> float:
        return self._interval

    @property
    def tokens_per_minute(self) -> int:
        return self._tokens_per_minute

    @property
    def requests_per_minute(self) -> int:
        return self._requests_per_minute

    @property
    def limits_tokens(self) -> bool:
        """True when any configured provider has a window at all."""
        return any(
            tokens > 0 and requests > 0
            for tokens, requests in (
                (self._tokens_per_minute, self._requests_per_minute),
                *self._limits.values(),
            )
        )

    def limits_key(self, key: str) -> bool:
        """True when this `provider:model` key is windowed.

        Decided per key, not globally: one provider having a limit configured
        must not impose a zero-sized window on another that has none.
        """
        tokens, requests = self.limits_for(key)
        return tokens > 0 and requests > 0

    def limits_for(self, key: str) -> tuple[int, int]:
        """`(tokens_per_minute, requests_per_minute)` for one `provider:model`.

        The provider half of the key selects the limits; a provider with no
        entry falls back to the configured pair.
        """
        provider, separator, _ = key.partition(":")
        if separator:
            override = self._limits.get(provider.strip().lower())
            if override is not None:
                return override
        return (self._tokens_per_minute, self._requests_per_minute)

    def window_for(self, model: str) -> ModelWindow:
        """The rolling window for one `provider:model` key, made on first use.

        Windows are per provider AND per model: a cheap classifier model and an
        expensive attacker model do not consume one another's budget, and two
        providers do not consume one another's at all, since the whole reason
        for a second provider is that it is a second pool.
        """
        window = self._windows.get(model)
        if window is None:
            tokens_per_minute, requests_per_minute = self.limits_for(model)
            window = ModelWindow(
                tokens_per_minute=tokens_per_minute,
                requests_per_minute=requests_per_minute,
            )
            self._windows[model] = window
        return window

    def tokens_used(self, model: str) -> int:
        """Tokens currently inside this model's window, after pruning."""
        window = self.window_for(model)
        window.prune(self._clock())
        return window.tokens_used

    async def _wait_turn(self) -> None:
        """Claim the next slot in the schedule, then sleep outside the lock."""
        async with self._gate:
            now = self._clock()
            start_at = now if self._next_allowed is None else max(now, self._next_allowed)
            self._next_allowed = start_at + self._interval
            delay = start_at - now
        await self._pause(delay)

    async def _reserve(self, model: str, tokens: int) -> _Entry | None:
        """Book `tokens` against this model's window, waiting until they fit."""
        if not self.limits_key(model):
            return None
        while True:
            async with self._gate:
                now = self._clock()
                window = self.window_for(model)
                delay = window.wait_for(now, tokens)
                if delay <= 0.0:
                    return window.record(now, tokens)
            logger.info(
                "provider.token_budget_wait",
                extra={
                    "model": model,
                    "delay_seconds": round(delay, 3),
                    "tokens": tokens,
                    "tokens_per_minute": self._tokens_per_minute,
                },
            )
            await self._pause(delay)

    async def _pause(self, delay: float) -> None:
        if delay <= 0:
            return
        self.waited_seconds += delay
        await self._sleep(delay)

    @asynccontextmanager
    async def slot(
        self, *, model: str = "", estimated_tokens: int = 0
    ) -> AsyncIterator[Reservation]:
        """Hold one provider slot for the duration of a call.

        Yields the `Reservation`, which the caller reconciles with the usage the
        provider reports. `CostMeter.call` does that for every metered call, so
        no agent has to remember to.
        """
        async with self._semaphore:
            if self._interval > 0:
                await self._wait_turn()
            entry = (
                await self._reserve(model, max(1, estimated_tokens)) if estimated_tokens else None
            )
            yield Reservation(entry)
