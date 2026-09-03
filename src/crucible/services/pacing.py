"""Pacing for provider calls.

Free-tier rate limits are the binding constraint on this project. Backoff in
`crucible.services.retry` reacts to a 429 after it has already cost a request;
this reacts before, by holding every provider call to a minimum interval and a
maximum concurrency. A live reclassify run met its limit 23 calls into a
40-call loop with no pacing at all, which is what this exists to prevent.

The pacer is owned by `CostMeter`, because "every LLM call routes through
`CostMeter`" is already a project invariant and that makes this one too.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from crucible.logging import get_logger

logger = get_logger(__name__)

SleepFn = Callable[[float], Awaitable[None]]
ClockFn = Callable[[], float]


class ProviderPacer:
    """Serialises provider calls to at most `max_concurrency`, spaced apart.

    The interval is measured between call *starts*, so a slow call does not add
    its own duration to the next one's wait.
    """

    def __init__(
        self,
        *,
        min_interval_seconds: float = 0.0,
        max_concurrency: int = 1,
        sleep: SleepFn = asyncio.sleep,
        clock: ClockFn = time.monotonic,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds cannot be negative")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        self._interval = min_interval_seconds
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._gate = asyncio.Lock()
        self._sleep = sleep
        self._clock = clock
        self._next_allowed: float | None = None
        #: Total seconds spent waiting, for tests and for the round log.
        self.waited_seconds = 0.0

    @classmethod
    def unlimited(cls) -> ProviderPacer:
        """No spacing and no bound: the default for stubbed, offline runs."""
        return cls(min_interval_seconds=0.0, max_concurrency=_UNBOUNDED)

    @classmethod
    def from_settings(cls, min_interval_seconds: float, max_concurrency: int) -> ProviderPacer:
        return cls(min_interval_seconds=min_interval_seconds, max_concurrency=max_concurrency)

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def min_interval_seconds(self) -> float:
        return self._interval

    async def _wait_turn(self) -> None:
        """Claim the next slot in the schedule, then sleep outside the lock."""
        async with self._gate:
            now = self._clock()
            start_at = now if self._next_allowed is None else max(now, self._next_allowed)
            self._next_allowed = start_at + self._interval
            delay = start_at - now
        if delay > 0:
            self.waited_seconds += delay
            logger.debug("provider.paced", extra={"delay_seconds": round(delay, 3)})
            await self._sleep(delay)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one provider slot for the duration of a call."""
        async with self._semaphore:
            if self._interval > 0:
                await self._wait_turn()
            yield


#: Effectively no bound, while keeping the semaphore's type honest.
_UNBOUNDED = 1024
