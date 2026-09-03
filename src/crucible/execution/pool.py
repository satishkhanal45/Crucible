"""A bounded pool of isolated targets.

docs/spec.md section 9 requires bounded concurrency with isolated sessions and
no shared mutable target state. Attacks inject and delete documents, so two
attempts running at once must not share a corpus: each pool slot owns a target
with its own corpus namespace, built lazily the first time the slot is used.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from crucible.logging import get_logger
from crucible.target.adapter import TargetAdapter

logger = get_logger(__name__)


def worker_namespace(index: int) -> str:
    """Name of the private corpus namespace a worker slot owns."""
    return f"worker-{index}"


TargetFactory = Callable[[str], Awaitable[TargetAdapter]]


class TargetPool:
    """Hands out one isolated target at a time, up to `size` concurrently."""

    def __init__(self, factory: TargetFactory, size: int = 5) -> None:
        if size < 1:
            raise ValueError("a target pool needs at least one slot")
        self._factory = factory
        self._size = size
        self._slots: asyncio.Queue[int] = asyncio.Queue()
        for index in range(size):
            self._slots.put_nowait(index)
        self._targets: dict[int, TargetAdapter] = {}
        self._build_lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return self._size

    @property
    def built(self) -> int:
        return len(self._targets)

    async def _target_for(self, index: int) -> TargetAdapter:
        existing = self._targets.get(index)
        if existing is not None:
            return existing
        async with self._build_lock:
            existing = self._targets.get(index)
            if existing is None:
                namespace = worker_namespace(index)
                logger.info("pool.building_target", extra={"namespace": namespace})
                existing = await self._factory(namespace)
                self._targets[index] = existing
        return existing

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[TargetAdapter]:
        index = await self._slots.get()
        try:
            yield await self._target_for(index)
        finally:
            self._slots.put_nowait(index)
