"""Injectable clocks (ADR-014).

Everything that waits or reads time takes a `Clock`. `VirtualClock` makes simulated runs
deterministic: time only moves when a test or the simulator calls `advance` or
`run_until_idle`.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from typing import Protocol, runtime_checkable

# Fallback when the loop does not expose its ready queue (non-CPython loops such as uvloop).
_FALLBACK_YIELDS = 50
# A settle that keeps finding runnable work this many times in a row is a busy loop.
_MAX_SETTLE_YIELDS = 100_000


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    """Wall-clock implementation. The only place allowed to touch real time."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class VirtualClock:
    """Deterministic clock: sleepers wake only when time is advanced.

    Timers fire in deadline order, ties broken by registration order. After each timer fires,
    the clock lets the event loop run until no callbacks are ready, so a woken coroutine (and
    whatever it wakes in turn) finishes its step before the next timer fires.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)
        self._counter = 0
        self._timers: list[tuple[float, int, asyncio.Future[None]]] = []

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._timers, (self._now + seconds, self._counter, future))
        self._counter += 1
        await future

    def pending_timers(self) -> int:
        return sum(1 for _, _, f in self._timers if not f.done())

    async def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"cannot advance by a negative amount: {seconds}")
        target = self._now + seconds
        await _settle()
        while (deadline := self._next_deadline()) is not None and deadline <= target:
            self._fire_next()
            await _settle()
        self._now = max(self._now, target)

    async def run_until_idle(self, max_steps: int = 100_000) -> None:
        """Advance timer by timer until no timers remain and no task is runnable.

        Raises RuntimeError after `max_steps` timers, which almost always means a coroutine
        re-arms a timer forever (a heartbeat or polling loop) and so can never go idle.
        """
        for _ in range(max_steps):
            await _settle()
            if self._next_deadline() is None:
                return
            self._fire_next()
        raise RuntimeError(
            f"VirtualClock.run_until_idle did not reach idle after {max_steps} timers; "
            "some task keeps scheduling new sleeps"
        )

    def _next_deadline(self) -> float | None:
        while self._timers and self._timers[0][2].done():
            heapq.heappop(self._timers)
        return self._timers[0][0] if self._timers else None

    def _fire_next(self) -> None:
        deadline, _, future = heapq.heappop(self._timers)
        self._now = max(self._now, deadline)
        future.set_result(None)


async def _settle() -> None:
    """Yield until the event loop has no ready callbacks other than this task."""
    loop = asyncio.get_running_loop()
    ready = getattr(loop, "_ready", None)
    if ready is None:
        for _ in range(_FALLBACK_YIELDS):
            await asyncio.sleep(0)
        return
    for _ in range(_MAX_SETTLE_YIELDS):
        await asyncio.sleep(0)
        if not ready:
            return
    raise RuntimeError("event loop never settled; a task is spinning on asyncio.sleep(0)")
