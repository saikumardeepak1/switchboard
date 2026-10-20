"""Speculative turn engine (ADR-006).

On `eager_end` the engine starts generation into a private buffer. `resumed` cancels it and
throws the buffer away. `end` either promotes the buffer (the final text matches what was
speculated on) or cancels it and starts again from the final text. Only promoted or freshly
started committed generations reach `committed_stream()`.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Literal

from switchboard.core.clock import Clock
from switchboard.turn.signals import TurnSignal


class CancelToken:
    """Handed to the generator. Carries cancellation and whether the turn is still speculative.

    While `speculative` is True the caller must refuse write tools (ADR-006). The flag drops to
    False when the speculation is promoted to a committed turn.
    """

    def __init__(self, *, speculative: bool) -> None:
        self.speculative = speculative
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True


StartGeneration = Callable[[str, CancelToken], AsyncIterator[str]]

SpeculationOutcome = Literal["pending", "cancelled", "promoted", "restarted"]


@dataclass(slots=True)
class SpeculationRecord:
    text: str
    started_at: float
    outcome: SpeculationOutcome = "pending"
    ended_at: float | None = None


@dataclass(slots=True)
class SpeculationMetrics:
    """Counters across the engine's lifetime.

    `cancelled` counts speculations dropped on `resumed` (or replaced by a newer eager text);
    `restarted` counts speculations dropped because the final text differed. `saved_ms` sums,
    over committed turns, how much of the generator's time to first chunk had already elapsed
    when the turn ended: first-chunk latency minus the wait after `end`.
    """

    speculations_started: int = 0
    cancelled: int = 0
    promoted: int = 0
    restarted: int = 0
    saved_ms: float = 0.0
    turn_saved_ms: list[float] = field(default_factory=list)


class _Generation:
    def __init__(self, text: str, token: CancelToken, started_at: float) -> None:
        self.text = text
        self.token = token
        self.started_at = started_at
        self.buffer: list[str] = []
        self.first_chunk_at: float | None = None
        self.committed_at: float | None = None
        self.saved_recorded = False
        self.done = False
        self.error: BaseException | None = None
        self.wakeup = asyncio.Event()
        self.task: asyncio.Task[None] | None = None


_PUNCT = re.compile(r"[^\w\s']")
_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Case, punctuation and whitespace insensitive form used to compare turn texts."""
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


class SpeculativeTurnEngine:
    def __init__(self, clock: Clock, start_generation: StartGeneration) -> None:
        self._clock = clock
        self._start = start_generation
        self._speculation: _Generation | None = None
        self._speculation_record: SpeculationRecord | None = None
        self._committed: asyncio.Queue[_Generation] = asyncio.Queue()
        self._live: set[_Generation] = set()
        self.metrics = SpeculationMetrics()
        self.history: list[SpeculationRecord] = []

    @property
    def speculating(self) -> bool:
        return self._speculation is not None

    async def handle(self, signal: TurnSignal) -> None:
        if signal.kind == "eager_end":
            await self._on_eager(signal)
        elif signal.kind == "resumed":
            await self._drop_speculation("cancelled")
        else:
            await self._on_end(signal)

    async def committed_stream(self) -> AsyncIterator[str]:
        """Yield the released output of the next committed turn, then stop.

        Raises whatever the generator raised, after yielding the chunks produced before it.
        """
        gen = await self._committed.get()
        index = 0
        while True:
            while index < len(gen.buffer):
                yield gen.buffer[index]
                index += 1
            if gen.done:
                if gen.error is not None:
                    raise gen.error
                return
            gen.wakeup.clear()
            await gen.wakeup.wait()

    async def aclose(self) -> None:
        """Cancel every running generation."""
        for gen in list(self._live):
            await self._cancel(gen)
        self._speculation = None

    async def _on_eager(self, signal: TurnSignal) -> None:
        current = self._speculation
        if current is not None:
            if normalize_text(current.text) == normalize_text(signal.text):
                return
            await self._drop_speculation("cancelled")
        self._speculation = self._launch(signal.text, speculative=True)
        self._speculation_record = SpeculationRecord(signal.text, self._clock.now())
        self.history.append(self._speculation_record)
        self.metrics.speculations_started += 1

    async def _on_end(self, signal: TurnSignal) -> None:
        spec = self._speculation
        if spec is not None and normalize_text(spec.text) == normalize_text(signal.text):
            self._speculation = None
            self._close_record("promoted")
            self.metrics.promoted += 1
            spec.token.speculative = False
            self._commit(spec)
            return
        if spec is not None:
            await self._drop_speculation("restarted")
        self._commit(self._launch(signal.text, speculative=False))

    def _commit(self, gen: _Generation) -> None:
        gen.committed_at = self._clock.now()
        self._record_saved(gen)
        self._committed.put_nowait(gen)

    def _record_saved(self, gen: _Generation) -> None:
        if gen.saved_recorded or gen.committed_at is None or gen.first_chunk_at is None:
            return
        gen.saved_recorded = True
        latency = gen.first_chunk_at - gen.started_at
        wait_after_end = max(0.0, gen.first_chunk_at - gen.committed_at)
        saved = (latency - wait_after_end) * 1000.0
        self.metrics.saved_ms += saved
        self.metrics.turn_saved_ms.append(saved)

    async def _drop_speculation(self, outcome: SpeculationOutcome) -> None:
        spec = self._speculation
        if spec is None:
            return
        self._speculation = None
        self._close_record(outcome)
        if outcome == "restarted":
            self.metrics.restarted += 1
        else:
            self.metrics.cancelled += 1
        await self._cancel(spec)

    def _close_record(self, outcome: SpeculationOutcome) -> None:
        if self._speculation_record is not None:
            self._speculation_record.outcome = outcome
            self._speculation_record.ended_at = self._clock.now()
            self._speculation_record = None

    def _launch(self, text: str, *, speculative: bool) -> _Generation:
        gen = _Generation(text, CancelToken(speculative=speculative), self._clock.now())
        gen.task = asyncio.create_task(self._pump(gen))
        self._live.add(gen)
        return gen

    async def _pump(self, gen: _Generation) -> None:
        try:
            async for chunk in self._start(gen.text, gen.token):
                if gen.token.cancelled:
                    break
                if gen.first_chunk_at is None:
                    gen.first_chunk_at = self._clock.now()
                    self._record_saved(gen)
                gen.buffer.append(chunk)
                gen.wakeup.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            gen.error = exc
        finally:
            gen.done = True
            gen.wakeup.set()
            self._live.discard(gen)

    async def _cancel(self, gen: _Generation) -> None:
        gen.token.cancel()
        gen.buffer.clear()
        task = gen.task
        if task is not None and not task.done():
            task.cancel()
            # wait() does not re-raise the task's CancelledError, so a cancellation aimed at
            # the caller still propagates.
            await asyncio.wait([task])
