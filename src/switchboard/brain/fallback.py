"""Deadline-triggered provider fallback (ADR-013).

Dead air is the failure callers notice first, so a provider that has not produced its first
chunk by the deadline is abandoned even if it has not errored. Once any chunk has been passed
on, the stream belongs to that provider: switching mid-answer would splice two different
replies together.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from switchboard.brain.llm import LLM
from switchboard.brain.types import BrainChunk, BrainRequest
from switchboard.core.clock import Clock

Outcome = Literal["served", "deadline", "error", "empty"]


@dataclass(frozen=True, slots=True)
class Attempt:
    provider: str
    outcome: Outcome
    started_at: float
    ended_at: float
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ServeRecord:
    """Which provider served one request, and every attempt before it."""

    provider: str | None
    attempts: tuple[Attempt, ...]


class FallbackExhaustedError(RuntimeError):
    """No provider produced a first chunk in time. The caller should hand off to a human."""

    def __init__(self, attempts: Sequence[Attempt]) -> None:
        self.attempts = tuple(attempts)
        summary = ", ".join(f"{a.provider}: {a.outcome}" for a in self.attempts)
        super().__init__(f"every provider failed before the first chunk ({summary})")


class FallbackLLM:
    """Try providers in order; each gets `ttft_deadline_s` on the clock to produce a chunk.

    A first chunk that arrives exactly at the deadline counts as on time. The deadline also
    applies to the last provider, which then raises `FallbackExhaustedError`, because the chain
    is expected to end in a human-handoff tier above this class.
    """

    def __init__(
        self,
        chain: Sequence[tuple[str, LLM]],
        clock: Clock,
        ttft_deadline_s: float,
    ) -> None:
        if not chain:
            raise ValueError("a fallback chain needs at least one provider")
        if ttft_deadline_s <= 0:
            raise ValueError("ttft_deadline_s must be positive")
        self._chain = tuple(chain)
        self._clock = clock
        self.ttft_deadline_s = ttft_deadline_s
        self.records: list[ServeRecord] = []

    @property
    def last(self) -> ServeRecord:
        if not self.records:
            raise LookupError("no request has been served yet")
        return self.records[-1]

    async def stream(self, request: BrainRequest) -> AsyncGenerator[BrainChunk, None]:
        attempts: list[Attempt] = []
        for name, llm in self._chain:
            started = self._clock.now()
            stream = llm.stream(request)
            outcome, first, detail = await self._first_chunk(stream)
            if outcome != "served" or first is None:
                await _close(stream)
                attempts.append(Attempt(name, outcome, started, self._clock.now(), detail))
                continue
            attempts.append(Attempt(name, "served", started, self._clock.now()))
            self.records.append(ServeRecord(provider=name, attempts=tuple(attempts)))
            try:
                yield first
                async for chunk in stream:
                    yield chunk
            finally:
                await _close(stream)
            return
        self.records.append(ServeRecord(provider=None, attempts=tuple(attempts)))
        raise FallbackExhaustedError(attempts)

    async def _first_chunk(
        self, stream: AsyncIterator[BrainChunk]
    ) -> tuple[Outcome, BrainChunk | None, str]:
        async def pull() -> BrainChunk:
            return await anext(stream)

        first = asyncio.ensure_future(pull())
        timer = asyncio.ensure_future(self._clock.sleep(self.ttft_deadline_s))
        try:
            await asyncio.wait({first, timer}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            await _cancel(timer)
            if not first.done():
                await _cancel(first)
        if first.cancelled():
            return "deadline", None, f"no first chunk within {self.ttft_deadline_s}s"
        exc = first.exception()
        if isinstance(exc, StopAsyncIteration):
            return "empty", None, "stream ended without a chunk"
        if exc is not None:
            return "error", None, f"{type(exc).__name__}: {exc}"
        return "served", first.result(), ""


async def _cancel(task: asyncio.Future[Any]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _close(stream: AsyncIterator[BrainChunk]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()
