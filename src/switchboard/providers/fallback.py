"""Deadline-triggered TTS fallback (ADR-013).

A TTS provider that has not produced its first audio by the deadline on the clock is abandoned,
even if it has not errored: the caller hears dead air either way. Every chunk carries the name
of the provider that served it, so the runtime can record it on `speech.played`.

Unlike the LLM chain, the last provider has no deadline. There is no handoff tier above speech,
and late audio beats none. Once a provider has produced a chunk the utterance stays with it,
because switching voices mid-sentence is worse than a slow one.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from switchboard.core.clock import Clock
from switchboard.providers.interfaces import TTS, AudioChunk

Outcome = Literal["served", "deadline", "error", "empty"]


@dataclass(frozen=True, slots=True)
class TtsAttempt:
    provider: str
    outcome: Outcome
    started_at: float
    ended_at: float
    detail: str = ""


@dataclass(frozen=True, slots=True)
class TtsServeRecord:
    provider: str | None
    attempts: tuple[TtsAttempt, ...]


class FallbackTTS:
    def __init__(
        self, chain: Sequence[tuple[str, TTS]], clock: Clock, first_byte_deadline_s: float
    ) -> None:
        if not chain:
            raise ValueError("a fallback chain needs at least one provider")
        if first_byte_deadline_s <= 0:
            raise ValueError("first_byte_deadline_s must be positive")
        self._chain = tuple(chain)
        self._clock = clock
        self.first_byte_deadline_s = first_byte_deadline_s
        self.records: list[TtsServeRecord] = []

    @property
    def last(self) -> TtsServeRecord:
        if not self.records:
            raise LookupError("nothing has been synthesized yet")
        return self.records[-1]

    async def synthesize(self, text: str) -> AsyncGenerator[AudioChunk, None]:
        attempts: list[TtsAttempt] = []
        for i, (name, tts) in enumerate(self._chain):
            started = self._clock.now()
            stream = tts.synthesize(text)
            last = i == len(self._chain) - 1
            outcome, first, detail = await self._first_chunk(
                stream, None if last else self.first_byte_deadline_s
            )
            if outcome != "served" or first is None:
                await _close(stream)
                attempts.append(TtsAttempt(name, outcome, started, self._clock.now(), detail))
                if outcome == "empty":
                    break
                continue
            attempts.append(TtsAttempt(name, "served", started, self._clock.now()))
            self.records.append(TtsServeRecord(name, tuple(attempts)))
            try:
                yield dataclasses.replace(first, provider=name)
                async for chunk in stream:
                    yield dataclasses.replace(chunk, provider=name)
            finally:
                await _close(stream)
            return
        self.records.append(TtsServeRecord(None, tuple(attempts)))

    async def _first_chunk(
        self, stream: AsyncIterator[AudioChunk], deadline_s: float | None
    ) -> tuple[Outcome, AudioChunk | None, str]:
        async def pull() -> AudioChunk:
            return await anext(stream)

        first = asyncio.ensure_future(pull())
        waiting: set[asyncio.Future[Any]] = {first}
        timer: asyncio.Future[None] | None = None
        if deadline_s is not None:
            timer = asyncio.ensure_future(self._clock.sleep(deadline_s))
            waiting.add(timer)
        try:
            await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if timer is not None:
                await _cancel(timer)
            if not first.done():
                await _cancel(first)
        if first.cancelled():
            return "deadline", None, f"no audio within {deadline_s}s"
        exc = first.exception()
        if isinstance(exc, StopAsyncIteration):
            return "empty", None, "no audio for this text"
        if exc is not None:
            return "error", None, f"{type(exc).__name__}: {exc}"
        return "served", first.result(), ""


async def _cancel(task: asyncio.Future[Any]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _close(stream: AsyncIterator[AudioChunk]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        with contextlib.suppress(Exception):
            await aclose()
