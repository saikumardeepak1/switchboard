"""TTS deadline fallback on the virtual clock (ADR-013)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from switchboard.core import VirtualClock
from switchboard.providers import AudioChunk, FallbackTTS, SimulatedTTS


async def collect(clock: VirtualClock, stream: AsyncIterator[AudioChunk]) -> list[AudioChunk]:
    async def run() -> list[AudioChunk]:
        return [chunk async for chunk in stream]

    task = asyncio.ensure_future(run())
    await clock.run_until_idle()
    return task.result()


def chain(clock: VirtualClock, primary_ttfb: float) -> FallbackTTS:
    return FallbackTTS(
        [
            ("primary", SimulatedTTS(clock, ttfb_s=primary_ttfb, chars_per_s=20.0)),
            ("backup", SimulatedTTS(clock, ttfb_s=0.1, chars_per_s=20.0)),
        ],
        clock,
        first_byte_deadline_s=0.3,
    )


async def test_a_primary_inside_the_deadline_serves() -> None:
    clock = VirtualClock()
    tts = chain(clock, primary_ttfb=0.2)
    chunks = await collect(clock, tts.synthesize("Hello there."))
    assert {c.provider for c in chunks} == {"primary"}
    assert chunks[0].at == pytest.approx(0.2)
    assert tts.last.provider == "primary"


async def test_a_primary_past_the_deadline_falls_to_the_backup() -> None:
    clock = VirtualClock()
    tts = chain(clock, primary_ttfb=2.0)
    chunks = await collect(clock, tts.synthesize("Hello there."))
    assert {c.provider for c in chunks} == {"backup"}
    # Deadline at 0.3, then the backup's own 0.1 s to first byte.
    assert chunks[0].at == pytest.approx(0.4)
    assert "".join(c.text for c in chunks) == "Hello there."
    assert [(a.provider, a.outcome) for a in tts.last.attempts] == [
        ("primary", "deadline"),
        ("backup", "served"),
    ]


class Broken:
    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        raise RuntimeError("tts down")
        yield  # pragma: no cover


async def test_an_erroring_primary_falls_through_at_once() -> None:
    clock = VirtualClock()
    tts = FallbackTTS(
        [("primary", Broken()), ("backup", SimulatedTTS(clock, ttfb_s=0.1, chars_per_s=20.0))],
        clock,
        first_byte_deadline_s=0.3,
    )
    chunks = await collect(clock, tts.synthesize("Hi."))
    assert chunks[0].provider == "backup" and chunks[0].at == pytest.approx(0.1)
    assert tts.last.attempts[0].outcome == "error"


async def test_the_last_provider_is_waited_for_rather_than_leaving_dead_air() -> None:
    clock = VirtualClock()
    tts = FallbackTTS(
        [("only", SimulatedTTS(clock, ttfb_s=1.0, chars_per_s=20.0))],
        clock,
        first_byte_deadline_s=0.3,
    )
    chunks = await collect(clock, tts.synthesize("Hi."))
    assert chunks[0].provider == "only" and chunks[0].at == pytest.approx(1.0)
