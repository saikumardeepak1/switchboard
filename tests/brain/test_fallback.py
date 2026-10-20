import asyncio
from collections.abc import AsyncIterator

import pytest

from switchboard.brain.fallback import FallbackExhaustedError, FallbackLLM
from switchboard.brain.llm import LLM, ScriptedLLM, ScriptStep
from switchboard.brain.types import BrainChunk, BrainRequest, Done, Message, TextDelta
from switchboard.core.clock import VirtualClock

REQ = BrainRequest(messages=(Message(role="user", content="where is my order"),))


def scripted(clock: VirtualClock, text: str, ttft: float) -> ScriptedLLM:
    return ScriptedLLM(clock, [ScriptStep(None, text=text)], ttft_s=ttft, tokens_per_s=10)


class Exploding:
    def __init__(self, clock: VirtualClock, after_s: float, after_chunks: int = 0) -> None:
        self.clock = clock
        self.after_s = after_s
        self.after_chunks = after_chunks

    async def stream(self, request: BrainRequest) -> AsyncIterator[BrainChunk]:
        await self.clock.sleep(self.after_s)
        for i in range(self.after_chunks):
            yield TextDelta(f"part{i} ")
        raise ConnectionError("provider down")


class Empty:
    async def stream(self, request: BrainRequest) -> AsyncIterator[BrainChunk]:
        return
        yield  # pragma: no cover


async def run(llm: LLM, clock: VirtualClock) -> list[tuple[float, BrainChunk]]:
    out: list[tuple[float, BrainChunk]] = []

    async def consume() -> None:
        async for chunk in llm.stream(REQ):
            out.append((clock.now(), chunk))

    task = asyncio.create_task(consume())
    await clock.run_until_idle()
    await task
    return out


async def test_fast_primary_serves() -> None:
    clock = VirtualClock()
    primary = scripted(clock, "primary answer", 0.3)
    backup = scripted(clock, "backup answer", 0.1)
    fb = FallbackLLM([("primary", primary), ("backup", backup)], clock, ttft_deadline_s=0.8)
    out = await run(fb, clock)
    assert out[0] == (pytest.approx(0.3), TextDelta("primary "))
    assert fb.last.provider == "primary"
    assert [a.outcome for a in fb.last.attempts] == ["served"]
    assert backup.requests == []


async def test_slow_primary_falls_back_on_deadline_at_expected_virtual_time() -> None:
    clock = VirtualClock()
    slow = scripted(clock, "too late", 2.0)
    backup = scripted(clock, "backup answer", 0.25)
    fb = FallbackLLM([("slow", slow), ("backup", backup)], clock, ttft_deadline_s=0.8)
    out = await run(fb, clock)
    # deadline at 0.8, backup ttft 0.25 -> first chunk at 1.05
    assert out[0] == (pytest.approx(1.05), TextDelta("backup "))
    text = "".join(c.text for _, c in out if isinstance(c, TextDelta))
    assert text == "backup answer"
    assert isinstance(out[-1][1], Done)
    rec = fb.last
    assert rec.provider == "backup"
    assert [(a.provider, a.outcome) for a in rec.attempts] == [
        ("slow", "deadline"),
        ("backup", "served"),
    ]
    assert rec.attempts[0].ended_at == pytest.approx(0.8)
    assert slow.cancelled == 1
    assert clock.pending_timers() == 0


async def test_error_before_first_chunk_falls_back_immediately() -> None:
    clock = VirtualClock()
    fb = FallbackLLM(
        [("broken", Exploding(clock, after_s=0.1)), ("backup", scripted(clock, "ok", 0.2))],
        clock,
        ttft_deadline_s=0.8,
    )
    out = await run(fb, clock)
    assert out[0] == (pytest.approx(0.3), TextDelta("ok"))
    first = fb.last.attempts[0]
    assert (first.outcome, first.ended_at) == ("error", pytest.approx(0.1))
    assert "provider down" in first.detail


async def test_empty_stream_counts_as_failure() -> None:
    clock = VirtualClock()
    fb = FallbackLLM([("empty", Empty()), ("backup", scripted(clock, "ok", 0.1))], clock, 0.5)
    await run(fb, clock)
    assert [a.outcome for a in fb.last.attempts] == ["empty", "served"]


async def test_sticks_with_provider_after_first_chunk_even_if_it_fails_later() -> None:
    clock = VirtualClock()
    backup = scripted(clock, "backup", 0.1)
    fb = FallbackLLM(
        [("flaky", Exploding(clock, after_s=0.1, after_chunks=1)), ("backup", backup)],
        clock,
        ttft_deadline_s=0.5,
    )
    got: list[BrainChunk] = []

    async def consume() -> None:
        async for chunk in fb.stream(REQ):
            got.append(chunk)

    task = asyncio.create_task(consume())
    await clock.run_until_idle()
    with pytest.raises(ConnectionError):
        await task
    assert got == [TextDelta("part0 ")]
    assert backup.requests == []
    assert fb.last.provider == "flaky"


async def test_slow_late_chunks_after_first_are_not_subject_to_deadline() -> None:
    clock = VirtualClock()
    slow_tokens = ScriptedLLM(clock, [ScriptStep(None, text="a b c")], ttft_s=0.1, tokens_per_s=1)
    fb = FallbackLLM([("p", slow_tokens)], clock, ttft_deadline_s=0.5)
    out = await run(fb, clock)
    assert [t for t, c in out if isinstance(c, TextDelta)] == [
        pytest.approx(0.1),
        pytest.approx(1.1),
        pytest.approx(2.1),
    ]


async def test_all_providers_fail_raises_with_attempts() -> None:
    clock = VirtualClock()
    fb = FallbackLLM(
        [("a", scripted(clock, "x", 5.0)), ("b", Exploding(clock, after_s=0.2))],
        clock,
        ttft_deadline_s=1.0,
    )

    async def consume() -> None:
        async for _ in fb.stream(REQ):
            pass

    task = asyncio.create_task(consume())
    await clock.run_until_idle()
    with pytest.raises(FallbackExhaustedError) as info:
        await task
    assert [a.outcome for a in info.value.attempts] == ["deadline", "error"]
    assert fb.last.provider is None
    assert clock.now() == pytest.approx(1.2)


async def test_consumer_cancel_while_waiting_cleans_up() -> None:
    clock = VirtualClock()
    slow = scripted(clock, "x", 5.0)
    fb = FallbackLLM([("slow", slow)], clock, ttft_deadline_s=1.0)

    async def consume() -> None:
        async for _ in fb.stream(REQ):
            pass

    task = asyncio.create_task(consume())
    await clock.advance(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert clock.pending_timers() == 0
    assert slow.cancelled == 1


def test_empty_chain_rejected() -> None:
    with pytest.raises(ValueError):
        FallbackLLM([], VirtualClock(), 1.0)
