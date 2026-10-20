import asyncio
from collections.abc import AsyncIterator

import pytest

from switchboard.core.clock import VirtualClock
from switchboard.turn.signals import TurnSignal
from switchboard.turn.speculative import CancelToken, SpeculativeTurnEngine

FIRST_CHUNK_S = 0.4
CHUNK_GAP_S = 0.05


class FakeBrain:
    """Streams three chunks: the first after FIRST_CHUNK_S, then one every CHUNK_GAP_S."""

    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock
        self.calls: list[tuple[str, CancelToken, float]] = []
        self.finished: list[str] = []

    async def generate(self, text: str, token: CancelToken) -> AsyncIterator[str]:
        self.calls.append((text, token, self.clock.now()))
        await self.clock.sleep(FIRST_CHUNK_S)
        for i in range(3):
            if i:
                await self.clock.sleep(CHUNK_GAP_S)
            yield f"{text}|{i}"
        self.finished.append(text)


class Consumer:
    def __init__(self, engine: SpeculativeTurnEngine, clock: VirtualClock) -> None:
        self.chunks: list[tuple[float, str]] = []
        self._engine = engine
        self._clock = clock
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        async for chunk in self._engine.committed_stream():
            self.chunks.append((self._clock.now(), chunk))


def signal(kind: str, at: float, text: str) -> TurnSignal:
    return TurnSignal(kind=kind, at=at, text=text, confidence=0.9)  # type: ignore[arg-type]


async def first_chunk_delay_after_end(speculate: bool) -> tuple[float, SpeculativeTurnEngine]:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine = SpeculativeTurnEngine(clock, brain.generate)
    consumer = Consumer(engine, clock)
    await clock.advance(0.2)
    if speculate:
        await engine.handle(signal("eager_end", clock.now(), "Where is my order?"))
    await clock.advance(0.3)
    end_at = clock.now()
    await engine.handle(signal("end", end_at, "where is  my order"))
    await clock.run_until_idle()
    await consumer.task
    return consumer.chunks[0][0] - end_at, engine


async def test_promote_releases_buffer_earlier_than_baseline() -> None:
    spec_delay, engine = await first_chunk_delay_after_end(speculate=True)
    base_delay, baseline = await first_chunk_delay_after_end(speculate=False)

    # Speculation started 300 virtual ms before the end signal, so it lands 300 ms sooner.
    assert spec_delay == pytest.approx(FIRST_CHUNK_S - 0.3)
    assert base_delay == pytest.approx(FIRST_CHUNK_S)
    assert base_delay - spec_delay == pytest.approx(0.3)

    m = engine.metrics
    assert (m.speculations_started, m.cancelled, m.promoted, m.restarted) == (1, 0, 1, 0)
    assert m.saved_ms == pytest.approx(300.0)
    assert baseline.metrics.saved_ms == pytest.approx(0.0)
    assert baseline.metrics.speculations_started == 0


async def test_promoted_stream_delivers_all_chunks_in_order_and_clears_speculative_flag() -> None:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine = SpeculativeTurnEngine(clock, brain.generate)
    consumer = Consumer(engine, clock)
    await engine.handle(signal("eager_end", clock.now(), "Hi there."))
    await asyncio.sleep(0)
    token = brain.calls[0][1]
    assert token.speculative
    await clock.advance(0.1)
    await engine.handle(signal("end", clock.now(), "hi there"))
    assert not token.speculative
    await clock.run_until_idle()
    await consumer.task
    assert [c for _, c in consumer.chunks] == ["Hi there.|0", "Hi there.|1", "Hi there.|2"]
    assert len(brain.calls) == 1


async def test_buffer_finished_before_end_is_released_at_once() -> None:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine = SpeculativeTurnEngine(clock, brain.generate)
    consumer = Consumer(engine, clock)
    await engine.handle(signal("eager_end", clock.now(), "ok"))
    await clock.advance(1.0)
    end_at = clock.now()
    await engine.handle(signal("end", end_at, "OK!"))
    await clock.run_until_idle()
    await consumer.task
    assert all(at == pytest.approx(end_at) for at, _ in consumer.chunks)
    assert len(consumer.chunks) == 3
    assert engine.metrics.saved_ms == pytest.approx(FIRST_CHUNK_S * 1000)


async def test_resumed_cancels_and_discards_speculation() -> None:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine = SpeculativeTurnEngine(clock, brain.generate)
    consumer = Consumer(engine, clock)
    await engine.handle(signal("eager_end", clock.now(), "I want"))
    await clock.advance(0.1)
    await engine.handle(signal("resumed", clock.now(), "I want"))
    token = brain.calls[0][1]
    assert token.cancelled
    assert engine.metrics.cancelled == 1
    assert engine.history[-1].outcome == "cancelled"

    await clock.advance(1.0)
    assert consumer.chunks == []
    assert brain.finished == []

    await engine.handle(signal("end", clock.now(), "I want a refund."))
    await clock.run_until_idle()
    await consumer.task
    assert [c for _, c in consumer.chunks] == [
        "I want a refund.|0",
        "I want a refund.|1",
        "I want a refund.|2",
    ]
    m = engine.metrics
    assert (m.speculations_started, m.cancelled, m.promoted, m.restarted) == (1, 1, 0, 0)
    assert not brain.calls[1][1].speculative


async def test_changed_text_restarts_with_final_text() -> None:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine = SpeculativeTurnEngine(clock, brain.generate)
    consumer = Consumer(engine, clock)
    await engine.handle(signal("eager_end", clock.now(), "Cancel my order."))
    await clock.advance(0.3)
    end_at = clock.now()
    await engine.handle(signal("end", end_at, "Cancel my order. Actually, no."))
    await clock.run_until_idle()
    await consumer.task

    assert brain.calls[0][1].cancelled
    assert brain.calls[1][0] == "Cancel my order. Actually, no."
    assert not brain.calls[1][1].speculative
    assert all(c.startswith("Cancel my order. Actually, no.|") for _, c in consumer.chunks)
    assert consumer.chunks[0][0] - end_at == pytest.approx(FIRST_CHUNK_S)
    m = engine.metrics
    assert (m.speculations_started, m.cancelled, m.promoted, m.restarted) == (1, 0, 0, 1)
    assert m.saved_ms == pytest.approx(0.0)
    assert engine.history[-1].outcome == "restarted"


async def test_second_eager_with_new_text_replaces_speculation() -> None:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine = SpeculativeTurnEngine(clock, brain.generate)
    await engine.handle(signal("eager_end", clock.now(), "one"))
    await clock.advance(0.1)
    await engine.handle(signal("eager_end", clock.now(), "one two"))
    assert brain.calls[0][1].cancelled
    assert engine.metrics.speculations_started == 2
    assert engine.metrics.cancelled == 1
    await engine.aclose()


async def test_generation_error_surfaces_in_committed_stream() -> None:
    clock = VirtualClock()

    async def broken(text: str, token: CancelToken) -> AsyncIterator[str]:
        await clock.sleep(0.1)
        yield "partial"
        raise RuntimeError("brain down")

    engine = SpeculativeTurnEngine(clock, broken)
    await engine.handle(signal("end", clock.now(), "hello"))
    got: list[str] = []
    with pytest.raises(RuntimeError, match="brain down"):

        async def drain() -> None:
            async for chunk in engine.committed_stream():
                got.append(chunk)

        task = asyncio.create_task(drain())
        await clock.run_until_idle()
        await task
    assert got == ["partial"]
