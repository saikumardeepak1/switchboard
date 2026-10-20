import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from switchboard.core.clock import VirtualClock
from switchboard.turn.signals import TurnSignal
from switchboard.turn.speculative import GenerationContext, SpeculativeTurnEngine

FIRST_CHUNK_S = 0.4
CHUNK_GAP_S = 0.05


class FakeBrain:
    """Streams three chunks: the first after FIRST_CHUNK_S, then one every CHUNK_GAP_S."""

    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock
        self.calls: list[tuple[str, GenerationContext[str], float]] = []
        self.finished: list[str] = []

    async def generate(self, text: str, ctx: GenerationContext[str]) -> AsyncIterator[str]:
        self.calls.append((text, ctx, self.clock.now()))
        await self.clock.sleep(FIRST_CHUNK_S)
        for i in range(3):
            if i:
                await self.clock.sleep(CHUNK_GAP_S)
            yield f"{text}|{i}"
        self.finished.append(text)


class Consumer:
    def __init__(self, engine: SpeculativeTurnEngine[str, str], clock: VirtualClock) -> None:
        self.chunks: list[tuple[float, str]] = []
        self._engine = engine
        self._clock = clock
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        async for chunk in self._engine.committed_stream():
            self.chunks.append((self._clock.now(), chunk))


def signal(kind: str, at: float, text: str) -> TurnSignal:
    return TurnSignal(kind=kind, at=at, text=text, confidence=0.9)  # type: ignore[arg-type]


async def first_chunk_delay_after_end(
    speculate: bool,
) -> tuple[float, SpeculativeTurnEngine[str, str]]:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine: SpeculativeTurnEngine[str, str] = SpeculativeTurnEngine(clock, brain.generate)
    consumer = Consumer(engine, clock)
    await clock.advance(0.2)
    if speculate:
        await engine.handle(signal("eager_end", clock.now(), "Where is my order?"), "p")
    await clock.advance(0.3)
    end_at = clock.now()
    await engine.handle(signal("end", end_at, "where is  my order"), "p")
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
    engine: SpeculativeTurnEngine[str, str] = SpeculativeTurnEngine(clock, brain.generate)
    consumer = Consumer(engine, clock)
    await engine.handle(signal("eager_end", clock.now(), "Hi there."), "p")
    await asyncio.sleep(0)
    token = brain.calls[0][1]
    assert token.speculative
    await clock.advance(0.1)
    await engine.handle(signal("end", clock.now(), "hi there"), "p")
    assert not token.speculative
    await clock.run_until_idle()
    await consumer.task
    assert [c for _, c in consumer.chunks] == ["Hi there.|0", "Hi there.|1", "Hi there.|2"]
    assert len(brain.calls) == 1


async def test_buffer_finished_before_end_is_released_at_once() -> None:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine: SpeculativeTurnEngine[str, str] = SpeculativeTurnEngine(clock, brain.generate)
    consumer = Consumer(engine, clock)
    await engine.handle(signal("eager_end", clock.now(), "ok"), "p")
    await clock.advance(1.0)
    end_at = clock.now()
    await engine.handle(signal("end", end_at, "OK!"), "p")
    await clock.run_until_idle()
    await consumer.task
    assert all(at == pytest.approx(end_at) for at, _ in consumer.chunks)
    assert len(consumer.chunks) == 3
    assert engine.metrics.saved_ms == pytest.approx(FIRST_CHUNK_S * 1000)


async def test_resumed_cancels_and_discards_speculation() -> None:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine: SpeculativeTurnEngine[str, str] = SpeculativeTurnEngine(clock, brain.generate)
    consumer = Consumer(engine, clock)
    await engine.handle(signal("eager_end", clock.now(), "I want"), "p")
    await clock.advance(0.1)
    await engine.handle(signal("resumed", clock.now(), "I want"))
    token = brain.calls[0][1]
    assert token.cancelled
    assert engine.metrics.cancelled == 1
    assert engine.history[-1].outcome == "cancelled"

    await clock.advance(1.0)
    assert consumer.chunks == []
    assert brain.finished == []

    await engine.handle(signal("end", clock.now(), "I want a refund."), "p")
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
    engine: SpeculativeTurnEngine[str, str] = SpeculativeTurnEngine(clock, brain.generate)
    consumer = Consumer(engine, clock)
    await engine.handle(signal("eager_end", clock.now(), "Cancel my order."), "p")
    await clock.advance(0.3)
    end_at = clock.now()
    await engine.handle(signal("end", end_at, "Cancel my order. Actually, no."), "p")
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
    engine: SpeculativeTurnEngine[str, str] = SpeculativeTurnEngine(clock, brain.generate)
    await engine.handle(signal("eager_end", clock.now(), "one"), "p")
    await clock.advance(0.1)
    await engine.handle(signal("eager_end", clock.now(), "one two"), "p")
    assert brain.calls[0][1].cancelled
    assert engine.metrics.speculations_started == 2
    assert engine.metrics.cancelled == 1
    await engine.aclose()


async def test_generation_error_surfaces_in_committed_stream() -> None:
    clock = VirtualClock()

    async def broken(text: str, ctx: GenerationContext[str]) -> AsyncIterator[str]:
        await clock.sleep(0.1)
        yield "partial"
        raise RuntimeError("brain down")

    engine: SpeculativeTurnEngine[str, str] = SpeculativeTurnEngine(clock, broken)
    await engine.handle(signal("end", clock.now(), "hello"), "p")
    got: list[str] = []
    with pytest.raises(RuntimeError, match="brain down"):

        async def drain() -> None:
            async for chunk in engine.committed_stream():
                got.append(chunk)

        task = asyncio.create_task(drain())
        await clock.run_until_idle()
        await task
    assert got == ["partial"]


@dataclass(frozen=True)
class Item:
    """A typed, non-string output, the way a runtime buffers gated sentences."""

    turn_id: str
    payload: str
    n: int


async def test_context_carries_turn_id_payload_and_typed_items() -> None:
    clock = VirtualClock()
    seen: list[GenerationContext[str]] = []

    async def generate(text: str, ctx: GenerationContext[str]) -> AsyncIterator[Item]:
        seen.append(ctx)
        await clock.sleep(0.1)
        for n in range(2):
            yield Item(ctx.turn_id, ctx.payload, n)

    engine: SpeculativeTurnEngine[Item, str] = SpeculativeTurnEngine(clock, generate)
    await engine.handle(signal("eager_end", clock.now(), "first try"), "spec", turn_id="t1")
    await asyncio.sleep(0)
    assert seen[0].speculative and seen[0].turn_id == "t1" and seen[0].payload == "spec"
    # The final text differs, so the engine restarts with the context given for the commit.
    await engine.handle(signal("end", clock.now(), "second try"), "fresh", turn_id="t2")

    got: list[Item] = []

    async def drain() -> None:
        async for item in engine.committed_stream():
            got.append(item)

    task = asyncio.create_task(drain())
    await clock.run_until_idle()
    await task
    assert seen[0].cancelled
    assert not seen[1].speculative and seen[1].turn_id == "t2"
    assert got == [Item("t2", "fresh", 0), Item("t2", "fresh", 1)]


async def test_promotion_keeps_the_speculation_context() -> None:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine: SpeculativeTurnEngine[str, str] = SpeculativeTurnEngine(clock, brain.generate)
    await engine.handle(signal("eager_end", clock.now(), "hello"), "spec", turn_id="t1")
    await asyncio.sleep(0)
    await engine.handle(signal("end", clock.now(), "Hello."), "ignored", turn_id="t9")
    (call,) = brain.calls
    assert (call[1].turn_id, call[1].payload, call[1].speculative) == ("t1", "spec", False)
    await engine.aclose()


async def test_a_signal_that_launches_needs_a_payload() -> None:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine: SpeculativeTurnEngine[str, str] = SpeculativeTurnEngine(clock, brain.generate)
    with pytest.raises(ValueError, match="payload"):
        await engine.handle(signal("eager_end", clock.now(), "hello"))
    await engine.handle(signal("resumed", clock.now(), "hello"))
    assert brain.calls == []


async def test_cancel_committed_spares_the_speculation() -> None:
    clock = VirtualClock()
    brain = FakeBrain(clock)
    engine: SpeculativeTurnEngine[str, str] = SpeculativeTurnEngine(clock, brain.generate)
    consumer = Consumer(engine, clock)
    await engine.handle(signal("end", clock.now(), "first turn"), "a")
    await clock.advance(0.1)
    await engine.handle(signal("eager_end", clock.now(), "second turn"), "b")
    await asyncio.sleep(0)
    await engine.cancel_committed()
    committed, speculation = brain.calls[0][1], brain.calls[1][1]
    assert committed.cancelled and not speculation.cancelled
    assert engine.speculating
    await engine.handle(signal("end", clock.now(), "second turn"), "b")
    await clock.run_until_idle()
    await consumer.task
    assert consumer.chunks == []  # the cancelled turn released nothing
    second = Consumer(engine, clock)
    await clock.run_until_idle()
    await second.task
    assert [c for _, c in second.chunks] == [f"second turn|{i}" for i in range(3)]
