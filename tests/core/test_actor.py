import asyncio

import pytest

from switchboard.core.actor import Actor, ActorRuntime
from switchboard.core.clock import VirtualClock


class Recorder:
    """Sleeps `delay` per message so overlapping handlers would be visible."""

    def __init__(self, key: str, clock: VirtualClock, log: list[tuple[str, str, float]]):
        self.key = key
        self.clock = clock
        self.log = log
        self.active = 0
        self.max_active = 0

    async def handle(self, message: object) -> object:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.log.append((self.key, f"start {message}", self.clock.now()))
        try:
            if message == "boom":
                raise ValueError("boom")
            await self.clock.sleep(1)
            self.log.append((self.key, f"end {message}", self.clock.now()))
            return f"{self.key}:{message}"
        finally:
            self.active -= 1


Log = list[tuple[str, str, float]]


def make_runtime() -> tuple[ActorRuntime, VirtualClock, Log, dict[str, Recorder]]:
    clock = VirtualClock()
    log: list[tuple[str, str, float]] = []
    actors: dict[str, Recorder] = {}

    def factory(key: str) -> Actor[object]:
        actors[key] = Recorder(key, clock, log)
        return actors[key]

    return ActorRuntime(factory, clock), clock, log, actors


async def test_fifo_and_one_at_a_time_per_key() -> None:
    rt, clock, log, actors = make_runtime()
    for m in ["m1", "m2", "m3"]:
        await rt.send("k", m)
    await clock.run_until_idle()
    assert log == [
        ("k", "start m1", 0.0),
        ("k", "end m1", 1.0),
        ("k", "start m2", 1.0),
        ("k", "end m2", 2.0),
        ("k", "start m3", 2.0),
        ("k", "end m3", 3.0),
    ]
    assert actors["k"].max_active == 1
    await rt.stop_all()


async def test_keys_run_concurrently() -> None:
    rt, clock, log, _ = make_runtime()
    await rt.send("a", "x")
    await rt.send("b", "y")
    await clock.advance(1)
    ends = [(k, t) for k, what, t in log if what.startswith("end")]
    assert sorted(ends) == [("a", 1.0), ("b", 1.0)]
    await rt.stop_all()


async def test_factory_called_once_per_key() -> None:
    calls: list[str] = []
    clock = VirtualClock()

    class Echo:
        async def handle(self, message: object) -> object:
            return message

    def factory(key: str) -> Actor[object]:
        calls.append(key)
        return Echo()

    rt = ActorRuntime(factory, clock)
    assert await rt.ask("a", 1) == 1
    assert await rt.ask("a", 2) == 2
    assert await rt.ask("b", 3) == 3
    assert calls == ["a", "b"]
    await rt.stop_all()


async def test_ask_returns_handler_value() -> None:
    rt, clock, _, _ = make_runtime()
    reply = asyncio.create_task(rt.ask("k", "q"))
    await clock.run_until_idle()
    assert await reply == "k:q"
    await rt.stop_all()


async def test_ask_raises_handler_exception_to_asker() -> None:
    rt, clock, _, _ = make_runtime()
    with pytest.raises(ValueError, match="boom"):
        await rt.ask("k", "boom")
    assert rt.failures == []
    await rt.stop_all()


async def test_send_failure_is_recorded_and_isolated() -> None:
    rt, clock, log, _ = make_runtime()
    await rt.send("k", "boom")
    await rt.send("k", "after")
    await rt.send("other", "fine")
    await clock.run_until_idle()
    assert len(rt.failures) == 1
    failure = rt.failures[0]
    assert failure.key == "k"
    assert failure.message == "boom"
    assert isinstance(failure.error, ValueError)
    assert ("k", "end after", 1.0) in log
    assert ("other", "end fine", 1.0) in log
    await rt.stop_all()


async def test_stop_drains_queued_messages_then_stops() -> None:
    rt, clock, log, actors = make_runtime()
    for m in ["m1", "m2"]:
        await rt.send("k", m)
    stopping = asyncio.create_task(rt.stop("k"))
    await clock.run_until_idle()
    await stopping
    assert [what for _, what, _ in log if what.startswith("end")] == ["end m1", "end m2"]
    assert "k" not in rt.active_keys()
    # A later message activates a fresh actor for the key.
    first = actors["k"]
    reply = asyncio.create_task(rt.ask("k", "again"))
    await clock.run_until_idle()
    assert await reply == "k:again"
    assert actors["k"] is not first
    await rt.stop_all()
    assert rt.active_keys() == set()


async def test_stop_unknown_key_is_noop() -> None:
    rt, _, _, _ = make_runtime()
    await rt.stop("nobody")
