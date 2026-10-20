import asyncio

import pytest

from switchboard.core.clock import Clock, RealClock, VirtualClock


def test_virtual_clock_starts_at_given_time() -> None:
    assert VirtualClock().now() == 0.0
    assert VirtualClock(start=12.5).now() == 12.5


def test_both_clocks_satisfy_protocol() -> None:
    clocks: list[Clock] = [VirtualClock(), RealClock()]
    assert all(isinstance(c.now(), float) for c in clocks)


async def test_real_clock_sleep_zero_returns() -> None:
    clock = RealClock()
    before = clock.now()
    await clock.sleep(0)
    assert clock.now() >= before


async def test_sleep_does_not_finish_until_advanced() -> None:
    clock = VirtualClock()
    task = asyncio.create_task(clock.sleep(5))
    await clock.advance(4.9)
    assert not task.done()
    await clock.advance(0.1)
    assert task.done()
    assert clock.now() == pytest.approx(5.0)


async def test_advance_fires_timers_in_time_order_with_ties_by_registration() -> None:
    clock = VirtualClock()
    order: list[tuple[str, float]] = []

    async def sleeper(name: str, delay: float) -> None:
        await clock.sleep(delay)
        order.append((name, clock.now()))

    tasks = [
        asyncio.create_task(sleeper("c", 3)),
        asyncio.create_task(sleeper("a", 1)),
        asyncio.create_task(sleeper("b1", 2)),
        asyncio.create_task(sleeper("b2", 2)),
    ]
    await clock.advance(10)
    assert order == [("a", 1.0), ("b1", 2.0), ("b2", 2.0), ("c", 3.0)]
    assert clock.now() == 10.0
    assert all(t.done() for t in tasks)


async def test_woken_coroutine_runs_before_next_timer() -> None:
    clock = VirtualClock()
    trace: list[str] = []

    async def first() -> None:
        await clock.sleep(1)
        trace.append("first woke")
        # Several hops through the loop before recording: must still precede "second".
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        trace.append("first done")

    async def second() -> None:
        await clock.sleep(2)
        trace.append("second woke")

    asyncio.create_task(first())
    asyncio.create_task(second())
    await clock.advance(5)
    assert trace == ["first woke", "first done", "second woke"]


async def test_timers_registered_during_advance_fire_within_window() -> None:
    clock = VirtualClock()
    times: list[float] = []

    async def chain() -> None:
        for _ in range(3):
            await clock.sleep(1)
            times.append(clock.now())

    asyncio.create_task(chain())
    await clock.advance(2.5)
    assert times == [1.0, 2.0]
    await clock.advance(1)
    assert times == [1.0, 2.0, 3.0]


async def test_cancelled_sleeper_is_skipped() -> None:
    clock = VirtualClock()
    task = asyncio.create_task(clock.sleep(1))
    await asyncio.sleep(0)
    task.cancel()
    await clock.advance(2)
    assert task.cancelled()


async def test_advance_rejects_negative() -> None:
    with pytest.raises(ValueError):
        await VirtualClock().advance(-1)


async def test_run_until_idle_drains_all_timers() -> None:
    clock = VirtualClock()
    done: list[float] = []

    async def worker(delays: list[float]) -> None:
        for d in delays:
            await clock.sleep(d)
        done.append(clock.now())

    asyncio.create_task(worker([1, 2, 3]))
    asyncio.create_task(worker([10]))
    await clock.run_until_idle()
    assert sorted(done) == [6.0, 10.0]
    assert clock.now() == 10.0


async def test_run_until_idle_waits_for_runnable_tasks_that_register_timers_later() -> None:
    clock = VirtualClock()
    done: list[float] = []

    async def late() -> None:
        for _ in range(5):
            await asyncio.sleep(0)
        await clock.sleep(3)
        done.append(clock.now())

    asyncio.create_task(late())
    await clock.run_until_idle()
    assert done == [3.0]


async def test_run_until_idle_with_nothing_scheduled_returns() -> None:
    clock = VirtualClock(start=4.0)
    await clock.run_until_idle()
    assert clock.now() == 4.0


async def test_run_until_idle_raises_on_endless_timer_loop() -> None:
    clock = VirtualClock()

    async def forever() -> None:
        while True:
            await clock.sleep(1)

    task = asyncio.create_task(forever())
    with pytest.raises(RuntimeError, match="idle"):
        await clock.run_until_idle(max_steps=50)
    task.cancel()


async def _scenario() -> list[tuple[str, float]]:
    clock = VirtualClock()
    trace: list[tuple[str, float]] = []

    async def actor(name: str, delays: list[float]) -> None:
        for d in delays:
            await clock.sleep(d)
            trace.append((name, clock.now()))

    for name, delays in [("x", [0.5, 0.5, 2]), ("y", [1, 1]), ("z", [0.25, 0.75, 1])]:
        asyncio.create_task(actor(name, delays))
    await clock.run_until_idle()
    return trace


async def test_virtual_clock_is_deterministic() -> None:
    first = await _scenario()
    second = await _scenario()
    assert first == second
    assert first[0] == ("z", 0.25)
