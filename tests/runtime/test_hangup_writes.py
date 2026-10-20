"""Hanging up during a write never leaves a landed effect marked failed (ADR-009)."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from switchboard.effects import IdempotencyLedger

from .harness import SimParty, World, every, make_world
from .test_scenarios import report_stuck_order, verify


def slow_mark(world: World, after_apply_s: float) -> None:
    """mark_delivered applies downstream, then its response takes `after_apply_s` to arrive."""
    registry = world.bundle.registry
    orig = registry.get("mark_delivered")
    assert orig is not None

    async def fn(order_id: str, location: str, *, idempotency_key: str) -> dict[str, Any]:
        out = await orig.fn(order_id=order_id, location=location, idempotency_key=idempotency_key)
        await world.clock.sleep(after_apply_s)
        return dict(out)

    registry._tools["mark_delivered"] = dataclasses.replace(orig, fn=fn)


async def hang_up_mid_write(world: World) -> Any:
    conv = await world.start()

    async def courier(c: SimParty) -> None:
        await verify(c)
        await report_stuck_order(c)
        while world.backend.downstream.applied["mark_delivered"] == 0:
            await c.clock.sleep(0.05)
        await conv.end("hangup")

    task = asyncio.ensure_future(courier(world.courier))
    await world.clock.run_until_idle()
    assert task.done()
    task.result()
    return conv


async def entry_state(ledger: IdempotencyLedger, key: str) -> str | None:
    entry = await ledger.get(key)
    return None if entry is None else entry.state


async def test_hang_up_during_a_write_waits_for_it_and_records_it() -> None:
    world = make_world()
    slow_mark(world, after_apply_s=1.0)
    conv = await hang_up_mid_write(world)
    events = await world.events(conv)
    (executed,) = every(events, "tool.executed", tool="mark_delivered")
    assert world.backend.downstream.applied["mark_delivered"] == 1
    assert executed.idempotency_key is not None
    assert await entry_state(world.ledger, executed.idempotency_key) == "done"
    assert events.index(executed) < events.index(every(events, "conversation.ended")[0])
    assert every(events, "tool.unresolved") == []


async def test_a_write_still_running_at_the_bound_is_left_pending_and_noted() -> None:
    world = make_world()
    slow_mark(world, after_apply_s=4.0)
    world.runtime.end_write_timeout_s = 2.0
    conv = await hang_up_mid_write(world)
    events = await world.events(conv)
    assert every(events, "tool.executed", tool="mark_delivered") == []
    (unresolved,) = every(events, "tool.unresolved")
    assert unresolved.tool == "mark_delivered"
    assert unresolved.idempotency_key is not None
    assert await entry_state(world.ledger, unresolved.idempotency_key) == "pending"
    assert world.backend.downstream.applied["mark_delivered"] == 1
