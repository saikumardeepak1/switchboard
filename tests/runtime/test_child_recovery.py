"""A child call that was ringing when the process died is redialed once on recovery (ADR-003)."""

from __future__ import annotations

import asyncio

from switchboard.runtime import Conversation, replay

from .harness import SimParty, World, every, first, make_world, run_script


async def verify_and_report(c: SimParty) -> None:
    await c.say("Hi, this is courier C-77.")
    await c.hear("verified")
    await c.quiet()
    await c.say("I can't mark order 48213 delivered, the customer wants it at home.")
    await c.hear("calling the customer now")


async def crash_while_ringing(world: World) -> Conversation:
    world.dialer.ring_s = 20.0
    conv = await world.start()
    script = asyncio.ensure_future(verify_and_report(world.courier))
    while not script.done():
        await world.clock.advance(0.01)
    script.result()
    # The dialer is still ringing when the process dies.
    assert len(world.dialer.dialed) == 1
    assert world.dialer.tasks == []
    await conv.abandon()
    world.courier.fresh_line(seed=7)
    return conv


async def test_an_open_child_call_is_redialed_once_with_the_same_goal() -> None:
    world = make_world()
    conv = await crash_while_ringing(world)
    before = await world.store.read(conv.id)
    child = first(before, "leg.opened", role="outbound_child")

    recovered = await world.new_runtime().recover(world.bundle, conv.id, io=world.courier.io)

    async def courier(c: SimParty) -> None:
        await c.hear("marked delivered")
        await c.quiet()

    await run_script(world, courier(world.courier))
    assert recovered.failures == []
    await recovered.end("resolved")
    events = await world.store.read(conv.id)

    # Same child leg id, dialed again exactly once, with the same goal and party.
    assert world.dialer.dialed == [("customer", child.leg_id), ("customer", child.leg_id)]
    reopened = every(events, "leg.opened", leg_id=child.leg_id)
    assert len(reopened) == 2
    assert {(e.party, e.goal, e.parent_leg_id) for e in reopened} == {
        (child.party, child.goal, child.parent_leg_id)
    }
    assert every(events, "leg.closed", reason="lost_in_recovery") == []
    result = first(events, "child.result", child_leg_id=child.leg_id)
    assert result.result["confirmed"] is True
    assert len(every(events, "tool.executed", tool="mark_delivered")) == 1
    assert world.backend.downstream.applied["mark_delivered"] == 1
    assert any("home address" in t for t in world.customer.tts.texts)
    assert recovered.state == replay(events)


async def test_a_child_already_redialed_is_not_dialed_a_third_time() -> None:
    world = make_world()
    conv = await crash_while_ringing(world)
    again = await world.new_runtime().recover(world.bundle, conv.id, io=world.courier.io)
    await world.clock.advance(1.0)
    await again.abandon()  # dies again while the redial rings
    world.courier.fresh_line(seed=8)

    last = await world.new_runtime().recover(world.bundle, conv.id, io=world.courier.io)
    events = await world.store.read(conv.id)
    child = first(events, "leg.opened", role="outbound_child")
    assert world.dialer.dialed == [("customer", child.leg_id)] * 2
    assert first(events, "leg.closed", leg_id=child.leg_id).reason == "lost_in_recovery"
    await last.end("resolved")
