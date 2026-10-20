"""Bridge speech covers a slow tool with a fixed, claim-free phrase (ADR-008)."""

from __future__ import annotations

import dataclasses
from typing import Any

from .harness import SimParty, World, every, make_world, run_script
from .test_scenarios import report_stuck_order, verify

BRIDGE = "Bear with me while I pull that up."


def configure(world: World, phrases: tuple[str, ...], lookup_s: float) -> None:
    config = dataclasses.replace(world.bundle.config, bridge_phrases=phrases)
    world.bundle = dataclasses.replace(world.bundle, config=config)
    registry = world.bundle.registry
    orig = registry.get("lookup_order")
    assert orig is not None

    async def slow(order_id: str) -> dict[str, Any]:
        await world.clock.sleep(lookup_s)
        return dict(await orig.fn(order_id=order_id))

    registry._tools["lookup_order"] = dataclasses.replace(orig, fn=slow)


async def courier(c: SimParty) -> None:
    await verify(c)
    await report_stuck_order(c)
    await c.hear("calling the customer now")
    await c.quiet()


async def run(phrases: tuple[str, ...], lookup_s: float) -> tuple[World, list[Any]]:
    world = make_world()
    configure(world, phrases, lookup_s)
    conv = await world.start()
    await run_script(world, courier(world.courier))
    assert conv.failures == []
    await conv.end("resolved")
    return world, list(await world.events(conv))


async def test_a_slow_tool_gets_a_bridge_phrase_before_its_result() -> None:
    world, events = await run((BRIDGE,), lookup_s=4.0)
    assert BRIDGE in world.courier.tts.texts
    (gated,) = [e for e in every(events, "speech.sentence_gated") if e.sentence == BRIDGE]
    assert gated.decision == "release" and gated.claims == []
    (lookup,) = every(events, "tool.executed", tool="lookup_order")
    assert events.index(gated) < events.index(lookup)
    assert gated.at - every(events, "tool.proposed", tool="lookup_order")[0].at >= 0.7


async def test_a_fast_tool_gets_no_bridge() -> None:
    world, _ = await run((BRIDGE,), lookup_s=0.1)
    assert BRIDGE not in world.courier.tts.texts


async def test_a_bridge_phrase_with_a_claim_is_dropped() -> None:
    world, events = await run(("Give me 5 seconds.",), lookup_s=4.0)
    assert all("5 seconds" not in t for t in world.courier.tts.texts)
    assert all("5 seconds" not in getattr(e, "sentence", "") for e in events)
