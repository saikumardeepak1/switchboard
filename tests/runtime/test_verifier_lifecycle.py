"""Tier 3 verdicts never wedge shutdown or barge-in, and a failing verifier fails closed."""

from __future__ import annotations

import asyncio

from switchboard.grounding import Evidence, GroundingGate, VerifierVerdict

from .harness import SimParty, World, every, make_world


class MiddlingScorer:
    """Sends every claimless sentence to tier 3."""

    def score(self, sentence: str, evidence: Evidence) -> float:
        return 0.5


class SlowVerifier:
    def __init__(self, world: World, delay_s: float) -> None:
        self.world = world
        self.delay_s = delay_s

    async def verify(self, sentence: str, evidence: Evidence) -> VerifierVerdict:
        await self.world.clock.sleep(self.delay_s)
        return VerifierVerdict(True)


def slow_world(delay_s: float = 5.0) -> World:
    world = make_world()
    world.gate = GroundingGate(MiddlingScorer(), SlowVerifier(world, delay_s))
    world.runtime = world.new_runtime()
    return world


async def test_hang_up_while_a_verdict_is_pending_ends_the_conversation() -> None:
    world = slow_world()
    conv = await world.start()

    async def courier(c: SimParty) -> None:
        await c.say("Hi, this is courier C-77.")
        await c.clock.sleep(1.0)
        await conv.end("hangup")

    task = asyncio.ensure_future(courier(world.courier))
    await world.clock.run_until_idle()
    assert task.done(), "conv.end() hung on a pending tier 3 verdict"
    task.result()
    assert conv.state.ended
    events = await world.events(conv)
    assert events[-1].type == "conversation.ended"


async def test_barge_in_while_a_verdict_is_pending_returns() -> None:
    world = slow_world(delay_s=30.0)
    conv = await world.start()
    leg = next(iter(conv.legs.values()))
    interrupted: list[asyncio.Task[None]] = []

    async def courier(c: SimParty) -> None:
        await c.say("Hi, this is courier C-77.")
        await c.hear("One moment")
        await c.quiet()
        # That sentence played unverified; the speaker now waits on its verdict to close it.
        utterance = leg.speaker._utterance
        assert utterance is not None and utterance.unverified
        interrupted.append(asyncio.ensure_future(leg.speaker._interrupt(utterance)))
        await c.clock.sleep(1.0)
        await conv.end("hangup")

    task = asyncio.ensure_future(courier(world.courier))
    await world.clock.run_until_idle()
    assert task.done()
    task.result()
    assert interrupted and interrupted[0].done(), "barge-in hung on a pending tier 3 verdict"
    interrupted[0].result()
    assert every(await world.events(conv), "speech.played", interrupted=True)


async def test_closing_a_leg_cancels_its_pending_verdicts() -> None:
    world = slow_world(delay_s=30.0)
    conv = await world.start()
    leg = next(iter(conv.legs.values()))

    async def courier(c: SimParty) -> None:
        await c.say("Hi, this is courier C-77.")
        await c.hear("One moment")
        await c.clock.sleep(0.5)
        await conv.end("hangup")

    task = asyncio.ensure_future(courier(world.courier))
    await world.clock.run_until_idle()
    assert task.done()
    task.result()
    assert all(t.done() for t in leg._verdicts)
