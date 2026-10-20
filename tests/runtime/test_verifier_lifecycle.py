"""Tier 3 verdicts never wedge shutdown or barge-in, and a failing verifier fails closed."""

from __future__ import annotations

import asyncio

from switchboard.grounding import Evidence, GroundingGate, VerifierVerdict

from .harness import GREETING, SimParty, World, every, make_world


class MiddlingScorer:
    """Sends every claimless assertion to tier 3 (filler releases before a scorer sees it)."""

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


class BrokenVerifier:
    async def verify(self, sentence: str, evidence: Evidence) -> VerifierVerdict:
        raise RuntimeError("verifier provider 503")


async def test_a_failing_verifier_blocks_and_the_speaker_keeps_going() -> None:
    world = make_world()
    world.gate = GroundingGate(MiddlingScorer(), BrokenVerifier())
    world.runtime = world.new_runtime()
    conv = await world.start()
    leg = next(iter(conv.legs.values()))

    async def courier(c: SimParty) -> None:
        await c.say("Hi, this is courier C-77.")
        await c.clock.sleep(5.0)
        await c.say("What is my pay this week?")
        await c.hear("182.40")
        await c.quiet()

    task = asyncio.ensure_future(courier(world.courier))
    await world.clock.run_until_idle()
    assert task.done(), "the speaker died and the pay answer never played"
    task.result()
    assert leg.speaker._task is not None and not leg.speaker._task.done()
    assert any(isinstance(f, RuntimeError) for f in conv.failures)
    events = await world.events(conv)
    tier3 = [(e.sentence, e.decision) for e in every(events, "speech.sentence_gated", tier=3)]
    assert tier3 and all(decision == "block" for _, decision in tier3)
    # Escalated sentences never reached TTS; filler and the tier 1 grounded one did.
    assert GREETING in [sentence for sentence, _ in tier3]
    assert world.courier.tts.texts == [
        "One moment.",
        "Let me check.",
        "You've earned $182.40 this week across 23 deliveries.",
    ]
    await conv.end("resolved")
