"""Tier 3 escalations race playback: block in time, or self-correct after the fact (ADR-005)."""

from __future__ import annotations

from switchboard.grounding import Evidence, GroundingGate, VerifierVerdict
from switchboard.runtime import replay
from switchboard.runtime.speaker import CORRECTION

from .harness import GREETING, SimParty, make_world, run_script


class MiddlingScorer:
    """Sends every claimless sentence to tier 3."""

    def score(self, sentence: str, evidence: Evidence) -> float:
        return 0.5


class Verifier:
    """Rejects any sentence that says "verified", after `delay_s` on the virtual clock.

    Other sentences are approved after a short fixed delay.
    """

    def __init__(self, clock: object, delay_s: float) -> None:
        self.clock = clock
        self.delay_s = delay_s

    async def verify(self, sentence: str, evidence: Evidence) -> VerifierVerdict:
        flagged = "verified" in sentence
        await self.clock.sleep(self.delay_s if flagged else 0.05)  # type: ignore[attr-defined]
        return VerifierVerdict(not flagged, "test verifier")


async def greet(c: SimParty) -> None:
    await c.say("Hi, this is courier C-77.")
    await c.hear("One moment")
    await c.clock.sleep(8.0)
    await c.quiet()


async def run_with_verifier(delay_s: float) -> tuple[list[object], SimParty, str]:
    world = make_world()
    world.gate = GroundingGate(MiddlingScorer(), Verifier(world.clock, delay_s))
    world.runtime = world.new_runtime()
    conv = await world.start()
    await run_script(world, greet(world.courier))
    assert conv.failures == []
    await conv.end("resolved")
    events = await world.events(conv)
    leg = next(e.leg_id for e in events if e.type == "leg.opened")
    return list(events), world.courier, leg or ""


async def test_a_fast_verifier_blocks_before_the_sentence_is_spoken() -> None:
    events, courier, _ = await run_with_verifier(0.05)
    decisions = [
        (e.sentence, e.decision, e.tier) for e in events if e.type == "speech.sentence_gated"
    ]
    assert (GREETING, "escalate", 2) in decisions
    assert (GREETING, "block", 3) in decisions
    assert courier.tts.texts == ["One moment."]


async def test_a_slow_verifier_lets_speech_play_then_corrects_it() -> None:
    events, courier, leg = await run_with_verifier(4.0)
    assert courier.tts.texts == ["One moment.", GREETING, CORRECTION]
    played = [e.text_played for e in events if e.type == "speech.played"]
    assert played[-1] == f"{GREETING} {CORRECTION}"
    assert [m.content for m in replay(events).messages(leg)][-1] == played[-1]
