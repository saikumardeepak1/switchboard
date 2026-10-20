"""The log is the system of record: replay alone rebuilds every grounding gate input (ADR-002)."""

from __future__ import annotations

from dataclasses import dataclass

from switchboard.brain import BrainRequest, Reply
from switchboard.core import AnyEvent, SpeechSentenceGated
from switchboard.memory import MemoryKind
from switchboard.runtime import replay

from .harness import TENANT, DeliveryBrain, SimParty, World, every, make_world, run_script

MEMORY_TEXT = "Drop-off code for the usual building is 4471."
REFUND_ANSWER = "You can refund up to 25 dollars."
WRONG_REFUND = "Anything above 40 dollars needs a human."
CODE_ANSWER = "Your usual drop-off code is 4471."


@dataclass
class FidelityBrain(DeliveryBrain):
    """Says numbers that only knowledge (25) or memory (4471) can ground, and one (40) that
    nothing grounds, so the log holds released and blocked sentences alike."""

    def parent(self, request: BrainRequest) -> Reply:
        last = request.messages[-1]
        if last.role == "user":
            text = last.content.lower()
            if "refund limit" in text:
                return Reply(f"{REFUND_ANSWER} {WRONG_REFUND}")
            if "code" in text:
                return Reply(CODE_ANSWER)
        return super().parent(request)


async def courier(c: SimParty) -> None:
    await c.say("Hi, this is courier C-77.")
    await c.hear("verified")
    await c.quiet()
    await c.say("What is the refund limit?")
    await c.hear("25 dollars")
    await c.quiet(1.5)
    await c.say("What is my drop-off code?")
    await c.hear("4471")
    await c.quiet()


async def run_call(world: World) -> list[AnyEvent]:
    world.memory.write(
        TENANT, "C-77", MemoryKind.PREFERENCE, {"text": MEMORY_TEXT}, now=world.clock.now()
    )
    conv = await world.start()
    await run_script(world, courier(world.courier))
    assert conv.failures == []
    await conv.end("resolved")
    return await world.events(conv)


async def test_every_gate_decision_is_reproduced_from_replay_alone() -> None:
    world = make_world(FidelityBrain())
    events = await run_call(world)

    loaded = every(events, "memory.loaded")
    assert len(loaded) == 1 and loaded[0].texts == [MEMORY_TEXT]
    retrieved = every(events, "knowledge.retrieved")
    assert retrieved and all(len(e.texts) == len(e.fact_ids) for e in retrieved)
    assert any("25 dollars" in t for e in retrieved for t in e.texts)

    gated = [
        (i, e) for i, e in enumerate(events) if isinstance(e, SpeechSentenceGated) and e.tier <= 2
    ]
    decisions = {e.sentence: e.decision for _, e in gated}
    assert decisions[REFUND_ANSWER] == "release"
    assert decisions[WRONG_REFUND] == "block"
    assert decisions[CODE_ANSWER] == "release"

    # Fresh gate, no store, no memory, no knowledge index: only the log prefix before each
    # decision. Every sentence must come out with the same decision, tier and claims.
    for i, event in gated:
        state = replay(events[:i])
        again = world.gate.review(event.sentence, state.grounding_evidence())
        assert (again.decision, again.tier, again.claim_dicts()) == (
            event.decision,
            event.tier,
            event.claims,
        ), event.sentence


async def test_content_capture_off_keeps_text_out_of_the_log_but_not_out_of_the_call() -> None:
    world = make_world(FidelityBrain())
    world.runtime = world.new_runtime()
    world.runtime.capture_content = False
    events = await run_call(world)

    (loaded,) = every(events, "memory.loaded")
    assert loaded.texts == [] and loaded.count == 1
    assert all(e.texts == [] for e in every(events, "knowledge.retrieved"))
    # The live gate still saw the memory and the knowledge.
    released = {e.sentence for e in every(events, "speech.sentence_gated", decision="release")}
    assert {REFUND_ANSWER, CODE_ANSWER} <= released


async def test_recovery_uses_the_memory_in_the_log_not_a_second_read() -> None:
    world = make_world(FidelityBrain())
    world.memory.write(
        TENANT, "C-77", MemoryKind.PREFERENCE, {"text": MEMORY_TEXT}, now=world.clock.now()
    )
    conv = await world.start()
    await conv.abandon()
    world.memory.forget(TENANT, "C-77")
    world.courier.fresh_line(seed=5)
    recovered = await world.new_runtime().recover(world.bundle, conv.id, io=world.courier.io)
    assert recovered.memory_texts == [MEMORY_TEXT]
    assert len(every(await world.events(conv), "memory.loaded")) == 1
    await recovered.end("resolved")
