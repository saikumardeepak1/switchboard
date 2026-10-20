"""A backchannel while the agent speaks is recorded, not taken as a turn (ADR-006, ADR-007)."""

from __future__ import annotations

from switchboard.runtime import replay

from .harness import LONG_GREETING, DeliveryBrain, SimParty, every, make_world, run_script


async def test_mm_hm_during_agent_speech_is_recorded_and_playback_continues() -> None:
    world = make_world(DeliveryBrain(long_greeting=True))

    async def courier(c: SimParty) -> None:
        await c.say("Hi, this is courier C-77.")
        await c.hear("refunds")
        await c.say("Mm-hm.")
        await c.hear("route today")
        await c.quiet()

    conv = await world.start()
    await run_script(world, courier(world.courier))
    assert conv.failures == []
    await conv.end("resolved")
    events = await world.events(conv)

    (backchannel,) = every(events, "turn.backchannel")
    assert backchannel.text == "Mm-hm."
    assert backchannel.utterance_id is not None
    # Not a turn: never committed, never speculated on, never sent to the model.
    assert all("mm-hm" not in e.text.lower() for e in every(events, "turn.committed"))
    assert all(
        "mm-hm" not in e.partial_text.lower() for e in every(events, "turn.speculation_started")
    )
    assert not any("Mm-hm" in m.content for r in world.llm.requests for m in r.messages)
    # Playback carried on to the end of the greeting.
    played = every(events, "speech.played", utterance_id=backchannel.utterance_id)
    assert [(p.text_played, p.interrupted) for p in played] == [(LONG_GREETING, False)]
    assert world.courier.cleared == 0
    assert conv.state == replay(events)


async def test_a_real_question_during_agent_speech_still_barges_in() -> None:
    world = make_world(DeliveryBrain(long_greeting=True))

    async def courier(c: SimParty) -> None:
        await c.say("Hi, this is courier C-77.")
        await c.hear("refunds")
        await c.say("Okay, what is my pay this week?")
        await c.hear("182.40")
        await c.quiet()

    conv = await world.start()
    await run_script(world, courier(world.courier))
    await conv.end("resolved")
    events = await world.events(conv)
    assert every(events, "turn.backchannel") == []
    assert every(events, "speech.played", interrupted=True)
