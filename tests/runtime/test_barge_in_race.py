"""Barge-in cancels what the agent is saying, not what the caller just started (ADR-006).

The race: the caller talks over the agent, the endpointing controller fires `eager_end` on that
speech and a speculation starts, and the barge-in policy confirms the interruption at the same
virtual instant. Both paths run through the leg; whichever lands first, the agent's output must
stop and the caller's speculation must survive to be promoted when the turn ends.
"""

from __future__ import annotations

import asyncio

import pytest

from switchboard.runtime import Leg
from switchboard.turn import TurnSignal

from .harness import LONG_GREETING, DeliveryBrain, SimParty, World, every, first, make_world

QUESTION = "What is my pay this week?"


def inbound(world: World) -> Leg:
    (conv,) = world.runtime.conversations.values()
    return next(leg for leg in conv.legs.values() if leg.role == "inbound")


@pytest.mark.parametrize("eager_first", [True, False])
async def test_barge_in_keeps_the_speculation_started_by_the_barge_in_speech(
    eager_first: bool,
) -> None:
    world = make_world(DeliveryBrain(long_greeting=True))
    conv = await world.start()

    async def courier(c: SimParty) -> None:
        await c.say("Hi, this is courier C-77.")
        await c.hear("refunds")
        leg = inbound(world)
        utterance = leg.speaker._utterance
        assert utterance is not None
        eager = TurnSignal("eager_end", c.clock.now(), QUESTION, 0.6)
        barge_in = leg.speaker._interrupt(utterance)
        speculate = leg._on_turn_signal(eager)
        first_, second = (speculate, barge_in) if eager_first else (barge_in, speculate)
        await asyncio.gather(first_, second)
        await c.clock.sleep(0.3)
        await leg._on_turn_signal(TurnSignal("end", c.clock.now(), QUESTION, 0.9))
        await c.hear("182.40")
        await c.quiet()

    task = asyncio.ensure_future(courier(world.courier))
    await world.clock.run_until_idle()
    assert task.done()
    task.result()
    assert conv.failures == []
    await conv.end("resolved")
    events = await world.events(conv)

    # The agent's output stopped where the caller cut in.
    cut = first(events, "speech.played", interrupted=True)
    assert cut.text_played and LONG_GREETING.startswith(cut.text_played)
    assert "route today" not in world.courier.heard
    assert world.courier.cleared == 1

    # The caller's speculation was promoted, not cancelled and restarted.
    spec = first(events, "turn.speculation_started", partial_text=QUESTION)
    assert every(events, "turn.speculation_cancelled", turn_id=spec.turn_id) == []
    assert first(events, "turn.committed", turn_id=spec.turn_id).text == QUESTION
    asked = [r for r in world.llm.requests if r.messages[-1].content == QUESTION]
    assert len(asked) == 1
