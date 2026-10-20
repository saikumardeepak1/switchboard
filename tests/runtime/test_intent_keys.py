"""A write re-proposed in a later turn of the same request replays; a new request runs again."""

from __future__ import annotations

from dataclasses import dataclass

from switchboard.brain import BrainRequest, Reply
from switchboard.core import ConversationStarted, LegOpened, ToolProposed, TurnCommitted
from switchboard.runtime import replay

from .harness import DeliveryBrain, SimParty, every, make_world, run_script
from .test_scenarios import happy_courier


@dataclass
class EagerMarker(DeliveryBrain):
    """Marks the order delivered again whenever the courier asks whether it is done."""

    def parent(self, request: BrainRequest) -> Reply:
        last = request.messages[-1]
        if last.role == "user" and "is it marked" in last.content.lower():
            return Reply("Let me make sure.", (self.mark(),))
        return super().parent(request)


async def test_the_same_write_proposed_in_a_later_turn_replays() -> None:
    world = make_world(EagerMarker())

    async def courier(c: SimParty) -> None:
        await happy_courier(c)
        await c.say("Thanks, is it marked now?")
        await c.hear("make sure")
        await c.hear("marked delivered")
        await c.quiet()

    conv = await world.start()
    await run_script(world, courier(world.courier))
    assert conv.failures == []
    await conv.end("resolved")
    events = await world.events(conv)

    executed = every(events, "tool.executed", tool="mark_delivered")
    assert [e.replayed for e in executed] == [False, True]
    assert executed[0].idempotency_key == executed[1].idempotency_key
    assert world.backend.downstream.applied["mark_delivered"] == 1
    # The first request opened the intent; the follow-up question did not.
    commits = every(events, "turn.committed")
    assert [c.requests for c in commits if c.requests] == [["mark_delivered"]]
    assert conv.state == replay(events)


def test_a_new_request_opens_a_new_intent_in_the_fold() -> None:
    base = {"conversation_id": "c", "at": 0.0, "bundle_id": "b", "leg_id": "l"}
    log = [
        ConversationStarted(**base, tenant_id="t", customer_id="u"),
        LegOpened(**base, role="inbound", party="caller"),
        TurnCommitted(
            **base, turn_id="t1", text="Refund it.", endpointing_ms=1, requests=["issue_refund"]
        ),
        ToolProposed(**base, call_id="a", tool="issue_refund", args={}),
        TurnCommitted(**base, turn_id="t2", text="Thanks, did it work?", endpointing_ms=1),
        ToolProposed(**base, call_id="b", tool="issue_refund", args={}),
        TurnCommitted(
            **base,
            turn_id="t3",
            text="Refund another.",
            endpointing_ms=1,
            requests=["issue_refund"],
        ),
        ToolProposed(**base, call_id="c", tool="issue_refund", args={}),
    ]
    numbered = [e.model_copy(update={"seq": i, "event_id": f"e{i}"}) for i, e in enumerate(log, 1)]
    state = replay(numbered)
    assert [state.calls[c].intent for c in "abc"] == [1, 1, 2]
