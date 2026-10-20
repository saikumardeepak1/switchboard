"""Compensations belong to the runtime: the model never sees or runs them (ADR-009)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from switchboard.brain import BrainRequest, Reply
from switchboard.runtime import replay

from .harness import DeliveryBrain, SimParty, World, every, make_world, run_script
from .test_scenarios import happy_courier, verify


def with_rollback(world: World, *outcomes: str) -> None:
    config = dataclasses.replace(world.bundle.config, rollback_on=outcomes)
    world.bundle = dataclasses.replace(world.bundle, config=config)


@dataclass
class Undoer(DeliveryBrain):
    """Tries to run a compensation itself once the order is marked delivered."""

    def parent(self, request: BrainRequest) -> Reply:
        last = request.messages[-1]
        if last.role == "user" and "undo" in last.content.lower():
            return Reply("", (self.call("unmark_delivered", order_id="48213"),))
        if last.role == "tool" and last.content.startswith("Denied by policy (runtime:"):
            return Reply("I can't undo that myself.")
        return super().parent(request)


async def test_the_model_is_never_offered_a_compensation_and_cannot_run_one() -> None:
    world = make_world(Undoer())
    conv = await world.start()

    async def courier(c: SimParty) -> None:
        await happy_courier(c)
        await c.say("Actually, undo that.")
        await c.hear("undo that myself")
        await c.quiet()

    await run_script(world, courier(world.courier))
    assert conv.failures == []
    await conv.end("resolved")
    events = await world.events(conv)
    offered = {t.name for r in world.llm.requests for t in r.tools}
    assert offered and not offered & {"unmark_delivered", "reverse_refund"}
    (proposed,) = every(events, "tool.proposed", tool="unmark_delivered")
    verdict = every(events, "policy.verdict", call_id=proposed.call_id)[-1]
    assert not verdict.allowed and verdict.rule_ids == ["runtime:compensation_only"]
    assert every(events, "tool.executed", tool="unmark_delivered") == []
    assert world.backend.downstream.applied["unmark_delivered"] == 0


async def test_an_outcome_marked_for_rollback_compensates_every_write() -> None:
    world = make_world()
    with_rollback(world, "handoff_failed")
    conv = await world.start()
    await run_script(world, happy_courier(world.courier))
    await conv.end("handoff_failed")
    events = await world.events(conv)
    (mark,) = every(events, "tool.executed", tool="mark_delivered")
    (undo,) = every(events, "tool.compensated")
    assert undo.call_id == mark.call_id
    assert undo.compensation_tool == "unmark_delivered"
    assert undo.result["status"] == "out_for_delivery"
    assert world.backend.downstream.applied["unmark_delivered"] == 1
    assert world.backend.orders["48213"].status == "out_for_delivery"
    assert events.index(undo) < events.index(every(events, "conversation.ended")[0])
    assert replay(events).compensations == [mark.call_id]


async def test_other_outcomes_leave_writes_in_place() -> None:
    world = make_world()
    with_rollback(world, "handoff_failed")
    conv = await world.start()
    await run_script(world, happy_courier(world.courier))
    await conv.end("resolved")
    assert every(await world.events(conv), "tool.compensated") == []
    assert world.backend.downstream.applied["unmark_delivered"] == 0


@dataclass
class RefundThenStuck(DeliveryBrain):
    """Refunds once, then keeps proposing a second refund the policy denies."""

    def parent(self, request: BrainRequest) -> Reply:
        last = request.messages[-1]
        if last.role == "tool" and (
            last.content.startswith("Denied by policy") or '"refund_id"' in last.content
        ):
            return Reply("", (self.call("issue_refund", order_id="48213", amount=10),))
        return super().parent(request)


async def test_an_aborted_turn_compensates_the_writes_it_made() -> None:
    world = make_world(RefundThenStuck())
    with_rollback(world, "turn_aborted")
    conv = await world.start()

    async def courier(c: SimParty) -> None:
        await verify(c)
        await c.say("Please refund the customer for order 48213.")
        await c.hear("teammate")
        await c.quiet()

    await run_script(world, courier(world.courier))
    assert conv.failures == []
    events = await world.events(conv)
    assert every(events, "handoff.requested")
    (refund,) = every(events, "tool.executed", tool="issue_refund")
    (undo,) = every(events, "tool.compensated")
    assert undo.call_id == refund.call_id and undo.compensation_tool == "reverse_refund"
    assert world.backend.refunds["rf-1"]["reversed"] is True
    await conv.end("handoff")
    # Ending the call does not compensate again.
    assert len(every(await world.events(conv), "tool.compensated")) == 1
