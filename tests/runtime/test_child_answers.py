"""A child call can confirm only what the other party actually said (ADR-003, ADR-004)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from switchboard.brain import BrainRequest, Reply
from switchboard.core import ChildResult, ConversationStarted, LegOpened, TurnCommitted
from switchboard.policy import CheckContext, ProposedCall
from switchboard.runtime import replay

from .harness import DeliveryBrain, SimParty, every, first, make_world, run_script
from .test_scenarios import happy_courier, report_stuck_order, verify


@dataclass
class Hallucinating(DeliveryBrain):
    """The call hit silence, and the model reports a confirmation anyway."""

    def child(self, request: BrainRequest) -> Reply:
        return Reply("Thanks, goodbye.", (self.call("report_result", confirmed=True, notes="ok"),))


async def silent(party: SimParty) -> None:
    return None


async def test_a_child_cannot_confirm_without_hearing_the_party() -> None:
    world = make_world(Hallucinating(), customer=silent)
    conv = await world.start()

    async def courier(c: SimParty) -> None:
        await verify(c)
        await report_stuck_order(c)
        await c.hear("didn't confirm")
        await c.quiet()

    await run_script(world, courier(world.courier))
    assert conv.failures == []
    events = await world.events(conv)
    child = first(events, "leg.opened", role="outbound_child").leg_id
    assert every(events, "turn.committed", leg_id=child) == []
    refusals = [v for v in every(events, "policy.verdict", leg_id=child) if not v.allowed]
    assert refusals and all(v.rule_ids == ["runtime:no_party_answer"] for v in refusals)
    result = first(events, "child.result")
    assert result.result["confirmed"] is False
    assert result.result["evidence"] == {}
    assert every(events, "tool.executed", tool="mark_delivered") == []
    assert world.backend.downstream.applied["mark_delivered"] == 0
    assert first(events, "leg.closed", leg_id=child)
    await conv.end("resolved")


async def test_a_confirmation_cites_the_child_leg_and_the_answering_turn() -> None:
    world = make_world()
    conv = await world.start()
    await run_script(world, happy_courier(world.courier))
    await conv.end("resolved")
    events = await world.events(conv)
    child = first(events, "leg.opened", role="outbound_child").leg_id
    answer = first(events, "turn.committed", leg_id=child)
    result = first(events, "child.result").result
    assert result["provenance"] == {
        "customer_confirmed_address": {"child_leg_id": child, "turn_event_id": answer.event_id}
    }


BASE: dict[str, Any] = {"conversation_id": "c", "at": 0.0, "bundle_id": "b"}


def _state_with(provenance: dict[str, Any] | None, answered_on: str = "kid") -> Any:
    result: dict[str, Any] = {
        "party": "customer",
        "confirmed": True,
        "evidence": {"customer_confirmed_address": True},
    }
    if provenance is not None:
        result["provenance"] = {"customer_confirmed_address": provenance}
    log = [
        ConversationStarted(**BASE, leg_id=None, tenant_id="t", customer_id="u"),
        LegOpened(**BASE, leg_id="l", role="inbound", party="courier"),
        LegOpened(**BASE, leg_id="kid", role="outbound_child", party="customer", parent_leg_id="l"),
        TurnCommitted(**BASE, leg_id=answered_on, turn_id="t1", text="Yes.", endpointing_ms=1),
        ChildResult(**BASE, leg_id="l", child_leg_id="kid", result=result),
    ]
    numbered = [e.model_copy(update={"seq": i, "event_id": f"e{i}"}) for i, e in enumerate(log, 1)]
    return replay(numbered)


def _allowed(state: Any) -> bool:
    world = make_world()
    verdict = world.bundle.policy.check(
        ProposedCall("mark_delivered", {"order_id": "48213", "location": "home"}),
        CheckContext(evidence=state.evidence, completed_steps=frozenset({"verify_identity"})),
    )
    return verdict.allowed


def test_policy_is_satisfied_only_with_the_childs_answer_as_provenance() -> None:
    good = {"child_leg_id": "kid", "turn_event_id": "e4"}
    assert _allowed(_state_with(good))
    assert not _allowed(_state_with(None))
    assert not _allowed(_state_with({"child_leg_id": "other", "turn_event_id": "e4"}))
    assert not _allowed(_state_with({"child_leg_id": "kid", "turn_event_id": "e2"}))
    # A turn on the parent leg is not the party's answer.
    assert not _allowed(_state_with(good, answered_on="l"))
