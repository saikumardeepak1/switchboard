"""End-to-end calls on the virtual clock: the delivery support agent with simulated parties."""

from __future__ import annotations

from typing import Any

import pytest

from switchboard.core import AnyEvent, InMemoryEventStore, NewEvent
from switchboard.effects import InMemoryLedger
from switchboard.memory import MemoryKind
from switchboard.runtime import SPECULATION_RULE, AdmissionRejected, replay

from .harness import (
    GREETING,
    LONG_GREETING,
    TENANT,
    DeliveryBrain,
    SimParty,
    World,
    customer_declines,
    every,
    first,
    is_subsequence,
    make_world,
    run_script,
)


async def verify(courier: SimParty) -> None:
    await courier.say("Hi, this is courier C-77.")
    await courier.hear("verified")
    await courier.quiet()


async def report_stuck_order(courier: SimParty) -> None:
    await courier.say("I can't mark order 48213 delivered, the customer wants it at home.")


async def happy_courier(courier: SimParty) -> None:
    await verify(courier)
    await report_stuck_order(courier)
    await courier.hear("calling the customer now")
    await courier.quiet()
    await courier.say("While I wait, what is my pay this week?")
    await courier.hear("182.40")
    await courier.hear("marked delivered")
    await courier.quiet()


def seq_of(event: AnyEvent) -> int:
    return event.seq


async def finished_call(world: World, script: Any) -> list[AnyEvent]:
    conv = await world.start()
    await run_script(world, script(world.courier))
    assert conv.failures == []
    await conv.end("resolved")
    return await world.events(conv)


# 1


async def test_happy_path_calls_the_customer_on_a_child_leg_while_the_courier_waits() -> None:
    world = make_world()
    events = await finished_call(world, happy_courier)

    inbound = first(events, "leg.opened", role="inbound")
    child = first(events, "leg.opened", role="outbound_child")
    assert child.parent_leg_id == inbound.leg_id
    assert child.party == "customer"
    assert world.dialer.dialed == [("customer", child.leg_id)]

    result = first(events, "child.result")
    assert result.leg_id == inbound.leg_id
    assert result.result["confirmed"] is True
    assert result.result["evidence"] == {"customer_confirmed_address": True}

    # The courier asked about pay and got the answer while the customer call was still running.
    pay_answer = next(
        e
        for e in every(events, "speech.sentence_gated", leg_id=inbound.leg_id, decision="release")
        if "182.40" in e.sentence
    )
    assert seq_of(child) < seq_of(pay_answer) < seq_of(result)
    heard_at = next(c.at for c in world.courier.chunks if "182.40" in c.text)
    assert heard_at < result.at

    # mark_delivered ran exactly once, and only after the confirmation arrived.
    marks = every(events, "tool.proposed", tool="mark_delivered")
    assert len(marks) == 1
    assert seq_of(marks[0]) > seq_of(result)
    executed = every(events, "tool.executed", tool="mark_delivered")
    assert len(executed) == 1 and not executed[0].replayed
    assert world.backend.downstream.applied["mark_delivered"] == 1
    assert world.backend.orders["48213"].status == "delivered"

    # Reads may run on the speculative turn, so verification lands before its commit.
    flow = [
        "conversation.started",
        "leg.opened",
        "tool.proposed:verify_courier",
        "policy.verdict",
        "tool.executed:verify_courier",
        "procedure.advanced",
        "turn.committed",
        "tool.executed:lookup_order",
        "procedure.advanced",
        "tool.proposed:call_party",
        "policy.verdict",
        "leg.opened",
        "tool.executed:call_party",
        "tool.executed:get_courier_pay",
        "child.result",
        "brain.routed",
        "tool.proposed:mark_delivered",
        "policy.verdict",
        "tool.executed:mark_delivered",
        "procedure.advanced",
        "speech.played",
        "leg.closed",
        "conversation.ended",
    ]
    labels = [e.type + (f":{e.tool}" if hasattr(e, "tool") else "") for e in events]
    assert is_subsequence(flow, labels)
    assert first(events, "leg.closed", leg_id=child.leg_id).reason == "completed"
    assert {e.bundle_id for e in events} == {world.bundle.bundle_id}
    assert world.courier.tts.texts[0] == "One moment."
    assert any("marked delivered" in t for t in world.courier.tts.texts)
    assert any("home address" in t for t in world.customer.tts.texts)


# 2


async def test_policy_blocks_marking_delivered_before_confirmation_and_the_model_repairs() -> None:
    world = make_world(DeliveryBrain(mark_early=True))

    async def courier(c: SimParty) -> None:
        await verify(c)
        await report_stuck_order(c)
        await c.hear("marked delivered")
        await c.quiet()

    events = await finished_call(world, courier)

    proposals = every(events, "tool.proposed", tool="mark_delivered")
    assert len(proposals) == 2
    denied = first(events, "policy.verdict", call_id=proposals[0].call_id)
    assert not denied.allowed
    assert denied.rule_ids == ["delivered_needs_confirmation"]
    assert denied.repair_hint is not None and "call_party" in denied.repair_hint

    # The repaired path calls the customer, and the tool only runs after the confirmation.
    call_party = first(events, "tool.proposed", tool="call_party")
    result = first(events, "child.result")
    assert seq_of(denied) < seq_of(call_party) < seq_of(result)
    executed = every(events, "tool.executed", tool="mark_delivered")
    assert [e.call_id for e in executed] == [proposals[1].call_id]
    assert seq_of(executed[0]) > seq_of(result)
    assert world.backend.downstream.applied["mark_delivered"] == 1

    # The model saw the denial as a tool result carrying the hint.
    repair_request = next(
        r for r in world.llm.requests if r.messages[-1].content.startswith("Denied by policy")
    )
    assert repair_request.messages[-1].tool_call_id == proposals[0].call_id


# 3


async def test_grounding_blocks_a_wrong_pay_amount_and_regenerates_the_right_one() -> None:
    world = make_world(DeliveryBrain(wrong_pay=True))

    async def courier(c: SimParty) -> None:
        await verify(c)
        await c.say("What is my pay this week?")
        await c.hear("182.40")
        await c.quiet()

    events = await finished_call(world, courier)

    gated = every(events, "speech.sentence_gated")
    blocked = [e for e in gated if e.decision == "block"]
    assert [e.sentence for e in blocked] == [
        "You've earned $128.40 this week across 23 deliveries."
    ]
    assert blocked[0].tier == 1
    released = first(
        events, "speech.sentence_gated", decision="release", utterance_id=blocked[0].utterance_id
    )
    assert "$182.40" in released.sentence
    assert seq_of(blocked[0]) < seq_of(released)

    tts = " ".join(world.courier.tts.texts)
    assert "128.40" not in tts and "128.40" not in world.courier.heard
    assert "182.40" in tts
    hint = world.llm.requests[-1].messages[-1]
    assert hint.role == "system" and "$182.40" in hint.content


# 4


async def test_barge_in_truncates_history_to_what_was_played() -> None:
    world = make_world(DeliveryBrain(long_greeting=True))

    async def courier(c: SimParty) -> None:
        await c.say("Hi, this is courier C-77.")
        await c.hear("refunds")
        await c.say("Sorry, what is my pay this week?")
        await c.hear("182.40")
        await c.quiet()

    events = await finished_call(world, courier)

    played = every(events, "speech.played", interrupted=True)
    assert len(played) == 1
    cut = played[0].text_played
    assert cut and LONG_GREETING.startswith(cut) and len(cut) < len(LONG_GREETING)
    assert world.courier.cleared == 1

    # The next model call saw only the words the courier actually heard.
    pay_request = next(
        r for r in world.llm.requests if r.messages[-1].content.startswith("Sorry, what is")
    )
    assistant = [
        m.content for m in pay_request.messages if m.role == "assistant" and not m.tool_call_id
    ]
    assert cut in assistant
    assert LONG_GREETING not in assistant
    assert cut in [m.content for m in replay(events).messages(played[0].leg_id or "")]


# 5


async def test_speculation_cancels_on_resume_and_defers_writes_until_commit() -> None:
    world = make_world(ttft_s=0.05)

    async def courier(c: SimParty) -> None:
        await verify(c)
        await c.say("Order 48213 was late.")
        await c.clock.sleep(0.3)
        await c.say("Please refund the customer ten dollars.")
        await c.hear("refunded")
        await c.quiet()

    events = await finished_call(world, courier)

    started = every(events, "turn.speculation_started")
    cancelled = every(events, "turn.speculation_cancelled")
    assert cancelled, "the resumed caller should cancel the first speculation"
    first_spec = started[0] if started[0].partial_text.startswith("Order 48213") else started[1]
    assert first_spec.turn_id in {e.turn_id for e in cancelled}
    # A read tool ran speculatively on the abandoned turn; that is allowed (ADR-006).
    looked_up = first(events, "tool.executed", tool="lookup_order")
    assert seq_of(looked_up) < seq_of(
        first(events, "turn.speculation_cancelled", turn_id=first_spec.turn_id)
    )

    refund = first(events, "tool.proposed", tool="issue_refund")
    verdicts = every(events, "policy.verdict", call_id=refund.call_id)
    assert [(v.allowed, v.rule_ids) for v in verdicts] == [
        (False, [SPECULATION_RULE]),
        (True, []),
    ]
    refund_turn = next(t for t in started if "refund" in t.partial_text)
    commit = first(events, "turn.committed", turn_id=refund_turn.turn_id)
    assert seq_of(verdicts[0]) < seq_of(commit) < seq_of(verdicts[1])
    executed = every(events, "tool.executed", tool="issue_refund")
    assert len(executed) == 1 and seq_of(executed[0]) > seq_of(commit)
    assert world.backend.downstream.applied["issue_refund"] == 1

    # The cancelled turn's words and tool calls are not in the model's history.
    state = replay(events)
    leg = first(events, "leg.opened", role="inbound").leg_id or ""
    history = [m.content for m in state.messages(leg)]
    assert "Order 48213 was late." not in history


# 6


class CrashingStore(InMemoryEventStore):
    """Dies right after the downstream write, before `tool.executed` reaches the log."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = True

    async def append(self, conversation_id: str, events: Any, expected_seq: int) -> list[AnyEvent]:
        batch: list[NewEvent] = list(events)
        if self.armed and any(
            e.type == "tool.executed" and getattr(e, "tool", "") == "mark_delivered" for e in batch
        ):
            self.armed = False
            raise ConnectionResetError("process killed")
        return await super().append(conversation_id, batch, expected_seq)


@pytest.mark.parametrize("ledger_survives", [True, False])
async def test_recovery_from_the_log_never_repeats_the_delivery(ledger_survives: bool) -> None:
    store = CrashingStore()
    world = make_world(store=store)
    conv = await world.start()

    async def until_crash(c: SimParty) -> None:
        await verify(c)
        await report_stuck_order(c)
        await c.hear("calling the customer now")

    await run_script(world, until_crash(world.courier))
    assert world.backend.downstream.applied["mark_delivered"] == 1
    before = await store.read(conv.id)
    assert not every(before, "tool.executed", tool="mark_delivered")
    assert first(
        before,
        "policy.verdict",
        call_id=first(before, "tool.proposed", tool="mark_delivered").call_id,
    ).allowed

    await conv.abandon()
    world.courier.fresh_line(seed=7)
    ledger = world.ledger if ledger_survives else InMemoryLedger(world.clock)
    runtime = world.new_runtime(ledger=ledger)
    recovered = await runtime.recover(world.bundle, conv.id, io=world.courier.io)

    async def after(c: SimParty) -> None:
        await c.say("Hello, are you still there?")
        await c.hear("Got it.")
        await c.quiet()

    await run_script(world, after(world.courier))
    await recovered.end("resolved")
    events = await store.read(conv.id)

    executed = every(events, "tool.executed", tool="mark_delivered")
    assert len(executed) == 1
    assert executed[0].replayed is ledger_survives
    assert world.backend.downstream.applied["mark_delivered"] == 1
    requests = [r for r in world.backend.downstream.requests if r[0] == "mark_delivered"]
    assert len(requests) == (1 if ledger_survives else 2)
    assert len({key for _, key in requests}) == 1
    assert recovered.state == replay(events)


# 7


async def test_admission_rejects_past_the_tenant_cap_and_frees_the_slot_on_end() -> None:
    world = make_world(tenant_caps={TENANT: 1})
    first_conv = await world.start()
    with pytest.raises(AdmissionRejected, match="cap of 1"):
        await world.start()
    assert world.admission.active(TENANT) == 1
    await first_conv.end("resolved")
    second = await world.start()
    assert world.admission.active(TENANT) == 1
    await second.end("resolved")
    assert world.admission.active(TENANT) == 0


# 8


async def test_customer_declines_so_delivery_stays_blocked_and_the_agent_hands_off() -> None:
    world = make_world(DeliveryBrain(stubborn=True), customer=customer_declines)

    async def courier(c: SimParty) -> None:
        await verify(c)
        await report_stuck_order(c)
        await c.hear("teammate")
        await c.quiet()

    events = await finished_call(world, courier)

    assert first(events, "child.result").result["confirmed"] is False
    assert not every(events, "tool.executed", tool="mark_delivered")
    marks = every(events, "tool.proposed", tool="mark_delivered")
    verdicts = [first(events, "policy.verdict", call_id=m.call_id) for m in marks]
    assert verdicts and all(not v.allowed for v in verdicts)
    assert all(v.rule_ids == ["delivered_needs_confirmation"] for v in verdicts)
    assert len(every(events, "handoff.requested")) == 1
    assert world.backend.orders["48213"].status == "out_for_delivery"
    assert world.backend.downstream.applied["mark_delivered"] == 0
    assert world.bundle.config.handoff_message in world.courier.tts.texts


# 9


async def test_live_state_equals_replay_of_the_log() -> None:
    world = make_world()
    conv = await world.start()
    await run_script(world, happy_courier(world.courier))
    await conv.end("resolved")
    events = await world.events(conv)
    assert conv.state == replay(events)
    assert conv.state.completed_steps == ["verify_identity", "investigate", "resolve"]
    assert conv.state.evidence["customer_confirmed_address"] is True
    assert conv.state.evidence["chat_shows_address_change"] is False


async def test_same_seed_gives_a_byte_identical_log() -> None:
    logs = []
    for _ in range(2):
        world = make_world()
        events = await finished_call(world, happy_courier)
        logs.append([e.model_dump_json() for e in events])
    assert logs[0] == logs[1]


async def test_memory_is_read_at_start_and_written_at_end() -> None:
    world = make_world()
    world.memory.write(
        TENANT,
        "C-77",
        MemoryKind.PREFERENCE,
        {"text": "Prefers short answers."},
        now=world.clock.now(),
    )
    events = await finished_call(world, happy_courier)
    assert any("Prefers short answers." in m.content for m in world.llm.requests[0].messages)
    summaries = [
        m
        for m in world.memory.read(TENANT, "C-77", world.clock.now())
        if m.kind.value == "interaction_summary"
    ]
    assert summaries and summaries[0].content["conversation_id"] == events[0].conversation_id


async def test_knowledge_is_retrieved_as_of_the_call_and_logged() -> None:
    world = make_world()

    async def courier(c: SimParty) -> None:
        await verify(c)
        await c.say("Is the geofence broken today?")
        await c.hear("Got it.")
        await c.quiet()

    events = await finished_call(world, courier)
    retrieved = every(events, "knowledge.retrieved")
    ids = {i for e in retrieved for i in e.fact_ids}
    assert "statuspage:geo4fence9x1:geofence" in ids
    assert all(e.as_of >= world.clock.now() - 60 for e in retrieved[-1:])
    request = world.llm.requests[-1]
    assert any("Geofence service degraded" in m.content for m in request.messages)


async def test_greeting_is_spoken_in_full_without_interruption() -> None:
    world = make_world()
    events = await finished_call(world, verify)
    played = every(events, "speech.played")
    assert [p.text_played for p in played] == ["One moment.", GREETING]
    assert not any(p.interrupted for p in played)
