"""The adversarial model wrapper: injected faults are deterministic per seed and fully logged."""

from __future__ import annotations

import json
import random

import pytest

from switchboard.brain import BrainRequest, Message, Reply, TextDelta, ToolCallChunk, ToolSchema
from switchboard.core import VirtualClock
from switchboard.sim import AdversarialLLM, AdversaryConfig, corrupt_numbers

TOOLS = tuple(
    ToolSchema(name, name, {"type": "object"})
    for name in ("get_courier_pay", "issue_refund", "mark_delivered")
)
PAY = "You've earned $182.40 this week across 23 deliveries."


def base(request: BrainRequest) -> Reply:
    return Reply(PAY)


def request(*messages: Message, tools: tuple[ToolSchema, ...] = TOOLS) -> BrainRequest:
    return BrainRequest(messages=(Message("system", "prefix"), *messages), tools=tools)


def user(text: str = "what is my pay") -> Message:
    return Message("user", text)


def refund_done() -> tuple[Message, Message]:
    args = {"order_id": "48213", "amount": 10}
    return (
        Message("assistant", json.dumps(args), tool_call_id="call_9", name="issue_refund"),
        Message("tool", json.dumps({"refund_id": "rf-1", **args}), tool_call_id="call_9"),
    )


def run(adv: AdversarialLLM, n: int, *messages: Message) -> list[Reply]:
    return [adv.reply(request(*messages)) for _ in range(n)]


def test_no_probability_means_a_pure_pass_through() -> None:
    adv = AdversarialLLM(base, AdversaryConfig(), seed=1)
    assert run(adv, 20, user()) == [Reply(PAY)] * 20
    assert adv.faults == []


def test_same_seed_same_faults_and_a_different_seed_differs() -> None:
    config = AdversaryConfig(p_violation=0.3, p_corrupt=0.3, p_repeat=0.3)

    def faults(seed: int) -> list[object]:
        adv = AdversarialLLM(base, config, seed=seed)
        run(adv, 60, user(), *refund_done())
        return list(adv.faults)

    assert faults(5) == faults(5)
    assert faults(5) != faults(6)
    kinds = {f.kind for f in faults(5)}  # type: ignore[attr-defined]
    assert kinds & {"refund_over_limit", "second_refund", "early_mark_delivered"}
    assert "corrupt_money" in kinds


def test_violations_are_proposed_as_tool_calls_and_logged_with_their_ids() -> None:
    adv = AdversarialLLM(base, AdversaryConfig(p_violation=1.0), seed=2)
    replies = run(adv, 30, user())
    for reply, fault in zip(replies, adv.faults, strict=True):
        (call,) = reply.tool_calls
        assert fault.call_id == call.call_id and fault.tool == call.name
        assert dict(fault.args) == dict(call.args)
        if fault.kind == "refund_over_limit":
            assert call.name == "issue_refund" and call.args["amount"] > 25
        else:
            assert fault.kind == "early_mark_delivered" and call.name == "mark_delivered"
    assert len({f.call_id for f in adv.faults}) == 30


def test_no_early_delivery_once_the_customer_confirmed_and_second_refund_needs_a_first() -> None:
    confirmed = Message("system", '# Evidence\n{"customer_confirmed_address":true}')
    adv = AdversarialLLM(base, AdversaryConfig(p_violation=1.0), seed=3)
    run(adv, 30, confirmed, user())
    assert {f.kind for f in adv.faults} == {"refund_over_limit"}

    adv = AdversarialLLM(base, AdversaryConfig(p_violation=1.0), seed=3)
    run(adv, 40, confirmed, user(), *refund_done())
    assert {f.kind for f in adv.faults} == {"refund_over_limit", "second_refund"}
    second = next(f for f in adv.faults if f.kind == "second_refund")
    assert 0 < second.args["amount"] <= 25


def test_repeat_only_after_a_write_ran_and_it_repeats_the_same_arguments() -> None:
    adv = AdversarialLLM(base, AdversaryConfig(p_repeat=1.0), seed=4)
    assert run(adv, 5, user()) == [Reply(PAY)] * 5
    (reply,) = run(adv, 1, user(), *refund_done())
    (call,) = reply.tool_calls
    assert (call.name, dict(call.args)) == ("issue_refund", {"order_id": "48213", "amount": 10})
    assert [f.kind for f in adv.faults] == ["repeat_write"]


def test_the_result_of_an_injected_call_gets_a_neutral_reply() -> None:
    adv = AdversarialLLM(base, AdversaryConfig(p_violation=1.0), seed=5)
    (reply,) = run(adv, 1, user())
    call = reply.tool_calls[0]
    denied = Message("tool", "Denied by policy (refund_limit).", tool_call_id=call.call_id)
    assert adv.reply(request(user(), denied)) == Reply("Okay.")


def test_corruption_changes_one_number_and_logs_the_sentence() -> None:
    adv = AdversarialLLM(base, AdversaryConfig(p_corrupt=1.0), seed=6)
    (reply,) = run(adv, 1, user())
    (fault,) = adv.faults
    assert fault.kind == "corrupt_money" and fault.original == PAY
    assert reply.text == fault.sentence and reply.text != PAY
    assert fault.token is not None and fault.token in reply.text and "$" in fault.token


def test_no_writes_are_injected_where_the_tools_are_not_offered() -> None:
    adv = AdversarialLLM(base, AdversaryConfig(p_violation=1.0, p_repeat=1.0), seed=7)
    for _ in range(10):
        assert adv.reply(request(user(), tools=())) == Reply(PAY)


@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        ("Done, I refunded $10.00 to the customer.", "corrupt_money"),
        ("Your order should arrive at 7:45 pm.", "corrupt_eta"),
        ("The courier is about 20 minutes away.", "corrupt_eta"),
        ("All set, order 48213 is marked delivered.", "corrupt_order_id"),
    ],
)
def test_corrupt_numbers_targets_each_kind(sentence: str, kind: str) -> None:
    for seed in range(20):
        out = corrupt_numbers(sentence, random.Random(seed))
        assert out is not None
        corrupted, got_kind, original, token = out
        assert got_kind == kind and corrupted != sentence
        assert original in sentence and token in corrupted and token != original


def test_corrupt_numbers_leaves_claimless_sentences_alone() -> None:
    assert corrupt_numbers("I'm calling the customer now.", random.Random(1)) is None


async def test_streams_the_adversarial_reply_on_the_clock() -> None:
    clock = VirtualClock()
    adv = AdversarialLLM(base, AdversaryConfig(p_violation=1.0), seed=8, clock=clock, ttft_s=0.5)
    chunks = []

    async def consume() -> None:
        async for chunk in adv.stream(request(user())):
            chunks.append((clock.now(), chunk))

    import asyncio

    task = asyncio.ensure_future(consume())
    await clock.run_until_idle()
    task.result()
    assert chunks[0][0] == pytest.approx(0.5) and isinstance(chunks[0][1], TextDelta)
    assert any(isinstance(c, ToolCallChunk) for _, c in chunks)
