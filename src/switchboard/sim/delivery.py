"""Scenarios for the reference agent in `examples/delivery_support`.

`DeliveryScript` stands in for the model. It is deterministic and reads only the request, the
way a model would: the last message, tool results and evidence in the history. It is written to
be tolerant of recognizer errors (it asks again when it cannot place what it heard) because
simulated callers run with word error injection.

The scenarios are the courier and customer flow from the example: a courier who cannot mark an
order delivered, a lookup that shows no address change in the chat, an outbound call to the
customer on a child leg while the courier keeps talking, and the delivery marked only once the
customer confirms.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import re
from pathlib import Path
from typing import Any

from switchboard.brain import BrainRequest, Message, Reply, ToolCallChunk
from switchboard.sim.assertions import (
    Assertion,
    caller_heard,
    never_before,
    no_corrupted_sentence_released,
    no_sentence_released_with_unsupported_claims,
    outcome,
    tool_executed_once,
    tool_never_executed,
)
from switchboard.sim.caller import Beat, HangUp, Say, WaitFor, WaitForAgent
from switchboard.sim.scenario import CallerSpec, Scenario

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "delivery_support"
STATUSPAGE = EXAMPLE / "fixtures" / "statuspage.json"
CHILD_PROMPT_START = "You are on an outbound phone call"

_COURIER_ID = re.compile(r"\bc[- ]?(\d{2,})\b")
_ORDER_ID = re.compile(r"\b(\d{5})\b")
_AMOUNT = re.compile(r"\$?(\d+(?:\.\d+)?)")
_NUMBER_WORDS = {"five": 5, "eight": 8, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20}
_ACK = re.compile(r"^\W*(ok|okay|sure|thanks|thank you|great|got it|alright)\b")


def money(value: Any) -> str:
    return f"${float(value):.2f}"


def _json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


class DeliveryScript:
    """A deterministic model for the delivery support agent and its outbound child calls."""

    def __init__(self) -> None:
        self._ids = itertools.count(1)

    def call(self, name: str, **args: Any) -> ToolCallChunk:
        return ToolCallChunk(f"call_{next(self._ids)}", name, args)

    def __call__(self, request: BrainRequest) -> Reply:
        if request.messages and request.messages[0].content.startswith(CHILD_PROMPT_START):
            return self.child(request)
        return self.parent(request)

    # the inbound agent

    def parent(self, request: BrainRequest) -> Reply:
        messages = request.messages
        last = messages[-1]
        if last.role == "system" and last.content.startswith("Your last sentence failed"):
            # Regenerate the reply as if the blocked sentence had never been produced.
            return self.parent(dataclasses.replace(request, messages=messages[:-1]))
        calls = _calls(messages)
        results = _results(messages, calls)
        done = {name for name, result in results if "error" not in result}
        order = _last_arg(calls, "order_id") or "48213"
        courier = _last_arg(calls, "courier_id") or "C-77"
        proposed = {name for name, _ in calls.values()}

        pending = _unanswered_child_result(messages)
        if pending is not None and last.role not in ("user", "tool"):
            result = _json(pending.content.split(": ", 1)[-1])
            if result.get("confirmed") is True:
                return Reply("The customer confirmed the new address.", (self.mark(order),))
            return Reply("The customer didn't confirm the change, so the order stays as it is.")

        if last.role == "tool":
            name = calls.get(last.tool_call_id or "", ("", {}))[0]
            if last.content.startswith("Denied by policy"):
                return self.denied(name, last.content, order, proposed)
            return self.after_tool(name, _json(last.content), order, done)

        if last.role != "user":
            return Reply("Okay.")
        return self.answer(last.content, order, courier, done, proposed)

    def answer(
        self, text: str, order: str, courier: str, done: set[str], proposed: set[str]
    ) -> Reply:
        low = text.lower()
        if "bye" in low:
            return Reply("Thanks for calling, goodbye.")
        if m := _COURIER_ID.search(low):
            if "verify_courier" in done:
                return Reply("You're already verified.")
            return Reply(
                "One moment.", (self.call("verify_courier", courier_id=f"C-{m.group(1)}"),)
            )
        if "refund" in low:
            if "issue_refund" in done:
                return Reply("I've already refunded this order.")
            amount = _amount(low)
            return Reply("", (self.call("issue_refund", order_id=order, amount=amount),))
        if "pay" in low or "earn" in low:
            return Reply("Let me check.", (self.call("get_courier_pay", courier_id=courier),))
        if m := _ORDER_ID.search(low):
            if "mark_delivered" in done:
                return Reply(f"Order {m.group(1)} is already marked delivered.")
            if "call_party" in proposed:
                return Reply("I'm already calling the customer now, please stay on the line.")
            return Reply(
                "Let me pull up that order.", (self.call("lookup_order", order_id=m.group(1)),)
            )
        if _ACK.search(low):
            return Reply("Sure.")
        return Reply("Sorry, could you say that again?")

    def after_tool(self, name: str, result: dict[str, Any], order: str, done: set[str]) -> Reply:
        if "error" in result:
            return Reply("Sorry, something went wrong on my side.")
        if name == "verify_courier":
            first = str(result.get("name", "")).split(" ")[0] or "there"
            return Reply(f"Thanks {first}, you're verified.")
        if name == "lookup_order":
            evidence = result.get("evidence", {})
            if isinstance(evidence, dict) and evidence.get("chat_shows_address_change"):
                return Reply("The chat shows the change.", (self.mark(order),))
            return Reply(
                "The chat doesn't show an address change, so I'll call the customer.",
                (self.call_customer(order),),
            )
        if name == "call_party":
            if result.get("status") == "calling":
                return Reply("I'm calling the customer now, please stay on the line.")
            return Reply("I couldn't reach the customer.")
        if name == "get_courier_pay":
            return Reply(
                f"You've earned {money(result['week_total'])} this week across "
                f"{result['deliveries']} deliveries."
            )
        if name == "mark_delivered":
            return Reply(f"All set, order {result.get('order_id', order)} is marked delivered.")
        if name == "issue_refund":
            return Reply(f"Done, I refunded {money(result['amount'])} to the customer.")
        return Reply("Okay.")

    def denied(self, name: str, content: str, order: str, proposed: set[str]) -> Reply:
        if "procedure:requires:verify_identity" in content:
            return Reply("I need to verify you first. What's your courier ID?")
        if name == "mark_delivered":
            if "call_party" not in proposed:
                return Reply("I need the customer to confirm first.", (self.call_customer(order),))
            return Reply("I can't mark it delivered until the customer confirms.")
        if name == "issue_refund":
            return Reply("I can't issue that refund on this call, a teammate can help with it.")
        return Reply("I can't do that one here.")

    def mark(self, order: str) -> ToolCallChunk:
        return self.call("mark_delivered", order_id=order, location="customer's home address")

    def call_customer(self, order: str) -> ToolCallChunk:
        return self.call(
            "call_party",
            party="customer",
            goal=f"Confirm the customer of order {order} asked for delivery to their home "
            "instead of the pin.",
            questions=["Did you ask the courier to bring it to your home address?"],
        )

    # the outbound child call

    def child(self, request: BrainRequest) -> Reply:
        m = _ORDER_ID.search(request.messages[0].content)
        order = f"order {m.group(1)}" if m else "your order"
        users = [msg for msg in request.messages if msg.role == "user"]
        if not users:
            return Reply(
                f"Hi, this is courier support about your {order}. Did you ask the courier to "
                "bring it to your home address instead of the office?"
            )
        low = users[-1].content.lower()
        if re.match(r"\W*(no|nope)\b", low) or "didn't" in low or "office" in low:
            return Reply(
                "Understood, thanks for your time. Goodbye.",
                (self.call("report_result", confirmed=False, notes="Did not ask for a change."),),
            )
        if re.match(r"\W*(yes|yeah|yep|correct)\b", low) or "asked" in low or "home" in low:
            return Reply(
                "Thanks, I'll let the courier know. Goodbye.",
                (self.call("report_result", confirmed=True, notes="Asked for home delivery."),),
            )
        return Reply("Sorry, I didn't catch that. Did you ask for delivery to your home address?")


def _unanswered_child_result(messages: tuple[Message, ...]) -> Message | None:
    """The latest child call result the agent has not reacted to yet.

    Speech from another turn can land after the result in history (the courier's pay answer
    while the customer call ends), so the last message is not always the result itself.
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.role == "system" and m.content.startswith("Result of the call to the"):
            later = messages[i + 1 :]
            reacted = any(
                x.role == "assistant" and x.content.startswith("The customer") for x in later
            )
            return None if reacted else m
    return None


def _calls(messages: tuple[Message, ...]) -> dict[str, tuple[str, dict[str, Any]]]:
    return {
        m.tool_call_id: (m.name or "", _json(m.content))
        for m in messages
        if m.role == "assistant" and m.tool_call_id
    }


def _results(
    messages: tuple[Message, ...], calls: dict[str, tuple[str, dict[str, Any]]]
) -> list[tuple[str, dict[str, Any]]]:
    return [
        (calls[m.tool_call_id][0], _json(m.content))
        for m in messages
        if m.role == "tool"
        and m.tool_call_id in calls
        and not m.content.startswith("Denied by policy")
    ]


def _last_arg(calls: dict[str, tuple[str, dict[str, Any]]], key: str) -> str | None:
    for _, args in reversed(list(calls.values())):
        if key in args:
            return str(args[key])
    return None


def _amount(text: str) -> float:
    if m := _AMOUNT.search(text):
        return float(m.group(1))
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return float(value)
    return 10.0


# scripts for the humans

VERIFY: tuple[Beat, ...] = (Say("Hi, this is courier C-77."), WaitFor("verified"), WaitForAgent())
REPORT: tuple[Beat, ...] = (
    Say("I can't mark order 48213 delivered, the customer wants it at home."),
    WaitFor("calling the customer now"),
    WaitForAgent(),
)
PAY: tuple[Beat, ...] = (Say("While I wait, what is my pay this week?"), WaitFor("182.40"))
DELIVERED: tuple[Beat, ...] = (
    WaitFor("marked delivered", timeout_s=60.0, repeat=False, since_start=True),
    WaitForAgent(),
)
REFUND: tuple[Beat, ...] = (
    Say("Also, please refund the customer ten dollars for the delay."),
    WaitFor("refunded"),
    WaitForAgent(),
)
BYE: tuple[Beat, ...] = (Say("Thanks, bye."), WaitForAgent(timeout_s=5.0), HangUp())


def customer(answer: str) -> CallerSpec:
    return CallerSpec(
        (
            WaitFor("home address", timeout_s=30.0, repeat=False),
            WaitForAgent(),
            Say(answer),
            WaitFor("goodbye", timeout_s=15.0),
            HangUp(),
        )
    )


CONFIRMS = "Yes, I asked the courier to bring it to my home."
DECLINES = "No, it should go to the office."


def standard_assertions() -> list[Assertion]:
    return [
        tool_executed_once("mark_delivered"),
        never_before("tool.executed:mark_delivered", "child.result:confirmed"),
        no_sentence_released_with_unsupported_claims(),
        no_corrupted_sentence_released(),
        caller_heard("courier", "182.40"),
        outcome("resolved"),
    ]


def happy_delivery() -> Scenario:
    return Scenario(
        "happy_delivery",
        EXAMPLE,
        {
            "courier": CallerSpec((*VERIFY, *REPORT, *PAY, *DELIVERED, *BYE)),
            "customer": customer(CONFIRMS),
        },
        DeliveryScript,
        standard_assertions(),
        tenant_id="dashfleet",
        customer_id="C-77",
        statuspage=STATUSPAGE,
    )


def declined_delivery() -> Scenario:
    courier = (
        *VERIFY,
        *REPORT,
        *PAY,
        WaitFor("didn't confirm", timeout_s=60.0, repeat=False, since_start=True),
        WaitForAgent(),
        *BYE,
    )
    return Scenario(
        "declined_delivery",
        EXAMPLE,
        {"courier": CallerSpec(courier), "customer": customer(DECLINES)},
        DeliveryScript,
        [
            tool_never_executed("mark_delivered"),
            no_sentence_released_with_unsupported_claims(),
            no_corrupted_sentence_released(),
            caller_heard("courier", "didn't confirm"),
            outcome("unresolved"),
        ],
        tenant_id="dashfleet",
        customer_id="C-77",
        statuspage=STATUSPAGE,
    )


def refund_delivery() -> Scenario:
    """The happy flow plus a legitimate $10 refund: two writes, both allowed by policy."""
    return Scenario(
        "refund_delivery",
        EXAMPLE,
        {
            "courier": CallerSpec((*VERIFY, *REPORT, *PAY, *DELIVERED, *REFUND, *BYE)),
            "customer": customer(CONFIRMS),
        },
        DeliveryScript,
        [*standard_assertions(), tool_executed_once("issue_refund")],
        tenant_id="dashfleet",
        customer_id="C-77",
        statuspage=STATUSPAGE,
    )
