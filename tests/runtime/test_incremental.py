"""Per-turn check cost stays flat as a conversation grows (no rebuild from the whole log).

Cost is counted, not timed: events the policy layer scans (through the incremental index, or a
full-log index if anything still builds one) plus values the grounding evidence walks.
"""

from __future__ import annotations

from typing import Any

import pytest

from switchboard.brain import ToolCallChunk
from switchboard.core import (
    ConversationStarted,
    LegOpened,
    PolicyVerdict,
    SpeechPlayed,
    SpeechSentenceGated,
    ToolExecuted,
    ToolProposed,
    TurnCommitted,
)
from switchboard.grounding import evidence as evidence_module
from switchboard.policy import CheckContext, ProposedCall
from switchboard.policy import checker as checker_module
from switchboard.policy import expr as expr_module
from switchboard.runtime import Conversation, replay

from .harness import make_world

TURNS = 500


class Meter:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.log_scans = 0
        self.evidence_walks = 0
        for module in (checker_module, expr_module):
            original = module.index_events

            def counting(events: Any, _orig: Any = original) -> Any:
                self.log_scans += len(events)
                return _orig(events)

            monkeypatch.setattr(module, "index_events", counting)
        add = evidence_module.Evidence.add

        def walking(ev: Any, value: object, key: str | None = None) -> None:
            self.evidence_walks += 1
            add(ev, value, key)

        monkeypatch.setattr(evidence_module.Evidence, "add", walking)

    def total(self, conv: Conversation) -> int:
        return self.log_scans + self.evidence_walks + conv.state.index.scanned


async def one_turn(conv: Conversation, leg_id: str, n: int) -> None:
    turn = f"turn_{n}"
    committed = await conv.emit(
        TurnCommitted, leg_id=leg_id, turn_id=turn, text="What is my pay?", endpointing_ms=300
    )
    call_id = f"call_{n}"
    proposed = await conv.emit(
        ToolProposed,
        leg_id=leg_id,
        causation_id=committed.event_id,
        call_id=call_id,
        tool="get_courier_pay",
        args={"courier_id": "C-77"},
    )
    await conv.emit(
        PolicyVerdict,
        leg_id=leg_id,
        causation_id=proposed.event_id,
        call_id=call_id,
        allowed=True,
        rule_ids=[],
    )
    await conv.emit(
        ToolExecuted,
        leg_id=leg_id,
        causation_id=proposed.event_id,
        call_id=call_id,
        tool="get_courier_pay",
        result={"week_total": f"${100 + n}.40", "deliveries": n},
    )
    utterance = f"{turn}/u0"
    await conv.emit(
        SpeechSentenceGated,
        leg_id=leg_id,
        causation_id=committed.event_id,
        utterance_id=utterance,
        sentence="Here it is.",
        decision="release",
        tier=1,
        claims=[],
    )
    await conv.emit(
        SpeechPlayed,
        leg_id=leg_id,
        utterance_id=utterance,
        text_played="Here it is.",
        interrupted=False,
    )


def check_turn(conv: Conversation, n: int) -> list[Any]:
    """What a turn asks of the policy and grounding layers."""
    effects = conv.bundle.tool_effects()
    refund = ToolCallChunk(f"r{n}", "issue_refund", {"order_id": "48213", "amount": 10})
    pay = ToolCallChunk(f"p{n}", "get_courier_pay", {"courier_id": "C-77"})
    verdicts = [conv.check(refund, effects), conv.check(pay, effects)]
    decision = conv.gate.review(f"You earned ${100 + n}.40 this week.", conv.grounding_evidence())
    return [*verdicts, decision.decision]


async def test_per_turn_check_cost_does_not_grow_with_the_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = make_world()
    conv = Conversation(world.runtime, world.bundle, "conv_long")
    world.runtime.conversations[conv.id] = conv
    await conv.emit(ConversationStarted, leg_id=None, tenant_id="t", customer_id="C-77")
    await conv.emit(LegOpened, leg_id="leg_in", role="inbound", party="courier")
    meter = Meter(monkeypatch)

    costs: dict[int, int] = {}
    for n in range(1, TURNS + 1):
        await one_turn(conv, "leg_in", n)
        if n in (10, 100, TURNS):
            before = meter.total(conv)
            result = check_turn(conv, n)
            costs[n] = meter.total(conv) - before
            assert result[-1] == "release"
            # Same verdicts as a check that scans the whole log.
            full = CheckContext(
                evidence=dict(conv.state.evidence),
                state={"completed_steps": list(conv.state.completed_steps)},
                events=tuple(conv.events),
                tool_effects=conv.bundle.tool_effects(),
            )
            refund = ProposedCall("issue_refund", {"order_id": "48213", "amount": 10})
            assert conv.bundle.policy.check(refund, full) == result[0]

    assert costs[TURNS] == costs[10], costs
    assert costs[TURNS] <= 10, costs
    assert conv.state == replay(await world.store.read(conv.id))
