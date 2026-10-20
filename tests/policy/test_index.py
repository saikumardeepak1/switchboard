"""The incremental event index gives the same answers as scanning the log, while scanning less."""

from __future__ import annotations

import random
from typing import Any

from switchboard.core.events import AnyEvent
from switchboard.policy import (
    CheckContext,
    EvalContext,
    EventIndex,
    ProposedCall,
    compile_policies,
    evaluate,
    parse,
)
from tests.policy.helpers import advanced, executed, proposed

EXPRESSIONS = [
    "count(tool.executed)",
    "count(tool.executed, tool=issue_refund)",
    "count(tool.executed, tool=issue_refund, call_id=call_3)",
    "count(tool.executed, tool=args.tool)",
    "count(tool.executed, result.ok=true)",
    "count(tool.proposed, args.order_id=args.order_id)",
    "count(procedure.advanced, step=verify_identity)",
    "count(tool.executed, tool=never_called)",
    "count(tool.executed, replayed=false)",
]


def random_log(rng: random.Random, n: int) -> list[AnyEvent]:
    tools = ["issue_refund", "lookup_order", "mark_delivered"]
    out: list[AnyEvent] = []
    for seq in range(1, n + 1):
        kind = rng.randrange(3)
        tool = rng.choice(tools)
        if kind == 0:
            result: dict[str, Any] = {"ok": rng.choice([True, False, 1])}
            out.append(executed(tool, seq=seq, result=result, call_id=f"call_{rng.randrange(5)}"))
        elif kind == 1:
            out.append(proposed(tool, {"order_id": str(rng.randrange(4))}, seq=seq))
        else:
            out.append(advanced(rng.choice(["verify_identity", "resolve"]), seq=seq))
    return out


def test_indexed_counts_match_a_full_scan() -> None:
    rng = random.Random(7)
    for _ in range(30):
        events = random_log(rng, rng.randrange(0, 60))
        index = EventIndex()
        for event in events:
            index.add(event)
        args = {"tool": rng.choice(["issue_refund", "lookup_order"]), "order_id": "2"}
        for source in EXPRESSIONS:
            expr = parse(source)
            scanned = evaluate(expr, EvalContext(args, {}, {}, events))
            indexed = evaluate(expr, EvalContext(args, {}, {}, (), index=index))
            assert indexed == scanned, source


def test_a_single_literal_filter_is_answered_without_scanning() -> None:
    index = EventIndex()
    for seq in range(1, 1001):
        index.add(executed("lookup_order" if seq % 10 else "issue_refund", seq=seq))
    expr = parse("count(tool.executed, tool=issue_refund)")
    assert evaluate(expr, EvalContext({}, {}, {}, (), index=index)).value == 100
    assert index.scanned == 0


def test_checker_uses_the_index_and_the_given_steps_instead_of_the_log() -> None:
    policy = compile_policies(
        [
            {
                "procedure": {
                    "steps": ["verify_identity", "resolve"],
                    "require_before": {"issue_refund": ["verify_identity"]},
                },
                "rules": [
                    {
                        "id": "one_refund",
                        "when": {"tool": "issue_refund"},
                        "require": "count(tool.executed, tool=issue_refund) < 1",
                    }
                ],
            }
        ],
        tool_effects={"issue_refund": "write"},
    )
    index = EventIndex()
    index.add(executed("issue_refund", seq=1))
    call = ProposedCall("issue_refund", {"amount": 5})
    unverified = policy.check(call, CheckContext(index=index, completed_steps=()))
    assert unverified.rule_ids == ["procedure:requires:verify_identity", "one_refund"]
    verified = policy.check(call, CheckContext(index=index, completed_steps=["verify_identity"]))
    assert verified.rule_ids == ["one_refund"]


def test_indexes_built_from_the_same_events_are_equal() -> None:
    events = random_log(random.Random(3), 40)
    a, b = EventIndex(), EventIndex()
    for event in events:
        a.add(event)
        b.add(event)
    assert a == b
    b.add(advanced("resolve", seq=99))
    assert a != b
