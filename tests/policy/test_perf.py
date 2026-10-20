import time

import pytest

from switchboard.policy import CheckContext, ProposedCall, compile_policies
from tests.policy.helpers import advanced, executed, proposed

TOOLS = ["issue_refund", "lookup_order", "mark_delivered", "update_address", "cancel_order"]
TEMPLATES = [
    "args.amount <= {n}",
    "evidence.customer_confirmed_address == true and args.amount < {n}",
    "count(tool.executed, tool=issue_refund) < {n}",
    "args.reason in ['damaged', 'missing', 'late'] or state.tier == 'gold'",
    "not (evidence.fraud_score > {n}) and count(procedure.advanced, step=verify_identity) >= 1",
]


@pytest.mark.perf
def test_p99_check_latency_under_1ms_for_50_rules() -> None:
    rules = [
        {
            "id": f"rule_{i}",
            "when": {"tool": TOOLS[i % len(TOOLS)]} if i % 7 else {"effect": "write"},
            "require": TEMPLATES[i % len(TEMPLATES)].format(n=i + 1),
            "hint": f"hint {i}",
        }
        for i in range(50)
    ]
    doc = {
        "procedure": {
            "steps": ["greet", "verify_identity", "resolve", "close"],
            "require_before": {tool: ["verify_identity"] for tool in TOOLS},
        },
        "rules": rules,
    }
    policy = compile_policies([doc], tool_effects={tool: "write" for tool in TOOLS})
    events = [advanced("verify_identity", seq=1)]
    for seq in range(2, 40):
        events.append(executed(TOOLS[seq % len(TOOLS)], seq=seq))
        events.append(proposed("lookup_order", {"order_id": str(seq)}, seq=seq + 100))
    ctx = CheckContext(
        evidence={"customer_confirmed_address": True, "fraud_score": 3},
        state={"tier": "gold"},
        events=events,
    )
    call = ProposedCall(tool="issue_refund", args={"amount": 20, "reason": "late"}, call_id="c")

    for _ in range(200):
        policy.check(call, ctx)
    samples = []
    for _ in range(2000):
        start = time.perf_counter()
        policy.check(call, ctx)
        samples.append(time.perf_counter() - start)
    samples.sort()
    p99 = samples[int(len(samples) * 0.99)]
    print(f"policy check p99: {p99 * 1e6:.1f} us over {len(rules)} rules, {len(events)} events")
    assert p99 < 1e-3
