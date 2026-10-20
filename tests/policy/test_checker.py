from collections.abc import Sequence
from typing import Any

import pytest

from switchboard.core.events import AnyEvent
from switchboard.policy import (
    CheckContext,
    CompiledPolicy,
    ProposedCall,
    Verdict,
    compile_policies,
)
from tests.policy.helpers import advanced, executed, load_fixture


@pytest.fixture
def policy() -> CompiledPolicy:
    return compile_policies([load_fixture("delivery.yaml")])


def context(
    evidence: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    events: Sequence[AnyEvent] = (),
    tool_effects: dict[str, str] | None = None,
) -> CheckContext:
    return CheckContext(
        evidence=evidence or {}, state=state or {}, events=events, tool_effects=tool_effects
    )


def call(tool: str, **args: Any) -> ProposedCall:
    return ProposedCall(tool=tool, args=args, call_id="call_9")


def test_allowed_verdict_is_empty(policy: CompiledPolicy) -> None:
    assert policy.check(call("issue_refund", amount=20), context()) == Verdict(True, [], None)


def test_argument_bound(policy: CompiledPolicy) -> None:
    verdict = policy.check(call("issue_refund", amount=40), context())
    assert verdict == Verdict(
        False, ["refund_limit"], "Refunds above $25 need a human. Offer a handoff."
    )


def test_missing_argument_fails_closed(policy: CompiledPolicy) -> None:
    verdict = policy.check(call("issue_refund"), context())
    assert not verdict.allowed
    assert verdict.rule_ids == ["refund_limit"]


def test_required_evidence(policy: CompiledPolicy) -> None:
    done = [advanced("verify_identity")]
    denied = policy.check(call("mark_delivered"), context(events=done))
    assert denied.rule_ids == ["delivered_needs_confirmation"]
    assert denied.repair_hint is not None and "Confirm the address" in denied.repair_hint
    allowed = policy.check(
        call("mark_delivered"),
        context(evidence={"customer_confirmed_address": True}, events=done),
    )
    assert allowed.allowed


def test_count_per_conversation(policy: CompiledPolicy) -> None:
    events = [executed("issue_refund", result={"ok": True})]
    verdict = policy.check(call("issue_refund", amount=5), context(events=events))
    assert verdict.rule_ids == ["one_refund_per_conversation"]
    assert verdict.repair_hint == (
        "Rule one_refund_per_conversation failed: count(tool.executed, tool=issue_refund) < 1"
    )


def test_procedure_prerequisite_from_event(policy: CompiledPolicy) -> None:
    denied = policy.check(call("lookup_order", order_id="1"), context())
    assert denied == Verdict(
        False,
        ["procedure:requires:verify_identity"],
        "Complete step 'verify_identity' before calling lookup_order.",
    )
    ok = policy.check(
        call("lookup_order", order_id="1"), context(events=[advanced("verify_identity")])
    )
    assert ok.allowed


def test_procedure_step_completed_by_tool(policy: CompiledPolicy) -> None:
    events = [executed("verify_identity", result={"ok": True})]
    assert policy.check(call("lookup_order"), context(events=events)).allowed
    failed = [executed("verify_identity", result={"ok": False})]
    assert not policy.check(call("lookup_order"), context(events=failed)).allowed


def test_several_failures_report_all_ids_and_join_hints(policy: CompiledPolicy) -> None:
    events = [executed("issue_refund", result={"ok": True})]
    verdict = policy.check(call("issue_refund", amount=99), context(events=events))
    assert verdict.rule_ids == ["refund_limit", "one_refund_per_conversation"]
    assert verdict.repair_hint == (
        "Refunds above $25 need a human. Offer a handoff. "
        "Rule one_refund_per_conversation failed: count(tool.executed, tool=issue_refund) < 1"
    )
    mixed = policy.check(call("mark_delivered"), context())
    assert mixed.rule_ids == [
        "procedure:requires:verify_identity",
        "delivered_needs_confirmation",
    ]
    assert mixed.repair_hint is not None
    assert mixed.repair_hint.startswith("Complete step 'verify_identity'")


def test_unrelated_tool_is_allowed(policy: CompiledPolicy) -> None:
    assert policy.check(call("say_goodbye"), context()).allowed


def test_state_paths() -> None:
    policy = compile_policies(
        [{"rules": [{"id": "open", "when": {"tool": "t"}, "require": "state.status == 'open'"}]}]
    )
    assert policy.check(call("t"), context(state={"status": "open"})).allowed
    assert not policy.check(call("t"), context(state={"status": "closed"})).allowed


EFFECT_DOC = {
    "rules": [
        {
            "id": "writes_need_identity",
            "when": {"effect": "write"},
            "require": "evidence.identity_verified == true",
        },
        {"id": "reads_logged", "when": {"effect": "read"}, "require": "state.logging == true"},
    ]
}


def test_effect_filter_with_compile_time_map() -> None:
    policy = compile_policies([EFFECT_DOC], tool_effects={"refund": "write", "lookup": "read"})
    assert policy.check(call("refund"), context()).rule_ids == ["writes_need_identity"]
    assert policy.check(call("lookup"), context()).rule_ids == ["reads_logged"]
    assert policy.check(call("lookup"), context(state={"logging": True})).allowed


def test_effect_filter_with_check_time_map_overrides_compile_time() -> None:
    policy = compile_policies([EFFECT_DOC], tool_effects={"refund": "read"})
    verdict = policy.check(call("refund"), context(tool_effects={"refund": "write"}))
    assert verdict.rule_ids == ["writes_need_identity"]


def test_unknown_effect_is_treated_as_write() -> None:
    policy = compile_policies([EFFECT_DOC])
    assert policy.check(call("mystery"), context()).rule_ids == ["writes_need_identity"]


def test_tool_and_effect_must_both_match() -> None:
    doc = {
        "rules": [{"id": "r", "when": {"tool": ["a", "b"], "effect": "write"}, "require": "false"}]
    }
    policy = compile_policies([doc], tool_effects={"a": "write", "b": "read"})
    assert not policy.check(call("a"), context()).allowed
    assert policy.check(call("b"), context()).allowed
    assert policy.check(call("c"), context()).allowed


def test_explain_reports_missing_paths(policy: CompiledPolicy) -> None:
    report = policy.explain(call("issue_refund"), context())
    assert report.verdict.rule_ids == ["refund_limit"]
    assert report.missing == {"refund_limit": ("args.amount",)}


def test_check_is_pure(policy: CompiledPolicy) -> None:
    events = [executed("issue_refund", result={"ok": True})]
    ctx = context(events=events)
    first = policy.check(call("issue_refund", amount=99), ctx)
    second = policy.check(call("issue_refund", amount=99), ctx)
    assert first == second
    assert len(events) == 1
