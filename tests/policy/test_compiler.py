from typing import Any

import pytest

from switchboard.policy import CompileError, compile_policies
from tests.policy.helpers import load_fixture


def rule(rule_id: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"id": rule_id, "when": {"tool": "t"}, "require": "true"}
    base.update(overrides)
    return base


def test_compiles_fixture() -> None:
    policy = compile_policies([load_fixture("delivery.yaml")])
    assert [r.id for r in policy.rules] == [
        "refund_limit",
        "delivered_needs_confirmation",
        "one_refund_per_conversation",
    ]
    assert policy.procedure.steps == ("greet", "verify_identity", "resolve", "close")
    assert policy.procedure.require_before["lookup_order"] == ("verify_identity",)


def test_merges_multiple_documents() -> None:
    docs = [
        {"procedure": {"steps": ["a", "b"], "require_before": {"t": ["a"]}}},
        {"procedure": {"require_before": {"t": ["b"], "u": ["a"]}}, "rules": [rule("r1")]},
        {"rules": [rule("r2")]},
    ]
    policy = compile_policies(docs)
    assert [r.id for r in policy.rules] == ["r1", "r2"]
    assert policy.procedure.require_before == {"t": ("a", "b"), "u": ("a",)}


def test_empty_input_compiles_to_permissive_policy() -> None:
    policy = compile_policies([])
    assert policy.rules == ()
    assert policy.procedure.steps == ()


def test_tool_filter_accepts_list() -> None:
    policy = compile_policies([{"rules": [rule("r", when={"tool": ["a", "b"]})]}])
    assert policy.rules[0].tools == frozenset({"a", "b"})


@pytest.mark.parametrize(
    ("docs", "message"),
    [
        ([{"rules": [rule("r"), rule("r")]}], "duplicate rule id 'r'"),
        ([{"rules": [rule("r")]}, {"rules": [rule("r")]}], "duplicate rule id 'r'"),
        ([{"rules": [rule("procedure:x")]}], "reserved"),
        ([{"rules": [rule("r", require="args.a ==")]}], "rule 'r': line 1, column 10"),
        ([{"rules": [rule("r", when={})]}], "rule 'r': 'when' needs 'tool' or 'effect'"),
        ([{"rules": [rule("r", when={"effect": "delete"})]}], "effect must be 'read' or 'write'"),
        ([{"rules": [rule("r", when={"tol": "x"})]}], "unknown key 'tol' in 'when'"),
        ([{"rules": [rule("r", extra=1)]}], "unknown key 'extra'"),
        ([{"rules": [{"id": "r", "when": {"tool": "t"}}]}], "rule 'r': missing 'require'"),
        ([{"rules": [{"when": {"tool": "t"}, "require": "true"}]}], "rules[0]: missing 'id'"),
        ([{"rules": [rule("r", require=5)]}], "'require' must be a string"),
        ([{"rules": [rule("r", hint=5)]}], "'hint' must be a string"),
        ([{"rules": "nope"}], "'rules' must be a list"),
        ([{"policies": []}], "unknown top-level key 'policies'"),
        (["nope"], "document 0 must be a mapping"),
        ([{"procedure": {"steps": ["a"]}}, {"procedure": {"steps": ["b"]}}], "more than once"),
        ([{"procedure": {"steps": ["a"], "require_before": {"t": ["z"]}}}], "unknown step 'z'"),
        ([{"procedure": {"steps": "a"}}], "'procedure.steps' must be a list of strings"),
        ([{"procedure": {"stepz": []}}], "unknown key 'stepz' in 'procedure'"),
        (
            [
                {"procedure": {"steps": ["a", "b"], "completes": {"t": "a"}}},
                {"procedure": {"completes": {"t": "b"}}},
            ],
            "conflicting 'completes' for tool 't'",
        ),
    ],
)
def test_compile_errors(docs: list[Any], message: str) -> None:
    with pytest.raises(CompileError) as info:
        compile_policies(docs)
    assert message in str(info.value)


def test_parse_error_message_points_at_source() -> None:
    with pytest.raises(CompileError) as info:
        compile_policies([{"rules": [rule("r", require="args.a = 1")]}])
    text = str(info.value)
    assert "args.a = 1" in text
    assert "       ^" in text
