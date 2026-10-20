from typing import Any

import pytest

from switchboard.policy.expr import (
    MISSING,
    And,
    Compare,
    Count,
    EvalContext,
    ListExpr,
    Literal,
    Not,
    Or,
    ParseError,
    Path,
    evaluate,
    holds,
    parse,
)
from tests.policy.helpers import advanced, executed, proposed


def ctx(
    args: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    events: list[Any] | None = None,
) -> EvalContext:
    return EvalContext(
        args=args or {}, evidence=evidence or {}, state=state or {}, events=events or []
    )


# Parsing: shapes and precedence


def test_parses_literals() -> None:
    assert parse("1") == Literal(1)
    assert parse("-2") == Literal(-2)
    assert parse("2.5") == Literal(2.5)
    assert parse("'hi'") == Literal("hi")
    assert parse('"hi \\"there\\""') == Literal('hi "there"')
    assert parse("true") == Literal(True)
    assert parse("false") == Literal(False)
    assert parse("null") == Literal(None)


def test_parses_paths() -> None:
    assert parse("args.amount") == Path("args", ("amount",))
    assert parse("evidence.order.status") == Path("evidence", ("order", "status"))
    assert parse("state") == Path("state", ())


def test_parses_list_literal_with_trailing_comma() -> None:
    assert parse("[1, 'a', true,]") == ListExpr((Literal(1), Literal("a"), Literal(True)))
    assert parse("[]") == ListExpr(())


def test_and_binds_tighter_than_or() -> None:
    tree = parse("true or false and false")
    assert tree == Or((Literal(True), And((Literal(False), Literal(False)))))


def test_not_binds_tighter_than_and() -> None:
    tree = parse("not true and false")
    assert tree == And((Not(Literal(True)), Literal(False)))


def test_comparison_binds_tighter_than_not() -> None:
    tree = parse("not args.a == 1")
    assert tree == Not(Compare("==", Path("args", ("a",)), Literal(1)))


def test_parentheses_override_precedence() -> None:
    tree = parse("(true or false) and false")
    assert tree == And((Or((Literal(True), Literal(False))), Literal(False)))


def test_or_chain_is_flat() -> None:
    assert parse("true or false or true") == Or((Literal(True), Literal(False), Literal(True)))


def test_parses_in_and_not_in() -> None:
    assert parse("args.x in [1, 2]") == Compare(
        "in", Path("args", ("x",)), ListExpr((Literal(1), Literal(2)))
    )
    assert parse("args.x not in [1]") == Compare(
        "not in", Path("args", ("x",)), ListExpr((Literal(1),))
    )


def test_parses_count_with_filters() -> None:
    tree = parse("count(tool.executed, tool=issue_refund, result.ok=true) < 1")
    assert tree == Compare(
        "<",
        Count(
            "tool.executed",
            (("tool", Literal("issue_refund")), ("result.ok", Literal(True))),
        ),
        Literal(1),
    )


def test_count_accepts_quoted_type_and_path_values() -> None:
    tree = parse("count('tool.proposed', args.order_id=args.order_id)")
    assert tree == Count("tool.proposed", (("args.order_id", Path("args", ("order_id",))),))


@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
def test_all_comparison_operators_parse(op: str) -> None:
    assert parse(f"args.a {op} 3") == Compare(op, Path("args", ("a",)), Literal(3))


# Parse errors


@pytest.mark.parametrize(
    ("source", "line", "column", "fragment"),
    [
        ("args.a ==", 1, 10, "expected a value"),
        ("args.a == 1 and", 1, 16, "expected a value"),
        ("(true", 1, 6, "expected ')'"),
        ("args.a = 1", 1, 8, "'='"),
        ("foo.bar", 1, 1, "unknown name 'foo'"),
        ("'unterminated", 1, 1, "unterminated string"),
        ("1 < 2 < 3", 1, 7, "chained comparison"),
        ("args.a == 1\n  and $", 2, 7, "unexpected character '$'"),
        ("count(nope.nope)", 1, 7, "unknown event type 'nope.nope'"),
        ("count(tool.executed, tool)", 1, 26, "expected '='"),
        ("true false", 1, 6, "unexpected 'false'"),
        ("", 1, 1, "expected a value"),
        ("args.", 1, 6, "expected a field name"),
    ],
)
def test_parse_errors_carry_position(source: str, line: int, column: int, fragment: str) -> None:
    with pytest.raises(ParseError) as info:
        parse(source)
    err = info.value
    assert (err.line, err.column) == (line, column), str(err)
    assert fragment in err.message
    assert f"line {line}, column {column}" in str(err)


def test_deep_nesting_is_a_parse_error_not_recursion_error() -> None:
    with pytest.raises(ParseError, match="nested too deeply"):
        parse("(" * 5000 + "1" + ")" * 5000)
    with pytest.raises(ParseError, match="nested too deeply"):
        parse("not " * 5000 + "true")


# Evaluation


def test_compares_args_numbers() -> None:
    expr = parse("args.amount <= 25")
    assert holds(expr, ctx(args={"amount": 25}))
    assert holds(expr, ctx(args={"amount": 10.5}))
    assert not holds(expr, ctx(args={"amount": 26}))


def test_missing_path_makes_comparisons_false_and_is_reported() -> None:
    expr = parse("args.amount <= 25")
    result = evaluate(expr, ctx())
    assert result.value is False
    assert result.missing == ("args.amount",)
    neq = evaluate(parse("evidence.x != 1"), ctx())
    assert neq.value is False
    assert neq.missing == ("evidence.x",)


def test_bare_missing_path_evaluates_to_sentinel() -> None:
    result = evaluate(parse("state.nothing.here"), ctx(state={"nothing": 3}))
    assert result.value is MISSING
    assert result.missing == ("state.nothing.here",)


def test_type_mismatch_ordering_is_false_not_error() -> None:
    assert not holds(parse("'a' < 3"), ctx())
    assert not holds(parse("args.a >= 3"), ctx(args={"a": [1]}))
    assert not holds(parse("null < 1"), ctx())


def test_bool_is_not_a_number() -> None:
    assert not holds(parse("true == 1"), ctx())
    assert not holds(parse("true < 2"), ctx())
    assert holds(parse("true != 1"), ctx())


def test_int_and_float_compare_numerically() -> None:
    assert holds(parse("1 == 1.0"), ctx())
    assert holds(parse("2 > 1.5"), ctx())


def test_strings_compare() -> None:
    assert holds(parse("evidence.status == 'delivered'"), ctx(evidence={"status": "delivered"}))
    assert holds(parse("'a' < 'b'"), ctx())


def test_in_list() -> None:
    expr = parse("args.reason in ['damaged', 'missing']")
    assert holds(expr, ctx(args={"reason": "damaged"}))
    assert not holds(expr, ctx(args={"reason": "late"}))
    assert not holds(expr, ctx())
    assert holds(parse("args.reason not in ['x']"), ctx(args={"reason": "y"}))
    assert not holds(parse("args.reason not in ['x']"), ctx())
    assert not holds(parse("1 in 3"), ctx())
    assert not holds(parse("true in [1]"), ctx())


def test_only_true_is_truthy() -> None:
    assert holds(parse("evidence.ok"), ctx(evidence={"ok": True}))
    assert not holds(parse("evidence.ok"), ctx(evidence={"ok": 1}))
    assert not holds(parse("evidence.ok"), ctx(evidence={"ok": "yes"}))
    assert not holds(parse("1 and true"), ctx())


def test_boolean_connectives() -> None:
    c = ctx(args={"a": 1, "b": 2})
    assert holds(parse("args.a == 1 and args.b == 2"), c)
    assert holds(parse("args.a == 9 or args.b == 2"), c)
    assert not holds(parse("not (args.a == 1)"), c)


def test_missing_reported_inside_boolean_expressions() -> None:
    result = evaluate(parse("args.a == 1 or evidence.b == 2"), ctx(args={"a": 1}))
    assert result.value is True
    assert result.missing == ()
    result = evaluate(parse("args.a == 9 or evidence.b == 2"), ctx(args={"a": 1}))
    assert result.value is False
    assert result.missing == ("evidence.b",)


def test_count_over_core_events() -> None:
    events = [
        executed("lookup_order", seq=1),
        executed("issue_refund", seq=2, result={"ok": True, "amount": 10}),
        proposed("issue_refund", {"amount": 30}, seq=3),
        executed("issue_refund", seq=4, result={"ok": False}),
        advanced("verify_identity", seq=5),
    ]
    c = ctx(args={"tool": "issue_refund"}, events=events)
    assert evaluate(parse("count(tool.executed)"), c).value == 3
    assert evaluate(parse("count(tool.executed, tool=issue_refund)"), c).value == 2
    assert evaluate(parse("count(tool.executed, tool=issue_refund, result.ok=true)"), c).value == 1
    assert evaluate(parse("count(tool.executed, tool=args.tool)"), c).value == 2
    assert evaluate(parse("count(tool.proposed, args.amount=30)"), c).value == 1
    assert evaluate(parse("count(procedure.advanced, step=verify_identity)"), c).value == 1
    assert evaluate(parse("count(handoff.requested)"), c).value == 0
    assert evaluate(parse("count(tool.executed, nonexistent=1)"), c).value == 0
    assert holds(parse("count(tool.executed, tool=issue_refund) < 3"), c)


def test_count_filter_on_missing_path_value_matches_nothing() -> None:
    c = ctx(events=[executed("issue_refund")])
    result = evaluate(parse("count(tool.executed, tool=args.tool)"), c)
    assert result.value == 0
    assert result.missing == ("args.tool",)


def test_evaluation_never_raises_on_odd_data() -> None:
    weird: dict[str, Any] = {"a": object(), "b": {"c": None}, "d": [1, {"e": 2}]}
    for source in [
        "args.a < 1",
        "args.a == args.a",
        "args.b.c.d == 1",
        "args.d in args.d",
        "args.d.e == 2",
        "args.a in [1, 2]",
    ]:
        evaluate(parse(source), ctx(args=weird))
