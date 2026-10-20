"""Property tests: the parser agrees with an independent reference evaluator.

Trees are generated directly, rendered to source with the fewest parentheses precedence allows,
and the parsed result is compared against evaluating the generated tree. A precedence bug in the
parser shows up as a truth-value disagreement.
"""

import contextlib
from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from switchboard.policy.expr import EvalContext, ParseError, evaluate, parse

VARS = {"a": 3, "b": -7, "c": 0, "d": 12}


@dataclass(frozen=True)
class Int:
    value: int


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Cmp:
    op: str
    left: Int | Var
    right: Int | Var


@dataclass(frozen=True)
class BoolLit:
    value: bool


@dataclass(frozen=True)
class NotT:
    operand: "BoolTree"


@dataclass(frozen=True)
class AndT:
    left: "BoolTree"
    right: "BoolTree"


@dataclass(frozen=True)
class OrT:
    left: "BoolTree"
    right: "BoolTree"


BoolTree = BoolLit | Cmp | NotT | AndT | OrT

# Precedence levels, loosest first.
PREC = {OrT: 1, AndT: 2, NotT: 3, Cmp: 4, BoolLit: 5}

atoms = st.one_of(
    st.integers(min_value=-1000, max_value=1000).map(Int),
    st.sampled_from(sorted(VARS)).map(Var),
)
comparisons = st.builds(Cmp, st.sampled_from(["==", "!=", "<", "<=", ">", ">="]), atoms, atoms)
leaves = st.one_of(st.booleans().map(BoolLit), comparisons)
trees: st.SearchStrategy[BoolTree] = st.recursive(
    leaves,
    lambda inner: st.one_of(
        st.builds(NotT, inner),
        st.builds(AndT, inner, inner),
        st.builds(OrT, inner, inner),
    ),
    max_leaves=12,
)


def render_atom(atom: Int | Var) -> str:
    return f"args.{atom.name}" if isinstance(atom, Var) else str(atom.value)


def render(tree: BoolTree, parent: int = 0) -> str:
    prec = PREC[type(tree)]
    match tree:
        case BoolLit(value):
            text = "true" if value else "false"
        case Cmp(op, left, right):
            text = f"{render_atom(left)} {op} {render_atom(right)}"
        case NotT(operand):
            text = f"not {render(operand, prec)}"
        case AndT(left, right):
            # Left-associative: a right operand at the same level needs parentheses.
            text = f"{render(left, prec)} and {render(right, prec + 1)}"
        case OrT(left, right):
            text = f"{render(left, prec)} or {render(right, prec + 1)}"
    return f"({text})" if prec < parent else text


def ref_atom(atom: Int | Var) -> int:
    return VARS[atom.name] if isinstance(atom, Var) else atom.value


def reference(tree: BoolTree) -> bool:
    match tree:
        case BoolLit(value):
            return value
        case Cmp(op, left, right):
            lv, rv = ref_atom(left), ref_atom(right)
            return {
                "==": lv == rv,
                "!=": lv != rv,
                "<": lv < rv,
                "<=": lv <= rv,
                ">": lv > rv,
                ">=": lv >= rv,
            }[op]
        case NotT(operand):
            return not reference(operand)
        case AndT(left, right):
            return reference(left) and reference(right)
        case OrT(left, right):
            return reference(left) or reference(right)


CTX = EvalContext(args=VARS, evidence={}, state={}, events=[])


@settings(max_examples=400, deadline=None)
@given(trees)
def test_parse_and_evaluate_matches_reference(tree: BoolTree) -> None:
    source = render(tree)
    assert evaluate(parse(source), CTX).value is reference(tree), source


@settings(max_examples=200, deadline=None)
@given(trees)
def test_fully_parenthesised_rendering_matches_too(tree: BoolTree) -> None:
    source = render(tree, parent=99)
    assert evaluate(parse(source), CTX).value is reference(tree), source


ALPHABET = st.sampled_from(list("()[],.=<>!'\"-_ \n\t0123456789") + ["and", "or", "not", "in"])
TOKENS = st.sampled_from(
    ["args", "evidence", "state", "count", "tool.executed", "true", "null", "x", "1.5", "'s'"]
)


@settings(max_examples=500, deadline=None)
@given(st.text())
def test_random_text_only_raises_parse_error(source: str) -> None:
    with contextlib.suppress(ParseError):
        parse(source)


@settings(max_examples=500, deadline=None)
@given(st.lists(st.one_of(ALPHABET, TOKENS), max_size=30).map(" ".join))
def test_random_token_soup_only_raises_parse_error(source: str) -> None:
    try:
        expr = parse(source)
    except ParseError:
        return
    evaluate(expr, CTX)
