"""The `require` expression language (ADR-004).

A hand-written tokenizer and recursive-descent parser. User strings are never handed to `eval`,
`exec` or `ast`; they become a tree of frozen dataclasses once, at compile time, and the tree is
evaluated against a context on every check.

Grammar, loosest binding first::

    expr     := or_expr
    or_expr  := and_expr ("or" and_expr)*
    and_expr := not_expr ("and" not_expr)*
    not_expr := "not" not_expr | compare
    compare  := primary (cmp_op primary)?          one comparison, no chaining
    cmp_op   := "==" | "!=" | "<" | "<=" | ">" | ">=" | "in" | "not" "in"
    primary  := literal | path | count | list | "(" expr ")"
    path     := ("args" | "evidence" | "state") ("." name)*
    literal  := int | float | string | "true" | "false" | "null"   (numbers may carry a "-")
    list     := "[" (expr ("," expr)* ","?)? "]"
    count    := "count" "(" event_type ("," field "=" value)* ")"
    field    := name ("." name)*                    a path into the event's fields
    value    := literal | path | name ("." name)*   a bare name is a string

Semantics. Evaluation never raises. A path that does not resolve evaluates to `MISSING` and is
recorded; every comparison involving `MISSING` is false, including `!=`. Ordering comparisons
between values that are not both numbers or both strings are false. Booleans are not numbers. In a
boolean position (operands of and/or/not, and the whole expression) only the value `true` counts
as true; anything else, including `MISSING`, counts as false.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from switchboard.core.events import EVENT_TYPES, Event

if TYPE_CHECKING:
    from switchboard.policy.index import EventIndex

__all__ = [
    "MISSING",
    "And",
    "Compare",
    "Count",
    "EvalContext",
    "Evaluation",
    "Expr",
    "ListExpr",
    "Literal",
    "Not",
    "Or",
    "ParseError",
    "Path",
    "evaluate",
    "eq",
    "evaluate_indexed",
    "event_field",
    "holds",
    "index_events",
    "parse",
]

ROOTS: Final = frozenset({"args", "evidence", "state"})
KEYWORDS: Final = frozenset({"and", "or", "not", "in", "true", "false", "null", "count"})
COMPARISON_OPS: Final = frozenset({"==", "!=", "<", "<=", ">", ">="})
MAX_DEPTH: Final = 64
_KNOWN_EVENT_TYPES: Final = frozenset(EVENT_TYPES)


class _Missing:
    """Sentinel for a path that does not resolve. Falsy, and unequal to everything but itself."""

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING: Final = _Missing()


class ParseError(ValueError):
    def __init__(self, message: str, line: int, column: int, source: str = "") -> None:
        self.message = message
        self.line = line
        self.column = column
        self.source = source
        super().__init__(f"line {line}, column {column}: {message}")

    def pointer(self) -> str:
        """The offending source line with a caret under the error column."""
        lines = self.source.split("\n")
        if not 1 <= self.line <= len(lines):
            return ""
        return f"{lines[self.line - 1]}\n{' ' * (self.column - 1)}^"


# Tokenizer


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str  # "int", "float", "str", "name", "op", "eof"
    text: str
    value: object
    line: int
    column: int

    def describe(self) -> str:
        return "end of input" if self.kind == "eof" else repr(self.text)


_TWO_CHAR_OPS: Final = frozenset({"==", "!=", "<=", ">="})
_ONE_CHAR_OPS: Final = frozenset("<>=()[],.-")
_ESCAPES: Final = {"\\": "\\", "'": "'", '"': '"', "n": "\n", "t": "\t"}


def _is_name_start(ch: str) -> bool:
    return ch == "_" or ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _is_name_char(ch: str) -> bool:
    return _is_name_start(ch) or ("0" <= ch <= "9")


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    i, line, col, n = 0, 1, 1, len(source)
    while i < n:
        ch = source[i]
        if ch == "\n":
            i, line, col = i + 1, line + 1, 1
            continue
        if ch in " \t\r":
            i, col = i + 1, col + 1
            continue
        start, start_col = i, col
        if "0" <= ch <= "9":
            while i < n and "0" <= source[i] <= "9":
                i += 1
            if i + 1 < n and source[i] == "." and "0" <= source[i + 1] <= "9":
                i += 1
                while i < n and "0" <= source[i] <= "9":
                    i += 1
                text = source[start:i]
                tokens.append(_Token("float", text, float(text), line, start_col))
            else:
                text = source[start:i]
                tokens.append(_Token("int", text, int(text), line, start_col))
            col += i - start
        elif _is_name_start(ch):
            while i < n and _is_name_char(source[i]):
                i += 1
            text = source[start:i]
            tokens.append(_Token("name", text, text, line, start_col))
            col += i - start
        elif ch in "'\"":
            quote, i = ch, i + 1
            chars: list[str] = []
            start_line = line
            col += 1
            while True:
                if i >= n:
                    raise ParseError("unterminated string", start_line, start_col, source)
                c = source[i]
                if c == quote:
                    i, col = i + 1, col + 1
                    break
                if c == "\\":
                    if i + 1 >= n:
                        raise ParseError("unterminated string", start_line, start_col, source)
                    esc = source[i + 1]
                    if esc not in _ESCAPES:
                        raise ParseError(f"unknown escape '\\{esc}'", line, col, source)
                    chars.append(_ESCAPES[esc])
                    i, col = i + 2, col + 2
                    continue
                chars.append(c)
                if c == "\n":
                    line, col = line + 1, 1
                else:
                    col += 1
                i += 1
            tokens.append(_Token("str", source[start:i], "".join(chars), start_line, start_col))
        elif source[i : i + 2] in _TWO_CHAR_OPS:
            text = source[i : i + 2]
            tokens.append(_Token("op", text, text, line, start_col))
            i, col = i + 2, col + 2
        elif ch in _ONE_CHAR_OPS:
            tokens.append(_Token("op", ch, ch, line, start_col))
            i, col = i + 1, col + 1
        else:
            raise ParseError(f"unexpected character {ch!r}", line, col, source)
    tokens.append(_Token("eof", "", None, line, col))
    return tokens


# Tree


class Expr:
    __slots__ = ()

    def ev(self, env: _Env) -> object:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Literal(Expr):
    value: object

    def ev(self, env: _Env) -> object:
        return self.value


@dataclass(frozen=True, slots=True)
class ListExpr(Expr):
    items: tuple[Expr, ...]

    def ev(self, env: _Env) -> object:
        return [item.ev(env) for item in self.items]


@dataclass(frozen=True, slots=True)
class Path(Expr):
    root: str
    fields: tuple[str, ...]

    def dotted(self) -> str:
        return ".".join((self.root, *self.fields))

    def ev(self, env: _Env) -> object:
        current = _descend(env.root(self.root), self.fields)
        if current is MISSING:
            env.missing.append(self.dotted())
        return current


@dataclass(frozen=True, slots=True)
class Count(Expr):
    event_type: str
    filters: tuple[tuple[str, Expr], ...]
    _split: tuple[tuple[str, ...], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_split", tuple(tuple(k.split(".")) for k, _ in self.filters))

    def ev(self, env: _Env) -> object:
        wanted = [expr.ev(env) for _, expr in self.filters]
        if any(v is MISSING for v in wanted):
            return 0
        pairs = tuple(zip(self._split, wanted, strict=True))
        if env.index is not None:
            return env.index.count(self.event_type, pairs)
        events = env.events_of(self.event_type)
        if not wanted:
            return len(events)
        return sum(1 for e in events if all(_eq(_event_field(e, k), v) for k, v in pairs))


@dataclass(frozen=True, slots=True)
class Compare(Expr):
    op: str
    left: Expr
    right: Expr

    def ev(self, env: _Env) -> object:
        return _compare(self.op, self.left.ev(env), self.right.ev(env))


@dataclass(frozen=True, slots=True)
class Not(Expr):
    operand: Expr

    def ev(self, env: _Env) -> object:
        return self.operand.ev(env) is not True


@dataclass(frozen=True, slots=True)
class And(Expr):
    operands: tuple[Expr, ...]

    def ev(self, env: _Env) -> object:
        return all(o.ev(env) is True for o in self.operands)


@dataclass(frozen=True, slots=True)
class Or(Expr):
    operands: tuple[Expr, ...]

    def ev(self, env: _Env) -> object:
        return any(o.ev(env) is True for o in self.operands)


# Parser


class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.tokens = _tokenize(source)
        self.pos = 0
        self.depth = 0

    @property
    def tok(self) -> _Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> _Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def advance(self) -> _Token:
        tok = self.tokens[self.pos]
        if tok.kind != "eof":
            self.pos += 1
        return tok

    def error(self, message: str, tok: _Token | None = None) -> ParseError:
        at = tok or self.tok
        return ParseError(message, at.line, at.column, self.source)

    def is_op(self, text: str) -> bool:
        return self.tok.kind == "op" and self.tok.text == text

    def is_name(self, text: str, tok: _Token | None = None) -> bool:
        t = tok or self.tok
        return t.kind == "name" and t.text == text

    def expect_op(self, text: str, context: str = "") -> _Token:
        if not self.is_op(text):
            raise self.error(f"expected '{text}'{context}, found {self.tok.describe()}")
        return self.advance()

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise self.error(f"expression nested too deeply (limit {MAX_DEPTH})")

    def parse(self) -> Expr:
        expr = self.parse_or()
        if self.tok.kind != "eof":
            raise self.error(f"unexpected {self.tok.describe()} after the end of the expression")
        return expr

    def parse_or(self) -> Expr:
        operands = [self.parse_and()]
        while self.is_name("or"):
            self.advance()
            operands.append(self.parse_and())
        return operands[0] if len(operands) == 1 else Or(tuple(operands))

    def parse_and(self) -> Expr:
        operands = [self.parse_not()]
        while self.is_name("and"):
            self.advance()
            operands.append(self.parse_not())
        return operands[0] if len(operands) == 1 else And(tuple(operands))

    def parse_not(self) -> Expr:
        if self.is_name("not"):
            self.advance()
            self.enter()
            operand = self.parse_not()
            self.depth -= 1
            return Not(operand)
        return self.parse_compare()

    def comparison_op(self) -> str | None:
        if self.tok.kind == "op" and self.tok.text in COMPARISON_OPS:
            return self.tok.text
        if self.is_name("in"):
            return "in"
        if self.is_name("not") and self.is_name("in", self.peek()):
            return "not in"
        return None

    def parse_compare(self) -> Expr:
        left = self.parse_primary()
        op = self.comparison_op()
        if op is None:
            if self.is_op("="):
                raise self.error("unexpected '='; use '==' to compare")
            return left
        self.advance()
        if op == "not in":
            self.advance()
        right = self.parse_primary()
        if self.comparison_op() is not None:
            raise self.error("chained comparison is not supported; join comparisons with 'and'")
        return Compare(op, left, right)

    def parse_primary(self) -> Expr:
        tok = self.tok
        if tok.kind in ("int", "float", "str"):
            self.advance()
            return Literal(tok.value)
        if tok.kind == "op":
            if tok.text == "-":
                return self.parse_negative()
            if tok.text == "(":
                self.advance()
                self.enter()
                inner = self.parse_or()
                self.depth -= 1
                self.expect_op(")", " to close '('")
                return inner
            if tok.text == "[":
                return self.parse_list()
            raise self.error(f"expected a value, found {tok.describe()}")
        if tok.kind == "name":
            return self.parse_name()
        raise self.error(f"expected a value, found {tok.describe()}")

    def parse_negative(self) -> Expr:
        self.advance()
        num = self.tok
        if num.kind not in ("int", "float"):
            raise self.error(f"expected a number after '-', found {num.describe()}")
        self.advance()
        assert isinstance(num.value, int | float)
        return Literal(-num.value)

    def parse_list(self) -> Expr:
        self.advance()
        self.enter()
        items: list[Expr] = []
        while not self.is_op("]"):
            items.append(self.parse_or())
            if self.is_op(","):
                self.advance()
            elif not self.is_op("]"):
                raise self.error(f"expected ',' or ']' in list, found {self.tok.describe()}")
        self.advance()
        self.depth -= 1
        return ListExpr(tuple(items))

    def parse_name(self) -> Expr:
        tok = self.tok
        name = tok.text
        if name in ("true", "false", "null"):
            self.advance()
            return Literal({"true": True, "false": False, "null": None}[name])
        if name in ROOTS:
            return self.parse_path()
        if name == "count":
            return self.parse_count()
        if name in KEYWORDS:
            raise self.error(f"expected a value, found {tok.describe()}")
        raise self.error(f"unknown name {name!r}; paths start with args, evidence or state")

    def parse_path(self) -> Path:
        root = self.advance().text
        return Path(root, tuple(self.dotted_tail()))

    def dotted_tail(self) -> list[str]:
        fields: list[str] = []
        while self.is_op("."):
            self.advance()
            if self.tok.kind != "name":
                raise self.error(f"expected a field name after '.', found {self.tok.describe()}")
            fields.append(self.advance().text)
        return fields

    def dotted_name(self, what: str) -> tuple[str, _Token]:
        tok = self.tok
        if tok.kind != "name":
            raise self.error(f"expected {what}, found {tok.describe()}")
        self.advance()
        return ".".join([tok.text, *self.dotted_tail()]), tok

    def parse_count(self) -> Count:
        self.advance()
        self.expect_op("(", " after 'count'")
        type_tok = self.tok
        if type_tok.kind == "str":
            self.advance()
            assert isinstance(type_tok.value, str)
            event_type = type_tok.value
        else:
            event_type, type_tok = self.dotted_name("an event type")
        if event_type not in _KNOWN_EVENT_TYPES:
            raise self.error(f"unknown event type {event_type!r}", type_tok)
        filters: list[tuple[str, Expr]] = []
        while self.is_op(","):
            self.advance()
            if self.is_op(")"):
                break
            key, _ = self.dotted_name("a field name")
            self.expect_op("=", f" after filter field {key!r}")
            filters.append((key, self.parse_filter_value()))
        self.expect_op(")", " to close 'count('")
        return Count(event_type, tuple(filters))

    def parse_filter_value(self) -> Expr:
        tok = self.tok
        if tok.kind == "name" and tok.text not in ROOTS and tok.text not in KEYWORDS:
            text, _ = self.dotted_name("a value")
            return Literal(text)
        if tok.kind == "name" and tok.text not in ROOTS | {"true", "false", "null"}:
            raise self.error(f"expected a value, found {tok.describe()}")
        if self.is_op("(") or self.is_op("["):
            raise self.error(f"expected a plain value, found {tok.describe()}")
        return self.parse_primary()


def parse(source: str) -> Expr:
    """Parse a `require` expression. Raises `ParseError` with a line and column on bad input."""
    return _Parser(source).parse()


# Evaluation


@dataclass(frozen=True, slots=True)
class EvalContext:
    args: Mapping[str, Any]
    evidence: Mapping[str, Any]
    state: Mapping[str, Any]
    events: Sequence[Event]
    # When set, `count(...)` reads this incrementally maintained index instead of `events`.
    index: EventIndex | None = None


@dataclass(frozen=True, slots=True)
class Evaluation:
    value: object
    missing: tuple[str, ...]


class _Env:
    __slots__ = ("_by_type", "args", "events", "evidence", "index", "missing", "state")

    def __init__(self, ctx: EvalContext, by_type: dict[str, list[Event]] | None = None) -> None:
        self.args = ctx.args
        self.evidence = ctx.evidence
        self.state = ctx.state
        self.events = ctx.events
        self.index = ctx.index
        self.missing: list[str] = []
        self._by_type = by_type

    def root(self, name: str) -> object:
        if name == "args":
            return self.args
        if name == "evidence":
            return self.evidence
        return self.state

    def events_of(self, event_type: str) -> Sequence[Event]:
        if self.index is not None:
            return self.index.of_type(event_type)
        if self._by_type is None:
            self._by_type = index_events(self.events)
        return self._by_type.get(event_type, [])


def index_events(events: Sequence[Event]) -> dict[str, list[Event]]:
    by_type: dict[str, list[Event]] = {}
    for event in events:
        by_type.setdefault(event.type, []).append(event)
    return by_type


def _descend(current: object, fields: Sequence[str]) -> object:
    for name in fields:
        if not isinstance(current, Mapping):
            return MISSING
        try:
            if name not in current:
                return MISSING
            current = current[name]
        except Exception:
            return MISSING
    return current


def event_field(event: Event, keys: tuple[str, ...]) -> object:
    """The value at a dotted path into an event's fields, or `MISSING`."""
    if keys[0] not in type(event).model_fields:
        return MISSING
    return _descend(getattr(event, keys[0]), keys[1:])


_event_field = event_field


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def eq(left: object, right: object) -> bool:
    """Equality as the rule language defines it: MISSING equals nothing, bools only bools."""
    if left is MISSING or right is MISSING:
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    try:
        return bool(left == right)
    except Exception:
        return False


_eq = eq


def _compare(op: str, left: object, right: object) -> bool:
    if left is MISSING or right is MISSING:
        return False
    if op == "==":
        return _eq(left, right)
    if op == "!=":
        return not _eq(left, right)
    if op in ("in", "not in"):
        if not isinstance(right, list | tuple):
            return False
        found = any(_eq(left, item) for item in right)
        return found if op == "in" else not found
    if _is_number(left) and _is_number(right):
        a: Any = left
        b: Any = right
    elif isinstance(left, str) and isinstance(right, str):
        a, b = left, right
    else:
        return False
    if op == "<":
        return bool(a < b)
    if op == "<=":
        return bool(a <= b)
    if op == ">":
        return bool(a > b)
    return bool(a >= b)


def run(expr: Expr, env: _Env) -> Evaluation:
    value = expr.ev(env)
    return Evaluation(value, tuple(dict.fromkeys(env.missing)))


def evaluate_indexed(expr: Expr, ctx: EvalContext, by_type: dict[str, list[Event]]) -> Evaluation:
    """Like `evaluate`, reusing events already grouped by type with `index_events`."""
    return run(expr, _Env(ctx, by_type))


def evaluate(expr: Expr, ctx: EvalContext) -> Evaluation:
    """Evaluate a parsed expression. Never raises; unresolved paths are listed in `missing`."""
    return run(expr, _Env(ctx))


def holds(expr: Expr, ctx: EvalContext) -> bool:
    return evaluate(expr, ctx).value is True
