"""Talker or thinker routing for a committed turn (ADR-008).

The heuristic router is a first-match rule table. Rules are plain data so a bundle can tune
them, and `explain` shows every rule's verdict for telemetry and debugging. Thinker rules come
first: when a turn is both a greeting and a refund request, the refund decides.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from switchboard.brain.types import Tier


@dataclass(frozen=True, slots=True)
class RouteState:
    """What the conversation already knows when the turn is routed.

    `answer_in_evidence`: the answer to a status question is already in the turn's evidence,
    so no tool call is needed. `tool_needed`: something upstream (procedure state, an open
    step) already knows this turn needs a tool.
    """

    answer_in_evidence: bool = False
    tool_needed: bool = False


@dataclass(frozen=True, slots=True)
class RouteDecision:
    tier: Tier
    reason: str


class Router(Protocol):
    def route(self, turn_text: str, state: RouteState) -> RouteDecision: ...


@dataclass(frozen=True, slots=True)
class RouteRule:
    """A rule matches when every condition it sets holds. Unset conditions are ignored.

    `patterns` match (any of them, `re.search`) against the normalized turn: lowercase,
    punctuation other than apostrophes replaced by spaces, whitespace collapsed.
    `state_flag` names a boolean attribute of RouteState that must be true.
    """

    name: str
    tier: Tier
    patterns: tuple[str, ...] = ()
    state_flag: str | None = None
    min_words: int | None = None
    max_words: int | None = None
    min_questions: int | None = None


@dataclass(frozen=True, slots=True)
class RouterConfig:
    rules: tuple[RouteRule, ...]
    default_tier: Tier = "thinker"
    # Joins a second question onto the first in unpunctuated speech ("where is it and when...").
    question_joiners: tuple[str, ...] = ()


_ACK_WORDS = (
    "yes|yeah|yep|yup|no|nope|ok|okay|sure|correct|right|great|perfect|alright|cool|fine|"
    "exactly|please|thanks|thank you|got it|sounds good|that's right|that is right|"
    "that's it|uh huh|mm hmm"
)

DEFAULT_ROUTER_CONFIG = RouterConfig(
    rules=(
        RouteRule(name="tool_needed", tier="thinker", state_flag="tool_needed"),
        RouteRule(
            name="policy_sensitive",
            tier="thinker",
            patterns=(
                r"\brefund",
                r"\bcancel",
                r"\bdeliver(ed)?\b",
                r"\bmark (it|this|the order) (as )?delivered",
                r"\bcharg(e|ed|es|ing)\b",
                r"\bchange (my |the |of )?address",
                r"\baddress change",
                r"\bnew address",
            ),
        ),
        RouteRule(name="multiple_questions", tier="thinker", min_questions=2),
        RouteRule(name="long_turn", tier="thinker", min_words=20),
        RouteRule(
            name="greeting",
            tier="talker",
            patterns=(
                r"^(hi|hello|hey|hiya|good (morning|afternoon|evening))"
                r"( (there|again|everyone|thanks|thank you))*$",
            ),
        ),
        RouteRule(name="acknowledgement", tier="talker", patterns=(rf"^(({_ACK_WORDS}) ?)+$",)),
        RouteRule(
            name="status_read_in_evidence",
            tier="talker",
            state_flag="answer_in_evidence",
            patterns=(
                r"\bwhere('s| is)\b",
                r"\bwhen (will|does|is)\b",
                r"\bstatus\b",
                r"\beta\b",
                r"\btrack",
                r"\barriv",
                r"\bhow long\b",
            ),
        ),
    ),
    default_tier="thinker",
    question_joiners=(
        r"\b(and|also|plus) "
        r"(what|when|where|who|why|how|which|can|could|will|would|is|are|do|does)\b",
    ),
)

_NON_WORD = re.compile(r"[^\w' ]+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _SPACES.sub(" ", _NON_WORD.sub(" ", text.lower())).strip()


@dataclass(frozen=True, slots=True)
class RuleCheck:
    rule: str
    tier: Tier
    matched: bool


@dataclass(frozen=True, slots=True)
class RouteExplanation:
    decision: RouteDecision
    checks: tuple[RuleCheck, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class _Compiled:
    rule: RouteRule
    patterns: tuple[re.Pattern[str], ...]


class HeuristicRouter:
    def __init__(self, config: RouterConfig = DEFAULT_ROUTER_CONFIG) -> None:
        self.config = config
        self._rules = tuple(
            _Compiled(rule, _compile(rule.name, rule.patterns)) for rule in config.rules
        )
        self._joiners = _compile("question_joiners", config.question_joiners)
        for rule in config.rules:
            if rule.state_flag is not None and not hasattr(RouteState(), rule.state_flag):
                raise ValueError(f"rule {rule.name!r}: unknown state flag {rule.state_flag!r}")

    def route(self, turn_text: str, state: RouteState) -> RouteDecision:
        text = normalize(turn_text)
        words = len(text.split())
        questions = self._count_questions(turn_text, text)
        for compiled in self._rules:
            if self._matches(compiled, text, words, questions, state):
                return RouteDecision(compiled.rule.tier, compiled.rule.name)
        return RouteDecision(self.config.default_tier, "default")

    def explain(self, turn_text: str, state: RouteState) -> RouteExplanation:
        text = normalize(turn_text)
        words = len(text.split())
        questions = self._count_questions(turn_text, text)
        checks = tuple(
            RuleCheck(c.rule.name, c.rule.tier, self._matches(c, text, words, questions, state))
            for c in self._rules
        )
        return RouteExplanation(decision=self.route(turn_text, state), checks=checks)

    def _count_questions(self, raw: str, text: str) -> int:
        joined = sum(len(p.findall(text)) for p in self._joiners)
        return max(raw.count("?"), joined + 1 if joined else 0)

    @staticmethod
    def _matches(
        compiled: _Compiled, text: str, words: int, questions: int, state: RouteState
    ) -> bool:
        rule = compiled.rule
        if rule.state_flag is not None and getattr(state, rule.state_flag) is not True:
            return False
        if rule.min_words is not None and words < rule.min_words:
            return False
        if rule.max_words is not None and words > rule.max_words:
            return False
        if rule.min_questions is not None and questions < rule.min_questions:
            return False
        return not compiled.patterns or any(p.search(text) for p in compiled.patterns)


def _compile(owner: str, patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    out: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            out.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"rule {owner!r}: invalid pattern {pattern!r}: {exc}") from exc
    return tuple(out)
