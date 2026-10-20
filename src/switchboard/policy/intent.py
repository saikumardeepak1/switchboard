"""Which committed user turns open a new request for a write tool (ADR-009, revised).

Write-tool idempotency keys carry an intent counter per tool instead of the proposing step, so a
write the model proposes again in a later turn of the same request replays from the ledger
instead of running twice. The counter must move only when the user asks for the action again,
and that call belongs to this layer, not to the model: a model that re-proposes a refund cannot
open a second refund by itself.

`RequestDetector` decides it deterministically from the committed turn text. Each write tool has
cue sets; a clause names the tool when every word of one cue set appears in it (after a crude
suffix strip, so "refunded" meets "refund"). By default the cue set is the words of the tool name
minus generic verbs: `mark_delivered` needs "mark" and "deliver", `issue_refund` needs "refund".
A bundle can name its own cues per tool.

Naming the tool is not enough: the clause must also ask for it. After leading fillers ("okay",
"great", "so") a clause asks when it opens with a request cue ("please", "can you", "I need",
"I want", "I'd like", "go ahead and", "I can't" for a stuck action) or with an imperative verb
("mark it delivered", "refund them"). Thanks, statements about what already happened ("the refund
came through", "so it's marked delivered"), status questions and negations ("don't refund it")
ask for nothing. A clause of politeness alone ("yes, please, refund it") carries over to the next.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

__all__ = ["RequestDetector"]

_GENERIC: Final = frozenset(
    {"get", "issue", "lookup", "look", "do", "run", "make", "perform", "execute", "create"}
)
_WORD = re.compile(r"[a-z']+")
_CLAUSE = re.compile(r"[,.;!?]+|\b(?:and|but|then)\b")
_FILLERS: Final = frozenset(
    {
        "okay", "ok", "alright", "great", "cool", "perfect", "good", "nice", "awesome", "so",
        "now", "yes", "yeah", "yep", "sure", "well", "oh", "um", "uh", "hi", "hey", "hello",
        "also", "just", "right", "fine", "mm", "hm", "too",
    }
)  # fmt: skip
_POLITE: Final = frozenset({"please", "kindly"})
# Request openers, as word sequences matched at the start of a clause (after fillers).
_OPENERS: Final = tuple(
    tuple(o.split())
    for o in (
        "can you", "could you", "would you", "will you", "can u", "would you please",
        "i need", "we need", "i want", "we want", "i'd like", "i would like", "we'd like",
        "go ahead and", "let's", "i need you to", "need you to", "help me",
        "i can't", "i cannot", "i'm unable to", "i am unable to", "it won't let me",
    )
)  # fmt: skip
# Verbs a clause may open with as an imperative, beyond the tool's own cue words.
_IMPERATIVES: Final = frozenset(
    {
        "issue", "make", "do", "get", "give", "send", "process", "change", "update", "set",
        "put", "cancel", "mark", "refund", "add", "remove", "apply", "credit", "reverse",
    }
)  # fmt: skip
_NEGATIONS: Final = frozenset({"don't", "dont", "not", "never", "no", "do"})
# After "I need to" or "I want to", these ask about the action rather than for it.
_INQUIRIES: Final = frozenset({"know", "check", "see", "ask", "understand", "find", "hear"})


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _default_cues(tool: str) -> tuple[frozenset[str], ...]:
    words = [w for w in tool.lower().split("_") if w and w not in _GENERIC]
    return (frozenset(_stem(w) for w in words),) if words else ()


def _clauses(text: str) -> list[list[str]]:
    return [_WORD.findall(part) for part in _CLAUSE.split(text.lower())]


def _strip(words: list[str], drop: frozenset[str]) -> list[str]:
    i = 0
    while i < len(words) and words[i] in drop:
        i += 1
    return words[i:]


def _asks(words: list[str], carried: bool) -> bool:
    """The clause, fillers already stripped, opens with a request and is not negated."""
    rest: list[str] | None = None
    if words and words[0] in _POLITE:
        rest = _strip(words, _POLITE | _FILLERS)
    else:
        for opener in sorted(_OPENERS, key=len, reverse=True):
            if tuple(words[: len(opener)]) == opener:
                rest = _strip(words[len(opener) :], _POLITE | _FILLERS)
                break
    if rest is None:
        if not carried:
            # An imperative in base form only: "refunded" reports, it does not order.
            return bool(words) and words[0] in _IMPERATIVES
        rest = words
    if rest and rest[0] in _NEGATIONS:
        return False
    return not (len(rest) >= 2 and rest[0] == "to" and rest[1] in _INQUIRIES)


class RequestDetector:
    def __init__(self, cues: Mapping[str, Sequence[Sequence[str]]] | None = None) -> None:
        self._cues = {
            tool: tuple(frozenset(_stem(w.lower()) for w in group) for group in groups)
            for tool, groups in (cues or {}).items()
        }

    def cues(self, tool: str) -> tuple[frozenset[str], ...]:
        return self._cues.get(tool) or _default_cues(tool)

    def requested(self, text: str, tools: Iterable[str]) -> list[str]:
        """The tools, sorted, that this user turn newly asks for."""
        wanted = set(tools)
        found: set[str] = set()
        carried = False
        for clause in _clauses(text):
            words = _strip(clause, _FILLERS)
            if not words:
                continue
            if all(w in _POLITE | _FILLERS for w in words):
                carried = True
                continue
            if _asks(words, carried):
                stems = {_stem(w) for w in words}
                found.update(
                    tool
                    for tool in wanted
                    if any(group and group <= stems for group in self.cues(tool))
                )
            carried = False
        return sorted(found)
