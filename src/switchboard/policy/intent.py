"""Which committed user turns open a new request for a write tool (ADR-009, revised).

Write-tool idempotency keys carry an intent counter per tool instead of the proposing step, so a
write the model proposes again in a later turn of the same request replays from the ledger
instead of running twice. The counter must move only when the user asks for the action again,
and that call belongs to this layer, not to the model: a model that re-proposes a refund cannot
open a second refund by itself.

`RequestDetector` decides it deterministically from the committed turn text. Each write tool has
cue sets; a turn requests the tool when every word of one cue set appears in it (after a crude
suffix strip, so "refunded" meets "refund"). By default the cue set is the words of the tool name
minus generic verbs: `mark_delivered` needs "mark" and "deliver", `issue_refund` needs "refund".
A status question ("did the refund go through?", "is it marked?") is not a request. A bundle
can name its own cues per tool.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

__all__ = ["RequestDetector"]

_GENERIC: Final = frozenset(
    {"get", "issue", "lookup", "look", "do", "run", "make", "perform", "execute", "create"}
)
_STATUS_OPENERS: Final = frozenset(
    {"did", "is", "was", "has", "have", "had", "does", "are", "were", "any"}
)
_MODAL_OPENERS: Final = frozenset({"will", "would", "can", "could"})
_WORD = re.compile(r"[a-z']+")
_CLAUSE = re.compile(r"[^,.;!?]*\?")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _default_cues(tool: str) -> tuple[frozenset[str], ...]:
    words = [w for w in tool.lower().split("_") if w and w not in _GENERIC]
    return (frozenset(_stem(w) for w in words),) if words else ()


def _status_question(text: str) -> bool:
    for clause in _CLAUSE.findall(text.lower()):
        words = _WORD.findall(clause)
        if not words:
            continue
        if words[0] in _STATUS_OPENERS:
            return True
        if words[0] in _MODAL_OPENERS and (len(words) < 2 or words[1] != "you"):
            return True
    return False


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
        if _status_question(text):
            return []
        stems = {_stem(w) for w in _WORD.findall(text.lower())}
        return sorted(
            tool
            for tool in set(tools)
            if any(group and group <= stems for group in self.cues(tool))
        )
