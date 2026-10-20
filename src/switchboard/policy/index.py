"""An incremental index over a conversation's events for `count(...)` terms (ADR-004).

Rebuilding a by-type index from the whole log on every check makes a long call's checks
quadratic in total. `EventIndex` is updated once per appended event instead: events by type,
plus posting lists keyed by (type, top-level field, scalar value). A count whose filters are all
literal top-level fields is answered from posting list lengths; anything else scans the
smallest matching posting list (or the type bucket), not the log. `scanned` counts the events
examined by those scans, so tests can assert on work done rather than wall time.

Values are keyed the way `==` compares them in the rule language: booleans never equal numbers,
and `1` equals `1.0`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from switchboard.core.events import Event
from switchboard.policy.expr import MISSING, eq, event_field

__all__ = ["EventIndex"]

# Envelope fields that no rule filters on and that would only grow the postings.
_SKIP: Final = frozenset({"seq", "at", "event_id", "causation_id", "conversation_id", "bundle_id"})

_Key = tuple[str, str, tuple[str, object]]


def _scalar_key(value: object) -> tuple[str, object] | None:
    if isinstance(value, bool):
        return ("b", value)
    if isinstance(value, int | float):
        return ("n", value)
    if isinstance(value, str):
        return ("s", value)
    if value is None:
        return ("z", None)
    return None


class EventIndex:
    def __init__(self) -> None:
        self._by_type: dict[str, list[Event]] = {}
        self._postings: dict[_Key, list[Event]] = {}
        self.scanned = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EventIndex):
            return NotImplemented
        return self._by_type == other._by_type

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        sizes = {t: len(v) for t, v in self._by_type.items()}
        return f"EventIndex({sizes})"

    def add(self, event: Event) -> None:
        self._by_type.setdefault(event.type, []).append(event)
        for name in type(event).model_fields:
            if name in _SKIP or name == "type":
                continue
            key = _scalar_key(getattr(event, name))
            if key is not None:
                self._postings.setdefault((event.type, name, key), []).append(event)

    def of_type(self, event_type: str) -> Sequence[Event]:
        return self._by_type.get(event_type, ())

    def count(self, event_type: str, filters: Sequence[tuple[tuple[str, ...], object]]) -> int:
        """Events of `event_type` whose fields equal every `(path, value)` filter."""
        if not filters:
            return len(self._by_type.get(event_type, ()))
        indexed: list[tuple[Sequence[Event], int]] = []
        for i, (path, value) in enumerate(filters):
            key = _scalar_key(value) if len(path) == 1 and path[0] not in _SKIP else None
            if key is not None:
                indexed.append((self._postings.get((event_type, path[0], key), ()), i))
        if indexed:
            candidates, used = min(indexed, key=lambda pair: len(pair[0]))
            rest = [f for i, f in enumerate(filters) if i != used]
            if not rest:
                return len(candidates)
        else:
            candidates, rest = self._by_type.get(event_type, ()), list(filters)
        self.scanned += len(candidates)
        return sum(
            1
            for e in candidates
            if all((v := event_field(e, p)) is not MISSING and eq(v, want) for p, want in rest)
        )
