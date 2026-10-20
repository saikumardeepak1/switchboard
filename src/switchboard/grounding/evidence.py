"""The turn's evidence set: canonical values from tool results, retrieved facts and memory.

Everything is walked once into per-kind indexes so tier 1 matching is a set lookup or a short
scan. Strings are run through the same claim extractor the gate uses on speech, so "$12.50" in a
tool result and "twelve fifty" in a sentence meet at the same canonical value.

Matching rules (ADR-005):

- money, number, identifier, date: exact. Identifiers ignore case and separators.
- values are typed by where they sit. A money claim matches only money evidence: a number under a
  money key (amount, price, total, refund, pay, fee, cost, ..., `*_cents`, `*_dollars`) or a
  currency amount written in text. An identifier matches a string, or a number under an id-like
  key. Any number matches a number claim. So `eta_minutes: 20` never backs "$20", and
  `order_id: "48213"` never backs "48,213 dollars". A date said
  without a year matches dated evidence on month and day; a date said with a year needs the year.
- time: exact, or within 5 minutes when the claim is hedged. A time said without am or pm matches
  either half of the day.
- duration: exact, or within 20% of the evidence when hedged. A spoken range matches anything
  inside its bounds. With `now_minutes`, every evidence time t also yields the duration t - now
  (forward, up to 12 hours), which is how "in about 20 minutes" checks against an ETA.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal

from switchboard.grounding.claims import Claim, ClaimValue, extract_claims

TIME_TOLERANCE_MIN = 5
DURATION_TOLERANCE = Decimal("0.2")
_MAX_DERIVED_MIN = 12 * 60

_KEY_UNITS = {
    "seconds": 1,
    "second": 1,
    "secs": 1,
    "sec": 1,
    "minutes": 60,
    "minute": 60,
    "mins": 60,
    "min": 60,
    "hours": 3600,
    "hour": 3600,
    "hrs": 3600,
    "days": 86_400,
    "weeks": 604_800,
}
_ID_KEY_HINTS = (
    "id",
    "number",
    "code",
    "ref",
    "ticket",
    "order",
    "sku",
    "tracking",
    "confirmation",
    "invoice",
    "case",
    "booking",
)
_MONEY_KEY_HINTS = frozenset(
    {
        "amount", "price", "total", "subtotal", "refund", "refunded", "pay", "paid", "payout",
        "fee", "fees", "cost", "costs", "charge", "charged", "balance", "credit", "tip", "tips",
        "earnings", "earned", "wage", "wages", "salary", "bonus", "fare", "cents", "cent",
        "dollars", "dollar", "usd", "money", "owed", "due", "deposit", "discount", "payment",
    }
)  # fmt: skip
_WORD_RE = re.compile(r"[a-z0-9]+")


def _id_key(value: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", value.upper())


def _key_parts(key: str | None) -> list[str]:
    return _WORD_RE.findall(key.lower()) if key else []


def _money_key(parts: list[str]) -> bool:
    return any(p in _MONEY_KEY_HINTS for p in parts)


def _id_like_key(parts: list[str]) -> bool:
    return any(p in _ID_KEY_HINTS for p in parts)


def _norm(d: Decimal) -> int | Decimal:
    return int(d) if d == d.to_integral_value() else d.normalize()


def _two_places(d: Decimal) -> bool:
    return d == d.quantize(Decimal("0.01"))


def _circular(a: int, b: int) -> int:
    diff = abs(a - b) % 1440
    return min(diff, 1440 - diff)


class Evidence:
    def __init__(
        self,
        *,
        tool_results: Iterable[object] = (),
        facts: Iterable[object] = (),
        memory: Iterable[str] = (),
        now_minutes: int | None = None,
    ) -> None:
        self.now_minutes = now_minutes
        self._money: set[int] = set()
        self._numbers: set[int | Decimal] = set()
        self._times: set[int] = set()
        self._durations: set[int] = set()
        self._dates_md: set[tuple[int, int]] = set()
        self._dates_iso: set[str] = set()
        self._ids: dict[str, str] = {}
        self._words: set[str] = set()
        for source in (*tool_results, *facts, *memory):
            self.add(source)

    def _parts(self) -> tuple[object, ...]:
        return (
            self.now_minutes,
            self._money,
            self._numbers,
            self._times,
            self._durations,
            self._dates_md,
            self._dates_iso,
            self._ids,
            self._words,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Evidence):
            return NotImplemented
        return self._parts() == other._parts()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Evidence(ids={len(self._ids)}, numbers={len(self._numbers)})"

    def union(self, other: Evidence) -> Evidence:
        """A new evidence set holding both. Costs the size of the two indexes, not a re-walk."""
        out = Evidence(now_minutes=self.now_minutes)
        for ev in (self, other):
            out._money |= ev._money
            out._numbers |= ev._numbers
            out._times |= ev._times
            out._durations |= ev._durations
            out._dates_md |= ev._dates_md
            out._dates_iso |= ev._dates_iso
            out._words |= ev._words
            for key, display in ev._ids.items():
                out._ids.setdefault(key, display)
        return out

    # building

    def add(self, value: object, key: str | None = None) -> None:
        """Walk a value (dicts, lists, dataclasses, pydantic models, scalars) into the index."""
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, str):
            self._add_text(value, key)
        elif isinstance(value, int | float | Decimal):
            self._add_number(Decimal(str(value)), key, is_int=isinstance(value, int))
        elif isinstance(value, Mapping):
            for k, v in value.items():
                self._words.update(_key_parts(str(k)))
                self.add(v, str(k))
        elif isinstance(value, list | tuple | set | frozenset):
            for item in value:
                self.add(item, key)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            self.add(dataclasses.asdict(value), key)
        elif callable(getattr(value, "model_dump", None)):
            self.add(value.model_dump(), key)  # type: ignore[attr-defined]
        else:
            self._add_text(str(value), key)

    def _add_number(self, d: Decimal, key: str | None, *, is_int: bool) -> None:
        parts = _key_parts(key)
        self._numbers.add(_norm(d))
        if any(p in ("cents", "cent") for p in parts):
            self._money.add(int(d))
        elif _money_key(parts) and _two_places(d):
            self._money.add(int(d * 100))
        for part in parts:
            if part in _KEY_UNITS:
                self._durations.add(int(d * _KEY_UNITS[part]))
        if is_int and _id_like_key(parts):
            self._add_id(str(int(d)))

    def _add_text(self, text: str, key: str | None) -> None:
        self._words.update(_WORD_RE.findall(text.lower()))
        parts = _key_parts(key)
        for claim in extract_claims(text):
            self._index_claim(claim, money_key=_money_key(parts), id_key=_id_like_key(parts))
        compact = _id_key(text)
        id_like_key = _id_like_key(parts)
        has_space = bool(re.search(r"\s", text.strip()))
        has_digit = any(ch.isdigit() for ch in compact)
        if has_digit and len(compact) <= 40 and (id_like_key or not has_space):
            self._add_id(text.strip())

    def _add_id(self, display: str) -> None:
        self._ids.setdefault(_id_key(display), display)

    def _index_claim(self, claim: Claim, *, money_key: bool, id_key: bool) -> None:
        v = claim.value
        if claim.kind == "money" and isinstance(v, int):
            self._money.add(v)
            self._numbers.add(_norm(Decimal(v) / 100))
        elif claim.kind == "number" and isinstance(v, int | Decimal):
            self._numbers.add(v)
            # A bare number in text is money or an id only when its key says so.
            if money_key and _two_places(Decimal(v)):
                self._money.add(int(Decimal(v) * 100))
            if id_key and isinstance(v, int) and "." not in claim.raw:
                self._add_id(str(v))
        elif claim.kind == "time" and isinstance(v, int):
            self._times.add(v)
            self._times.update(a for a in claim.alternatives if isinstance(a, int))
        elif claim.kind == "duration" and isinstance(v, int):
            self._durations.add(v)
            self._durations.update(a for a in claim.alternatives if isinstance(a, int))
        elif claim.kind == "date":
            if isinstance(v, tuple):
                self._dates_md.add(v)
            elif isinstance(v, str):
                self._dates_iso.add(v)
                parsed = date.fromisoformat(v)
                self._dates_md.add((parsed.month, parsed.day))
                self._numbers.add(parsed.year)
        elif claim.kind == "identifier" and isinstance(v, str):
            self._add_id(v)

    # reading

    @property
    def words(self) -> frozenset[str]:
        """Lowercase word bag over all evidence text and keys, for lexical scorers."""
        return frozenset(self._words)

    def durations(self) -> set[int]:
        """Explicit durations plus ETA minus now for every evidence time, when now is known."""
        out = set(self._durations)
        if self.now_minutes is not None:
            for t in self._times:
                ahead = (t - self.now_minutes) % 1440
                if 0 < ahead <= _MAX_DERIVED_MIN:
                    out.add(ahead * 60)
        return out

    def supports(self, claim: Claim) -> bool:
        v = claim.value
        if claim.kind == "money":
            return v in self._money
        if claim.kind == "number":
            return v in self._numbers
        if claim.kind == "identifier":
            return isinstance(v, str) and _id_key(v) in self._ids
        if claim.kind == "date":
            if isinstance(v, tuple):
                return v in self._dates_md
            return v in self._dates_iso
        if claim.kind == "time":
            tolerance = TIME_TOLERANCE_MIN if claim.approximate else 0
            candidates = [c for c in (v, *claim.alternatives) if isinstance(c, int)]
            return any(_circular(c, t) <= tolerance for c in candidates for t in self._times)
        if claim.kind == "duration" and isinstance(v, int):
            pool = self.durations()
            if not claim.approximate:
                return v in pool
            bounds = [b for b in claim.alternatives if isinstance(b, int)]
            for e in pool:
                if abs(v - e) <= DURATION_TOLERANCE * e:
                    return True
                if len(bounds) == 2 and bounds[0] <= e <= bounds[1]:
                    return True
            return False
        return False

    def nearest(self, claim: Claim) -> ClaimValue | None:
        """The closest evidence value of the claim's kind, for rewrite hints."""
        v = claim.value
        if claim.kind == "identifier" and isinstance(v, str):
            target = _id_key(v)
            scored = [
                (difflib.SequenceMatcher(None, target, k).ratio(), d)
                for k, d in sorted(self._ids.items())
            ]
            best = max(scored, default=None, key=lambda s: s[0])
            return best[1] if best is not None and best[0] >= 0.5 else None
        if claim.kind == "date":
            if isinstance(v, tuple):
                pool_md = sorted(self._dates_md)
                return min(
                    pool_md,
                    key=lambda d: abs((d[0] * 31 + d[1]) - (v[0] * 31 + v[1])),
                    default=None,
                )
            if isinstance(v, str):
                day = date.fromisoformat(v).toordinal()
                return min(
                    sorted(self._dates_iso),
                    key=lambda d: abs(date.fromisoformat(d).toordinal() - day),
                    default=None,
                )
            return None
        if isinstance(v, int | Decimal):
            if claim.kind == "time" and isinstance(v, int):
                return min(sorted(self._times), key=lambda t: _circular(t, v), default=None)
            pool: list[int | Decimal]
            if claim.kind == "money":
                pool = sorted(self._money)
            elif claim.kind == "duration":
                pool = sorted(self.durations())
            else:
                pool = sorted(self._numbers)
            want = Decimal(v)
            return min(pool, key=lambda p: abs(Decimal(p) - want), default=None)
        return None
