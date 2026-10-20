"""Checkable claim extraction for the grounding gate (ADR-005).

`extract_claims` turns one spoken sentence into typed, normalized claims: money, number, time,
date, duration and identifier. It is pure and fast because it runs on every sentence before TTS.

Canonical values:

- money: integer cents
- number: `int` when integral, otherwise a normalized `Decimal`
- time: minutes since midnight
- duration: whole seconds
- date: `(month, day)`, or an ISO `YYYY-MM-DD` string when a year was said
- identifier: uppercase string without spaces (hyphens kept)

A time said without am or pm lists the other half of the day in `alternatives`, and a range such as
"20 to 30 minutes" keeps its bounds there. Vague quantities ("a couple", "a few", "several") and
filler ("one moment", "just a second") are deliberately not claims: they carry no checkable fact,
and treating them as claims would block harmless bridge speech (ADR-008).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

ClaimKind = Literal["money", "number", "time", "date", "duration", "identifier"]
ClaimValue = int | Decimal | str | tuple[int, int]


@dataclass(frozen=True)
class Claim:
    kind: ClaimKind
    raw: str
    value: ClaimValue
    span: tuple[int, int]
    approximate: bool = False
    alternatives: tuple[ClaimValue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-friendly form, for `SpeechSentenceGated.claims`."""
        out: dict[str, Any] = {
            "kind": self.kind,
            "raw": self.raw,
            "value": _jsonable(self.value),
            "span": [self.span[0], self.span[1]],
            "approximate": self.approximate,
        }
        if self.alternatives:
            out["alternatives"] = [_jsonable(v) for v in self.alternatives]
        return out


def _jsonable(value: ClaimValue) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


# Vocabulary

UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}
_NUMBER_WORDS = UNITS.keys() | TEENS.keys() | TENS.keys() | SCALES.keys() | ORDINALS.keys()
_HOUR_WORDS = {w: v for w, v in (UNITS | TEENS).items() if 1 <= v <= 12}

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

DURATION_UNITS = {
    "second": 1,
    "seconds": 1,
    "sec": 1,
    "secs": 1,
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "hour": 3600,
    "hours": 3600,
    "hr": 3600,
    "hrs": 3600,
    "day": 86_400,
    "days": 86_400,
    "week": 604_800,
    "weeks": 604_800,
}
# "a minute", "one second": conversational filler, not a promise about time.
_FILLER_UNITS = {"second", "sec", "minute", "min"}
_DURATION_ADJECTIVES = {"more", "business", "full", "whole", "extra", "additional"}

DOLLAR_WORDS = {"dollar", "dollars", "buck", "bucks", "usd"}
CENT_WORDS = {"cent", "cents"}
_MONEY_CONTEXT = DOLLAR_WORDS | {
    "$",
    "refund",
    "refunded",
    "charge",
    "charged",
    "price",
    "cost",
    "costs",
    "fee",
    "fees",
    "total",
    "balance",
    "credit",
    "credited",
    "pay",
    "paid",
    "payment",
    "owe",
    "amount",
    "bill",
    "billed",
}

ID_KEYWORDS = {
    "order",
    "ticket",
    "case",
    "confirmation",
    "tracking",
    "reference",
    "ref",
    "invoice",
    "booking",
    "reservation",
    "claim",
    "account",
    "policy",
    "transaction",
    "shipment",
    "parcel",
    "incident",
}
_ID_FILLERS = {"number", "no", "num", "nr", "#", "id", "code", "is", "was", ":", "."}

_HEDGES_BEFORE = {
    "about",
    "around",
    "roughly",
    "approximately",
    "approx",
    "nearly",
    "almost",
    "maybe",
    "~",
    "under",
    "within",
}
_HEDGES_BEFORE_PAIRS = {
    ("close", "to"),
    ("up", "to"),
    ("less", "than"),
    ("more", "than"),
    ("at", "least"),
    ("at", "most"),
    ("just", "under"),
    ("just", "over"),
    ("or", "take"),
}

_AMPM = {"am": "am", "a.m.": "am", "a.m": "am", "pm": "pm", "p.m.": "pm", "p.m": "pm"}
_TIME_PREPS = {"at", "by", "until", "till", "before", "after", "around", "about"}
_AFTER_BARE_HOUR = {"tonight", "today", "tomorrow", "this", "in", "on", "sharp", "and"}

# "one" is very often a pronoun; these neighbours mark that use.
_ONE_PRONOUN_BEFORE = {
    "the",
    "that",
    "this",
    "which",
    "each",
    "every",
    "any",
    "no",
    "some",
    "another",
    "little",
    "big",
    "other",
    "last",
    "next",
    "new",
    "right",
    "wrong",
    "only",
    "same",
    "a",
}
_ONE_PRONOUN_AFTER = {
    "of",
    "another",
    "moment",
    "sec",
    "second",
    "minute",
    "more",
    "who",
    "that",
    "which",
    "thing",
    "way",
    "day",
    "last",
    "by",
    "you",
    "on",
    "at",
    "time",
}


# Tokenizer

_TOKEN_RE = re.compile(
    r"""
    (?P<iso>\d{4}-\d{2}-\d{2}(?:[T\x20]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?)
    |(?P<clock>\d{1,2}:\d{2}(?::\d{2})?)
    |(?P<code>[A-Za-z]+-?\d[A-Za-z0-9-]*)
    |(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?P<ord>st\b|nd\b|rd\b|th\b)?
    |(?P<ampm>\b[aApP]\.[mM]\.?)
    |(?P<word>[A-Za-z]+(?:['-][A-Za-z]+)*)
    |(?P<sym>[^\w\s])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class _Tok:
    kind: str
    text: str
    low: str
    start: int
    end: int
    num: Decimal | None = None
    ordinal: bool = False
    integral: bool = False


def _tokenize(text: str) -> list[_Tok]:
    toks: list[_Tok] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup or "sym"
        if kind == "ord":
            kind = "num"
        start, end = m.span()
        raw = m.group(0)
        if kind == "num":
            digits = m.group("num")
            toks.append(
                _Tok(
                    "num",
                    raw,
                    raw.lower(),
                    start,
                    end,
                    num=Decimal(digits.replace(",", "")),
                    ordinal=m.group("ord") is not None,
                    integral="." not in digits,
                )
            )
        elif kind == "word" and "-" in raw:
            parts = raw.split("-")
            if all(p.lower() in _NUMBER_WORDS for p in parts):
                pos = start
                for p in parts:
                    toks.append(_Tok("word", p, p.lower(), pos, pos + len(p)))
                    pos += len(p) + 1
            else:
                toks.append(_Tok("word", raw, raw.lower(), start, end))
        else:
            toks.append(_Tok(kind, raw, raw.lower(), start, end))
    return toks


# Value helpers


def _norm(d: Decimal) -> int | Decimal:
    if d == d.to_integral_value():
        return int(d)
    return d.normalize()


def _cents(amount: Decimal) -> int:
    return int((amount * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _clock(hour: int, minute: int, meridiem: str | None) -> tuple[int, tuple[int, ...]] | None:
    if not (0 <= minute <= 59) or not (0 <= hour <= 23):
        return None
    if meridiem is not None and 1 <= hour <= 12:
        base = hour % 12 + (12 if meridiem == "pm" else 0)
        return base * 60 + minute, ()
    if 1 <= hour <= 12:
        other = (hour + 12) % 24 if hour < 12 else 0
        return hour * 60 + minute, (other * 60 + minute,)
    return hour * 60 + minute, ()


def _valid_day(month: int, day: int) -> bool:
    return 1 <= month <= 12 and 1 <= day <= _DAYS_IN_MONTH[month - 1]


def format_value(kind: ClaimKind, value: ClaimValue) -> str:
    """Render a canonical value the way a rewrite instruction should quote it."""
    if kind == "money" and isinstance(value, int):
        sign = "-" if value < 0 else ""
        dollars, cents = divmod(abs(value), 100)
        return f"{sign}${dollars:,}.{cents:02d}"
    if kind == "time" and isinstance(value, int):
        hour, minute = divmod(value % 1440, 60)
        return f"{hour % 12 or 12}:{minute:02d} {'am' if hour < 12 else 'pm'}"
    if kind == "duration" and isinstance(value, int):
        parts: list[str] = []
        rest = value
        for name, size in (("day", 86_400), ("hour", 3600), ("minute", 60), ("second", 1)):
            count, rest = divmod(rest, size)
            if count:
                parts.append(f"{count} {name}{'' if count == 1 else 's'}")
        return " ".join(parts) or "0 seconds"
    if kind == "date" and isinstance(value, tuple):
        return f"{MONTH_NAMES[value[0] - 1]} {value[1]}"
    return str(value)


# Scanner


@dataclass(frozen=True, slots=True)
class _Hit:
    kind: ClaimKind
    value: ClaimValue
    start: int
    end: int
    alternatives: tuple[ClaimValue, ...] = ()
    approximate: bool = False


_Match = tuple[list[_Hit], int]


class _Scanner:
    def __init__(self, text: str) -> None:
        self.text = text
        self.toks = _tokenize(text.replace("’", "'"))
        self.n = len(self.toks)
        lows = {t.low for t in self.toks}
        self.money_context = bool(lows & _MONEY_CONTEXT)

    # token access

    def low(self, i: int) -> str:
        return self.toks[i].low if 0 <= i < self.n else ""

    def word(self, i: int) -> str:
        if 0 <= i < self.n and self.toks[i].kind == "word":
            return self.toks[i].low
        return ""

    def num(self, i: int) -> Decimal | None:
        if 0 <= i < self.n and self.toks[i].kind == "num":
            return self.toks[i].num
        return None

    def int_num(self, i: int) -> int | None:
        if 0 <= i < self.n and self.toks[i].kind == "num" and self.toks[i].integral:
            value = self.toks[i].num
            return int(value) if value is not None else None
        return None

    def hit(
        self,
        kind: ClaimKind,
        value: ClaimValue,
        i: int,
        j: int,
        alternatives: tuple[ClaimValue, ...] = (),
        approximate: bool = False,
    ) -> _Match:
        start, end = self.toks[i].start, self.toks[j - 1].end
        return [_Hit(kind, value, start, end, alternatives, approximate)], j

    # spoken numbers

    def cardinal(self, i: int) -> tuple[int, int] | None:
        """Parse a spoken cardinal up to the billions. Returns (value, next index)."""
        total = 0
        current = 0
        last = ""
        last_scale = 10**12
        j = i
        while j < self.n:
            w = self.word(j)
            if w in ("a", "an"):
                if last == "" and (self.word(j + 1) == "hundred" or self.word(j + 1) in SCALES):
                    current, last = 1, "a"
                    j += 1
                    continue
                break
            if w in UNITS:
                if w == "zero" and last != "":
                    break
                if last not in ("", "tens", "hundred", "scale", "and"):
                    break
                if last == "" and w == "zero":
                    return 0, j + 1
                current += UNITS[w]
                last = "unit"
            elif w in TEENS:
                if last not in ("", "hundred", "scale", "and"):
                    break
                current += TEENS[w]
                last = "teen"
            elif w in TENS:
                if last not in ("", "hundred", "scale", "and"):
                    break
                current += TENS[w]
                last = "tens"
            elif w == "hundred":
                if last == "a":
                    current = 100
                elif last in ("unit", "teen") and current < 100:
                    current *= 100
                else:
                    break
                last = "hundred"
            elif w in SCALES:
                scale = SCALES[w]
                if last not in ("unit", "teen", "tens", "hundred", "a") or scale >= last_scale:
                    break
                total += current * scale
                current = 0
                last_scale = scale
                last = "scale"
            elif w == "and":
                nxt = self.word(j + 1)
                if last in ("hundred", "scale") and (
                    (nxt in UNITS and nxt != "zero") or nxt in TEENS or nxt in TENS
                ):
                    last = "and"
                else:
                    break
            else:
                break
            j += 1
        if last in ("", "a", "and"):
            return None
        return total + current, j

    def quantity(self, i: int) -> tuple[Decimal, int, str] | None:
        """A count in digits or words, with "point five" and "and a half" tails."""
        d = self.num(i)
        if d is not None and not self.toks[i].ordinal:
            j = i + 1
            if self.word(j) in SCALES:
                d *= SCALES[self.word(j)]
                j += 1
            return d, j, "digits"
        if self.word(i) == "half" and self.word(i + 1) in ("a", "an"):
            return Decimal("0.5"), i + 2, "half"
        if self.word(i) in ("a", "an") and self.word(i + 1) not in ("hundred", *SCALES):
            return Decimal(1), i + 1, "article"
        parsed = self.cardinal(i)
        if parsed is None:
            return None
        value, j = parsed
        out = Decimal(value)
        if self.word(j) == "point" and self.word(j + 1) in UNITS:
            k = j + 1
            frac = ""
            while self.word(k) in UNITS:
                frac += str(UNITS[self.word(k)])
                k += 1
            out = Decimal(f"{value}.{frac}")
            j = k
        if (self.word(j), self.word(j + 1), self.word(j + 2)) == ("and", "a", "half"):
            out += Decimal("0.5")
            j += 3
        return out, j, "spoken"

    def digit_groups(self, i: int) -> tuple[str, int]:
        """Spoken digits as in IDs: "four eight two", "forty eight", "double five"."""
        out = ""
        j = i
        while j < self.n:
            w = self.word(j)
            if w in UNITS:
                out += str(UNITS[w])
            elif w == "oh":
                out += "0"
            elif w in TEENS:
                out += str(TEENS[w])
            elif w in TENS:
                nxt = self.word(j + 1)
                if nxt in UNITS and nxt != "zero":
                    out += str(TENS[w] + UNITS[nxt])
                    j += 1
                else:
                    out += str(TENS[w])
            elif w in ("double", "triple") and self.word(j + 1) in UNITS:
                out += str(UNITS[self.word(j + 1)]) * (2 if w == "double" else 3)
                j += 1
            else:
                break
            j += 1
        return out, j

    def meridiem(self, j: int) -> tuple[str | None, int]:
        """Return (am/pm or None, index after any consumed am/pm token)."""
        low = self.low(j)
        if low in _AMPM:
            return _AMPM[low], j + 1
        w1, w2, w3 = self.word(j), self.word(j + 1), self.word(j + 2)
        if w1 == "tonight" or (w1 == "at" and w2 == "night"):
            return "pm", j
        if (w1 == "in" and w2 == "the") or w1 == "this":
            part = w3 if w1 == "in" else w2
            if part == "morning":
                return "am", j
            if part in ("afternoon", "evening"):
                return "pm", j
        return None, j

    def hour_at(self, i: int) -> int | None:
        value = self.int_num(i)
        if value is not None and not self.toks[i].ordinal and 0 <= value <= 23:
            return value
        return _HOUR_WORDS.get(self.word(i))

    # matchers, tried in order at each position

    def m_iso(self, i: int) -> _Match | None:
        tok = self.toks[i]
        if tok.kind != "iso":
            return None
        y, mo, d = int(tok.text[0:4]), int(tok.text[5:7]), int(tok.text[8:10])
        try:
            date(y, mo, d)
        except ValueError:
            return [], i + 1
        hits = [_Hit("date", f"{y:04d}-{mo:02d}-{d:02d}", tok.start, tok.start + 10)]
        if len(tok.text) > 10:
            hour, minute = int(tok.text[11:13]), int(tok.text[14:16])
            if hour <= 23 and minute <= 59:
                hits.append(_Hit("time", hour * 60 + minute, tok.start + 11, tok.end))
        return hits, i + 1

    def m_slash(self, i: int) -> _Match | None:
        a = self.int_num(i)
        if a is None or self.low(i + 1) != "/":
            return None
        b = self.int_num(i + 2)
        if b is None:
            return None
        if (a, b) == (24, 7):
            return [], i + 3
        if not _valid_day(a, b):
            return None
        if self.low(i + 3) == "/" and (y := self.int_num(i + 4)) is not None:
            year = y + 2000 if y < 100 else y
            try:
                date(year, a, b)
            except ValueError:
                return None
            return self.hit("date", f"{year:04d}-{a:02d}-{b:02d}", i, i + 5)
        return self.hit("date", (a, b), i, i + 3)

    def m_identifier(self, i: int) -> _Match | None:
        if self.word(i) not in ID_KEYWORDS:
            return None
        j = i + 1
        skipped = 0
        while skipped < 3 and self.low(j) in _ID_FILLERS:
            j += 1
            skipped += 1
        tok = self.toks[j] if j < self.n else None
        if tok is None:
            return None
        if tok.kind == "code":
            return self.hit("identifier", tok.text.upper(), i, j + 1)
        if tok.kind == "num" and tok.integral and not tok.ordinal:
            digits = tok.text.replace(",", "")
            if len(digits) >= 3:
                return self.hit("identifier", digits, i, j + 1)
            return None
        letters = ""
        k = j
        while k < self.n and len(self.toks[k].text) == 1 and self.toks[k].text.isupper():
            letters += self.toks[k].text
            k += 1
        tail = self.toks[k] if k < self.n else None
        if letters and tail is not None and tail.kind == "num" and tail.integral:
            return self.hit("identifier", letters + tail.text.replace(",", ""), i, k + 1)
        digits, end = self.digit_groups(k)
        if len(digits) >= 3:
            return self.hit("identifier", letters + digits, i, end)
        return None

    def m_code(self, i: int) -> _Match | None:
        tok = self.toks[i]
        if tok.kind != "code":
            return None
        if sum(ch.isdigit() for ch in tok.text) >= 3:
            return self.hit("identifier", tok.text.upper(), i, i + 1)
        return [], i + 1

    def m_money_digits(self, i: int) -> _Match | None:
        if self.low(i) == "$" and (amount := self.num(i + 1)) is not None:
            j = i + 2
            if self.word(j) in SCALES:
                amount *= SCALES[self.word(j)]
                j += 1
            if self.word(j) in DOLLAR_WORDS:
                j += 1
            return self.hit("money", _cents(amount), i, j)
        amount = self.num(i)
        if amount is None or self.toks[i].ordinal:
            return None
        j = i + 1
        if self.word(j) in SCALES and self.word(j + 1) in DOLLAR_WORDS:
            amount *= SCALES[self.word(j)]
            j += 1
        if self.word(j) in DOLLAR_WORDS:
            cents = _cents(amount)
            j += 1
            k = j + 1 if self.word(j) == "and" else j
            extra = self.int_num(k)
            if extra is not None and extra < 100 and self.word(k + 1) in CENT_WORDS:
                cents += extra
                j = k + 2
            return self.hit("money", cents, i, j)
        if self.word(j) in CENT_WORDS and self.toks[i].integral:
            return self.hit("money", int(amount), i, j + 1)
        return None

    def day_at(self, j: int) -> tuple[int, int] | None:
        """A day of month in digits, ordinal words, or small cardinals."""
        value = self.int_num(j)
        if value is not None:
            return (value, j + 1) if 1 <= value <= 31 else None
        w = self.word(j)
        if w in ORDINALS:
            return ORDINALS[w], j + 1
        if w in ("twenty", "thirty") and self.word(j + 1) in ORDINALS:
            day = TENS[w] + ORDINALS[self.word(j + 1)]
            return (day, j + 2) if day <= 31 else None
        parsed = self.cardinal(j)
        if parsed is not None and 1 <= parsed[0] <= 31:
            return parsed
        return None

    def year_at(self, j: int) -> tuple[int, int] | None:
        k = j + 1 if self.low(j) == "," else j
        tok = self.toks[k] if k < self.n else None
        if tok is not None and tok.kind == "num" and tok.integral and len(tok.text) == 4:
            year = int(tok.text)
            if 1900 <= year <= 2199:
                return year, k + 1
        return None

    def date_hit(self, i: int, j: int, month: int, day: int) -> _Match | None:
        if not _valid_day(month, day):
            return None
        year = self.year_at(j)
        if year is not None:
            try:
                date(year[0], month, day)
            except ValueError:
                return None
            return self.hit("date", f"{year[0]:04d}-{month:02d}-{day:02d}", i, year[1])
        return self.hit("date", (month, day), i, j)

    def m_date_words(self, i: int) -> _Match | None:
        w = self.word(i)
        if w in MONTHS:
            j = i + 1
            if self.low(j) == "." and w != "may":
                j += 1
            if self.word(j) == "the":
                j += 1
            day = self.day_at(j)
            if day is None:
                return None
            return self.date_hit(i, day[1], MONTHS[w], day[0])
        tok = self.toks[i]
        is_ordinal_word = w in ORDINALS or (
            w in ("twenty", "thirty") and self.word(i + 1) in ORDINALS
        )
        if not (tok.kind == "num" and tok.integral) and not is_ordinal_word:
            return None
        day = self.day_at(i)
        if day is None:
            return None
        j = day[1] + 1 if self.word(day[1]) == "of" else day[1]
        month_word = self.word(j)
        if month_word not in MONTHS or (month_word == "may" and self.word(day[1]) != "of"):
            return None
        return self.date_hit(i, j + 1, MONTHS[month_word], day[0])

    def m_time(self, i: int) -> _Match | None:
        tok = self.toks[i]
        w = self.word(i)
        if tok.kind == "clock":
            parts = tok.text.split(":")
            mer, j = self.meridiem(i + 1)
            clock = _clock(int(parts[0]), int(parts[1]), mer)
            if clock is None:
                return [], i + 1
            return self.hit("time", clock[0], i, j, clock[1])
        if w in ("noon", "midday"):
            return self.hit("time", 720, i, i + 1)
        if w == "midnight":
            return self.hit("time", 0, i, i + 1)
        k = i + 1 if w == "a" and self.word(i + 1) == "quarter" else i
        if self.word(k) in ("half", "quarter") and self.word(k + 1) in (
            "past",
            "after",
            "to",
            "till",
        ):
            hour = self.hour_at(k + 2)
            if hour is not None and 1 <= hour <= 12:
                offset = 30 if self.word(k) == "half" else 15
                if self.word(k + 1) in ("to", "till"):
                    hour, minute = (hour - 1) or 12, 60 - offset
                else:
                    minute = offset
                mer, j = self.meridiem(k + 3)
                clock = _clock(hour, minute, mer)
                if clock is not None:
                    return self.hit("time", clock[0], i, j, clock[1])
        hour = self.hour_at(i)
        if hour is None or not 1 <= hour <= 12:
            return None
        j = i + 1
        oclock = self.word(j) in ("o'clock", "oclock")
        if oclock:
            j += 1
        mer, after = self.meridiem(j)
        explicit = after > j
        if oclock or explicit:
            clock = _clock(hour, 0, mer)
            if clock is not None:
                return self.hit("time", clock[0], i, after, clock[1])
        nxt = self.toks[j] if j < self.n else None
        bare_ok = (
            nxt is None
            or (nxt.kind == "sym" and nxt.low not in ("$", "%", "/", ":", "-"))
            or nxt.low in _AFTER_BARE_HOUR
        )
        if self.low(i - 1) in _TIME_PREPS and bare_ok:
            clock = _clock(hour, 0, mer)
            if clock is not None:
                return self.hit("time", clock[0], i, j, clock[1])
        return None

    def duration_unit(self, j: int) -> tuple[int, str, int] | None:
        k = j
        while self.word(k) in _DURATION_ADJECTIVES:
            k += 1
        w = self.word(k)
        if w in DURATION_UNITS:
            return DURATION_UNITS[w], w, k + 1
        return None

    def m_duration(self, i: int) -> _Match | None:
        q = self.quantity(i)
        if q is None:
            return None
        amount, j, style = q
        if self.low(j) in ("to", "-", "or"):
            q2 = self.quantity(j + 1)
            if q2 is not None and q2[2] in ("digits", "spoken"):
                unit2 = self.duration_unit(q2[1])
                if unit2 is not None:
                    lo, hi = round(amount * unit2[0]), round(q2[0] * unit2[0])
                    return self.hit("duration", (lo + hi) // 2, i, unit2[2], (lo, hi), True)
        unit = self.duration_unit(j)
        if unit is None:
            return None
        size, name, j = unit
        if amount == 1 and style in ("article", "spoken") and name in _FILLER_UNITS:
            return [], j
        total = amount * size
        if (self.word(j), self.word(j + 1), self.word(j + 2)) == ("and", "a", "half"):
            total += Decimal(size) / 2
            j += 3
        else:
            k = j + 1 if self.word(j) == "and" else j
            q2 = self.quantity(k)
            if q2 is not None and q2[2] != "article":
                unit2 = self.duration_unit(q2[1])
                if unit2 is not None and unit2[0] < size:
                    total += q2[0] * unit2[0]
                    j = unit2[2]
        return self.hit("duration", int(total.to_integral_value(ROUND_HALF_UP)), i, j)

    def m_money_spoken(self, i: int) -> _Match | None:
        parsed = self.cardinal(i)
        if parsed is None:
            return None
        dollars, j = parsed
        if self.word(j) in DOLLAR_WORDS:
            cents = dollars * 100
            j += 1
            k = j + 1 if self.word(j) == "and" else j
            extra = self.cardinal(k)
            if extra is not None and extra[0] < 100:
                cents += extra[0]
                j = extra[1]
                if self.word(j) in CENT_WORDS:
                    j += 1
            return self.hit("money", cents, i, j)
        if self.word(j) in CENT_WORDS:
            return self.hit("money", dollars, i, j + 1)
        if self.money_context and 1 <= dollars <= 99 and self.word(j) in TEENS.keys() | TENS.keys():
            pair = self.cardinal(j)
            if pair is not None and 10 <= pair[0] <= 99 and self.low(pair[1]) not in _AMPM:
                end = pair[1] + 1 if self.word(pair[1]) in DOLLAR_WORDS else pair[1]
                return self.hit("money", dollars * 100 + pair[0], i, end)
        return None

    def m_digit_run(self, i: int) -> _Match | None:
        j = i
        digits = ""
        while self.word(j) in UNITS or (self.word(j) == "oh" and digits):
            digits += "0" if self.word(j) == "oh" else str(UNITS[self.word(j)])
            j += 1
        if len(digits) >= 4:
            return self.hit("identifier", digits, i, j)
        return None

    def m_time_spoken(self, i: int) -> _Match | None:
        hour = _HOUR_WORDS.get(self.word(i))
        if hour is None:
            return None
        w = self.word(i + 1)
        if w == "oh" and self.word(i + 2) in UNITS and self.word(i + 2) != "zero":
            minute, j = UNITS[self.word(i + 2)], i + 3
        elif w in TEENS:
            minute, j = TEENS[w], i + 2
        elif w in TENS and TENS[w] <= 50:
            minute, j = TENS[w], i + 2
            if self.word(j) in UNITS and self.word(j) != "zero":
                minute += UNITS[self.word(j)]
                j += 1
        else:
            return None
        blockers = DOLLAR_WORDS | CENT_WORDS | DURATION_UNITS.keys() | SCALES.keys() | {"hundred"}
        if self.word(j) in blockers:
            return None
        mer, after = self.meridiem(j)
        clock = _clock(hour, minute, mer)
        if clock is None:
            return None
        return self.hit("time", clock[0], i, after, clock[1])

    def m_vague_pair(self, i: int) -> _Match | None:
        q = self.quantity(i)
        if q is None or q[2] == "article" or self.word(q[1]) != "or":
            return None
        q2 = self.quantity(q[1] + 1)
        if q2 is not None and q[0] <= 10 and q2[0] <= 10:
            return [], q2[1]
        return None

    def m_number(self, i: int) -> _Match | None:
        tok = self.toks[i]
        if tok.kind == "num":
            if tok.ordinal:
                return [], i + 1
            q = self.quantity(i)
            assert q is not None
            amount, j, _ = q
            if self.low(j) in ("%", "percent"):
                j += 1
            return self.hit("number", _norm(amount), i, j)
        w = self.word(i)
        if w == "one" and (
            self.word(i - 1) in _ONE_PRONOUN_BEFORE or self.word(i + 1) in _ONE_PRONOUN_AFTER
        ):
            return [], i + 1
        if w in ORDINALS:
            return [], i + 1
        if w in ("a", "an") and self.word(i + 1) not in ("hundred", *SCALES):
            return None
        parsed = self.quantity(i)
        if parsed is None:
            return None
        amount, j, _ = parsed
        if self.word(j) in ORDINALS:
            return [], j + 1
        if self.low(j) in ("%", "percent"):
            j += 1
        return self.hit("number", _norm(amount), i, j)

    # driver

    def hedged(self, i: int, j: int) -> bool:
        before1 = self.low(i - 1)
        before2 = (self.low(i - 2), before1)
        after = (self.low(j), self.low(j + 1))
        return (
            before1 in _HEDGES_BEFORE
            or before2 in _HEDGES_BEFORE_PAIRS
            or after in (("or", "so"), ("give", "or"))
            or self.low(j) == "ish"
            or after == ("-", "ish")
        )

    def run(self) -> list[Claim]:
        matchers = (
            self.m_iso,
            self.m_slash,
            self.m_identifier,
            self.m_code,
            self.m_money_digits,
            self.m_date_words,
            self.m_time,
            self.m_duration,
            self.m_money_spoken,
            self.m_digit_run,
            self.m_time_spoken,
            self.m_vague_pair,
            self.m_number,
        )
        out: list[Claim] = []
        i = 0
        while i < self.n:
            for matcher in matchers:
                found = matcher(i)
                if found is None:
                    continue
                hits, j = found
                approx = self.hedged(i, j)
                for h in hits:
                    hedgeable = h.kind in ("money", "number", "time", "duration")
                    out.append(
                        Claim(
                            kind=h.kind,
                            raw=self.text[h.start : h.end],
                            value=h.value,
                            span=(h.start, h.end),
                            approximate=h.approximate or (approx and hedgeable),
                            alternatives=h.alternatives,
                        )
                    )
                i = max(j, i + 1)
                break
            else:
                i += 1
        return out


def extract_claims(sentence: str) -> list[Claim]:
    """Extract checkable claims from one sentence, in order of appearance."""
    return _Scanner(sentence).run()
