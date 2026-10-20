"""Round trips: render a canonical value in several spoken and digit styles, extract it back.

The spoken formatter here is written independently of the extractor on purpose, so a shared bug
cannot make both sides agree.
"""

from hypothesis import given
from hypothesis import strategies as st

from switchboard.grounding.claims import Claim, extract_claims

ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _below_hundred(n: int, hyphen: bool) -> str:
    if n < 20:
        return ONES[n]
    tens, unit = divmod(n, 10)
    if unit == 0:
        return TENS[tens]
    return f"{TENS[tens]}{'-' if hyphen else ' '}{ONES[unit]}"


def _below_thousand(n: int, use_and: bool, hyphen: bool, use_a: bool) -> str:
    hundreds, rest = divmod(n, 100)
    parts: list[str] = []
    if hundreds:
        lead = "a" if use_a and hundreds == 1 else ONES[hundreds]
        parts.append(f"{lead} hundred")
        if rest and use_and:
            parts.append("and")
    if rest or not hundreds:
        parts.append(_below_hundred(rest, hyphen))
    return " ".join(parts)


def spoken(n: int, use_and: bool = False, hyphen: bool = False, use_a: bool = False) -> str:
    if n == 0:
        return "zero"
    parts: list[str] = []
    millions, rest = divmod(n, 1_000_000)
    thousands, units = divmod(rest, 1000)
    if millions:
        parts.append(f"{_below_thousand(millions, use_and, hyphen, False)} million")
    if thousands:
        parts.append(f"{_below_thousand(thousands, use_and, hyphen, False)} thousand")
    if units:
        if parts and units < 100 and use_and:
            parts.append("and")
        lead_a = use_a and not parts
        parts.append(_below_thousand(units, use_and, hyphen, lead_a))
    return " ".join(parts)


def _only(claims: list[Claim], kind: str) -> Claim:
    assert len(claims) == 1, claims
    assert claims[0].kind == kind, claims
    return claims[0]


@given(
    n=st.integers(min_value=2, max_value=999_999_999),
    style=st.sampled_from(["digits", "commas", "spoken", "spoken_and", "hyphen", "spoken_a"]),
)
def test_number_round_trip(n: int, style: str) -> None:
    text = {
        "digits": str(n),
        "commas": f"{n:,}",
        "spoken": spoken(n),
        "spoken_and": spoken(n, use_and=True),
        "hyphen": spoken(n, hyphen=True),
        "spoken_a": spoken(n, use_and=True, use_a=True),
    }[style]
    claim = _only(extract_claims(f"We counted {text} boxes."), "number")
    assert claim.value == n


def _dollar_word(d: int) -> str:
    return "dollar" if d == 1 else "dollars"


def _cent_word(c: int) -> str:
    return "cent" if c == 1 else "cents"


@given(
    dollars=st.integers(min_value=0, max_value=999_999),
    cents=st.integers(min_value=0, max_value=99),
    style=st.sampled_from(
        ["symbol", "symbol_commas", "suffix", "spoken_and", "spoken_bare", "spoken_pair"]
    ),
)
def test_money_round_trip(dollars: int, cents: int, style: str) -> None:
    if style.startswith("spoken") and dollars == 0:
        dollars = 7
    if style == "spoken_pair":
        # "twelve fifty" only reads as money for short amounts with real cents.
        dollars = max(1, dollars % 100)
        cents = max(10, cents)
    total = dollars * 100 + cents
    if style == "symbol":
        text = f"${dollars}.{cents:02d}"
    elif style == "symbol_commas":
        text = f"${dollars:,}.{cents:02d}"
    elif style == "suffix":
        text = f"{dollars}.{cents:02d} dollars"
    elif style == "spoken_and":
        text = f"{spoken(dollars)} {_dollar_word(dollars)}"
        if cents:
            text += f" and {spoken(cents)} {_cent_word(cents)}"
    elif style == "spoken_bare":
        text = f"{spoken(dollars)} {_dollar_word(dollars)}"
        if cents:
            text += f" {spoken(cents)}"
    else:
        text = f"{spoken(dollars)} {spoken(cents)}"
    claim = _only(extract_claims(f"Your refund is {text}."), "money")
    assert claim.value == total


@given(
    minutes=st.integers(min_value=0, max_value=1439),
    style=st.sampled_from(["twelve_hour", "twenty_four", "spoken"]),
)
def test_time_round_trip(minutes: int, style: str) -> None:
    hour, minute = divmod(minutes, 60)
    meridiem = "am" if hour < 12 else "pm"
    hour12 = hour % 12 or 12
    if style == "twelve_hour":
        text = f"{hour12}:{minute:02d} {meridiem}"
    elif style == "twenty_four":
        text = f"{hour:02d}:{minute:02d}"
    else:
        if minute == 0:
            tail = ""
        elif minute < 10:
            tail = f" oh {ONES[minute]}"
        else:
            tail = f" {_below_hundred(minute, hyphen=False)}"
        text = f"{ONES[hour12]}{tail} {meridiem}"
    claim = _only(extract_claims(f"The driver arrives at {text}."), "time")
    assert claim.value == minutes


@given(
    n=st.integers(min_value=2, max_value=600),
    unit=st.sampled_from([("minutes", 60), ("hours", 3600), ("seconds", 1)]),
    style=st.sampled_from(["digits", "spoken", "hyphen"]),
)
def test_duration_round_trip(n: int, unit: tuple[str, int], style: str) -> None:
    name, scale = unit
    text = {"digits": str(n), "spoken": spoken(n), "hyphen": spoken(n, hyphen=True)}[style]
    claim = _only(extract_claims(f"It will be ready in {text} {name}."), "duration")
    assert claim.value == n * scale
    assert not claim.approximate


@given(n=st.integers(min_value=2, max_value=600))
def test_hedged_duration_is_approximate(n: int) -> None:
    claim = _only(extract_claims(f"It will be ready in about {spoken(n)} minutes."), "duration")
    assert claim.value == n * 60
    assert claim.approximate


@given(digits=st.text(alphabet="0123456789", min_size=3, max_size=10))
def test_spoken_identifier_round_trip(digits: str) -> None:
    text = " ".join(ONES[int(d)] for d in digits)
    claim = _only(extract_claims(f"Your order number is {text}."), "identifier")
    assert claim.value == digits
