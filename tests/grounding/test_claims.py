from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from switchboard.grounding.claims import Claim, ClaimValue, extract_claims, format_value

# (sentence, [(kind, value, approximate), ...])
Expected = list[tuple[str, ClaimValue, bool]]

CASES: list[tuple[str, Expected]] = [
    # money
    ("Your refund is $12.50.", [("money", 1250, False)]),
    ("That comes to 12.50 dollars.", [("money", 1250, False)]),
    ("The refund is twelve dollars fifty.", [("money", 1250, False)]),
    ("Your refund will be twelve fifty.", [("money", 1250, False)]),
    ("It costs twelve dollars and fifty cents.", [("money", 1250, False)]),
    ("There is a fee of fifty cents.", [("money", 50, False)]),
    ("The total is $1,234.05.", [("money", 123405, False)]),
    ("We issued a credit of $2 million.", [("money", 200_000_000, False)]),
    ("That is a hundred dollars.", [("money", 10000, False)]),
    ("You were charged 12 dollars and 5 cents.", [("money", 1205, False)]),
    ("It is about $18.", [("money", 1800, True)]),
    # numbers
    ("There are twenty three packages.", [("number", 23, False)]),
    ("There are twenty-three packages.", [("number", 23, False)]),
    ("We have a hundred and five drivers.", [("number", 105, False)]),
    (
        "We served one million two hundred thousand and seven customers.",
        [("number", 1_200_007, False)],
    ),
    ("The count is 4,500.", [("number", 4500, False)]),
    ("The rate is 2.5 percent.", [("number", Decimal("2.5"), False)]),
    ("We have 3 drivers nearby.", [("number", 3, False)]),
    ("I counted one.", [("number", 1, False)]),
    # times
    ("The driver arrives at 7:45 pm.", [("time", 1185, False)]),
    ("The driver arrives at seven forty five.", [("time", 465, False)]),
    ("Pickup is at noon.", [("time", 720, False)]),
    ("Pickup is at midnight.", [("time", 0, False)]),
    ("Pickup is at 7 am.", [("time", 420, False)]),
    ("It closes at 19:30.", [("time", 1170, False)]),
    ("We'll call you at seven pm.", [("time", 1140, False)]),
    ("It arrives at half past seven.", [("time", 450, False)]),
    ("It arrives at a quarter to eight p.m.", [("time", 1185, False)]),
    ("The driver comes at seven oh five in the evening.", [("time", 1145, False)]),
    ("The driver should be there around 7:45 pm.", [("time", 1185, True)]),
    ("We close at 9 tonight.", [("time", 1260, False)]),
    # durations
    ("Expect it in 20 minutes.", [("duration", 1200, False)]),
    ("It takes about two hours.", [("duration", 7200, True)]),
    ("It should arrive in around 20 minutes.", [("duration", 1200, True)]),
    ("That will take an hour and a half.", [("duration", 5400, False)]),
    ("Allow 2 hours and 30 minutes.", [("duration", 9000, False)]),
    ("Give it 20-30 minutes.", [("duration", 1500, True)]),
    ("It arrives in twenty minutes or so.", [("duration", 1200, True)]),
    ("It takes 3 days.", [("duration", 259_200, False)]),
    ("Give me two minutes.", [("duration", 120, False)]),
    ("It is roughly forty-five minutes away.", [("duration", 2700, True)]),
    ("That takes one and a half hours.", [("duration", 5400, False)]),
    # dates
    ("Delivery is on October 21st.", [("date", (10, 21), False)]),
    ("Delivery is on Oct 21.", [("date", (10, 21), False)]),
    ("Delivery is on Oct. 21, 2026.", [("date", "2026-10-21", False)]),
    ("Delivery is on 2026-10-21.", [("date", "2026-10-21", False)]),
    ("Delivery is on the 21st of October.", [("date", (10, 21), False)]),
    ("Delivery is on October twenty first.", [("date", (10, 21), False)]),
    ("Delivery is on 10/21.", [("date", (10, 21), False)]),
    ("It arrives May 3rd.", [("date", (5, 3), False)]),
    # identifiers
    ("Your order 48213 has shipped.", [("identifier", "48213", False)]),
    ("Your order number is four eight two one three.", [("identifier", "48213", False)]),
    ("I opened ticket AB-1029 for you.", [("identifier", "AB-1029", False)]),
    ("Your confirmation number is A B one oh two nine.", [("identifier", "AB1029", False)]),
    ("Order #48213 is ready.", [("identifier", "48213", False)]),
    ("Please reference ticket ab-1029.", [("identifier", "AB-1029", False)]),
    ("Your case number is forty eight two thirteen.", [("identifier", "48213", False)]),
    ("The tracking code is ZX90311.", [("identifier", "ZX90311", False)]),
    # mixed
    (
        "Your refund of $18 will arrive in about 20 minutes.",
        [("money", 1800, False), ("duration", 1200, True)],
    ),
    (
        "Order 48213 arrives at 7:45 pm on October 21st.",
        [("identifier", "48213", False), ("time", 1185, False), ("date", (10, 21), False)],
    ),
    # negatives: vague quantities and filler carry no checkable fact
    ("It will take a couple of minutes.", []),
    ("We have a few drivers nearby.", []),
    ("That takes several days.", []),
    ("A few dollars at most.", []),
    ("One moment please.", []),
    ("Just a second while I check.", []),
    ("Give me a minute.", []),
    ("Let me check that for you.", []),
    ("Is this the one you ordered?", []),
    ("Your order is on its way.", []),
    ("May I have your name?", []),
    ("This is your first order with us.", []),
    ("We are open 24/7.", []),
    ("I can try one or two things.", []),
    ("I am happy to help.", []),
    ("Have a nice day.", []),
    ("Some time today, I hope.", []),
    ("Let me order a replacement for you.", []),
    ("It happens on a case by case basis.", []),
    ("", []),
]


def _shape(claims: list[Claim]) -> Expected:
    return [(c.kind, c.value, c.approximate) for c in claims]


@pytest.mark.parametrize(("sentence", "expected"), CASES, ids=[c[0] or "<empty>" for c in CASES])
def test_extraction_table(sentence: str, expected: Expected) -> None:
    assert _shape(extract_claims(sentence)) == expected


def test_table_has_enough_cases() -> None:
    assert len(CASES) >= 40
    assert sum(1 for _, exp in CASES if not exp) >= 15


def test_raw_and_span_point_into_sentence() -> None:
    sentence = "Your refund is $12.50 today."
    (claim,) = extract_claims(sentence)
    assert claim.raw == "$12.50"
    start, end = claim.span
    assert sentence[start:end] == claim.raw


def test_identifier_span_covers_keyword_phrase() -> None:
    sentence = "Your order number is four eight two one three."
    (claim,) = extract_claims(sentence)
    assert claim.raw == "order number is four eight two one three"


def test_time_without_meridiem_carries_pm_alternative() -> None:
    (claim,) = extract_claims("See you at seven forty five.")
    assert claim.value == 465
    assert claim.alternatives == (1185,)


def test_time_with_meridiem_has_no_alternative() -> None:
    (claim,) = extract_claims("See you at 7:45 pm.")
    assert claim.alternatives == ()


def test_twelve_without_meridiem_alternative_is_midnight_reading() -> None:
    (claim,) = extract_claims("Pickup is at 12:30.")
    assert claim.value == 750
    assert claim.alternatives == (30,)


def test_range_duration_keeps_bounds() -> None:
    (claim,) = extract_claims("Give it 20 to 30 minutes.")
    assert claim.value == 1500
    assert claim.approximate
    assert claim.alternatives == (1200, 1800)


def test_claim_is_frozen() -> None:
    (claim,) = extract_claims("$5")
    with pytest.raises(FrozenInstanceError):
        claim.value = 10  # type: ignore[misc]


def test_to_dict_is_json_friendly() -> None:
    (claim,) = extract_claims("Delivery is on October 21st.")
    assert claim.to_dict() == {
        "kind": "date",
        "raw": "October 21st",
        "value": [10, 21],
        "span": [15, 27],
        "approximate": False,
    }
    (number,) = extract_claims("The rate is 2.5 percent.")
    assert number.to_dict()["value"] == "2.5"


def test_curly_apostrophe_oclock() -> None:
    assert _shape(extract_claims("Call back at seven o’clock.")) == [("time", 420, False)]


@pytest.mark.parametrize(
    ("kind", "value", "text"),
    [
        ("money", 1250, "$12.50"),
        ("money", 123405, "$1,234.05"),
        ("time", 1185, "7:45 pm"),
        ("time", 720, "12:00 pm"),
        ("time", 5, "12:05 am"),
        ("duration", 1320, "22 minutes"),
        ("duration", 5400, "1 hour 30 minutes"),
        ("duration", 45, "45 seconds"),
        ("duration", 172_800, "2 days"),
        ("date", (10, 21), "October 21"),
        ("date", "2026-10-21", "2026-10-21"),
        ("number", Decimal("2.5"), "2.5"),
        ("identifier", "AB-1029", "AB-1029"),
    ],
)
def test_format_value(kind: str, value: ClaimValue, text: str) -> None:
    assert format_value(kind, value) == text  # type: ignore[arg-type]
