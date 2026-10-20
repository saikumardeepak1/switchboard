from dataclasses import dataclass

import pytest

from switchboard.grounding.claims import Claim, extract_claims
from switchboard.grounding.evidence import Evidence


def one(sentence: str) -> Claim:
    (claim,) = extract_claims(sentence)
    return claim


TOOL_RESULT = {
    "order": {
        "id": "48213",
        "refund": 12.5,
        "items": [{"sku": "AB-1029", "qty": 3}],
        "status": "Out for delivery, arriving 2026-10-21",
    }
}


@pytest.mark.parametrize(
    "sentence",
    [
        "Your refund is $12.50.",
        "Your refund is twelve dollars fifty.",
        "Your order 48213 has shipped.",
        "Your order number is four eight two one three.",
        "I found item AB-1029.",
        "You ordered 3 of them.",
        "It arrives October 21st.",
        "It arrives on 2026-10-21.",
    ],
)
def test_nested_tool_result_supports(sentence: str) -> None:
    evidence = Evidence(tool_results=[TOOL_RESULT])
    assert evidence.supports(one(sentence)), sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "Your refund is $18.",
        "Your refund is $12.05.",
        "Your order 48231 has shipped.",
        "You ordered 4 of them.",
        "It arrives October 22nd.",
        "It arrives on 2025-10-21.",
    ],
)
def test_nested_tool_result_rejects(sentence: str) -> None:
    evidence = Evidence(tool_results=[TOOL_RESULT])
    assert not evidence.supports(one(sentence)), sentence


def test_strings_from_facts_and_memory_are_parsed() -> None:
    evidence = Evidence(
        facts=["Refund of $12.50 issued on Oct 20"],
        memory=["Customer prefers deliveries after 6 pm"],
    )
    assert evidence.supports(one("The refund was $12.50."))
    assert evidence.supports(one("We deliver after 6 pm."))
    assert evidence.supports(one("It was issued October 20th."))


def test_cents_keys_are_read_as_cents() -> None:
    evidence = Evidence(tool_results=[{"refund_cents": 1250}])
    assert evidence.supports(one("Your refund is $12.50."))


def test_time_exact_unless_approximate() -> None:
    evidence = Evidence(tool_results=[{"eta": "7:47 pm"}])
    assert not evidence.supports(one("The driver arrives at 7:45 pm."))
    assert evidence.supports(one("The driver arrives around 7:45 pm."))
    assert not evidence.supports(one("The driver arrives around 7:55 pm."))


def test_time_without_meridiem_matches_either_half_of_day() -> None:
    evidence = Evidence(tool_results=[{"eta": "19:45"}])
    assert evidence.supports(one("The driver arrives at seven forty five."))


def test_duration_from_unit_named_key() -> None:
    evidence = Evidence(tool_results=[{"eta_minutes": 22}])
    assert evidence.supports(one("It arrives in 22 minutes."))
    assert not evidence.supports(one("It arrives in 20 minutes."))
    assert evidence.supports(one("It arrives in about 20 minutes."))
    assert not evidence.supports(one("It arrives in about 30 minutes."))


def test_duration_range_supported_inside_bounds() -> None:
    evidence = Evidence(tool_results=[{"wait_minutes": 28}])
    assert evidence.supports(one("Give it 20 to 30 minutes."))
    assert not evidence.supports(one("Give it 5 to 10 minutes."))


def test_eta_minus_now_supports_duration() -> None:
    evidence = Evidence(tool_results=[{"eta": "7:52 pm"}], now_minutes=19 * 60 + 30)
    assert evidence.supports(one("It arrives in 22 minutes."))
    assert evidence.supports(one("It arrives in about 20 minutes."))
    assert not evidence.supports(one("It arrives in 20 minutes."))


def test_eta_iso_timestamp_and_now() -> None:
    evidence = Evidence(tool_results=[{"eta": "2026-10-20T19:52:00Z"}], now_minutes=19 * 60 + 30)
    assert evidence.supports(one("It arrives in 22 minutes."))
    assert evidence.supports(one("It arrives at 7:52 pm."))


def test_eta_crossing_midnight() -> None:
    evidence = Evidence(tool_results=[{"eta": "00:10"}], now_minutes=23 * 60 + 50)
    assert evidence.supports(one("It arrives in 20 minutes."))


def test_without_now_no_duration_is_derived() -> None:
    evidence = Evidence(tool_results=[{"eta": "7:52 pm"}])
    assert not evidence.supports(one("It arrives in 22 minutes."))


def test_year_less_date_claim_matches_dated_evidence_but_not_reverse() -> None:
    assert Evidence(facts=["2026-10-21"]).supports(one("It is due October 21st."))
    assert not Evidence(facts=["October 21"]).supports(one("It is due Oct 21, 2026."))


def test_identifier_ignores_case_and_separators() -> None:
    evidence = Evidence(tool_results=[{"ticket": "ab 1029"}])
    assert evidence.supports(one("I opened ticket AB-1029."))


def test_integer_ids_in_tool_results() -> None:
    evidence = Evidence(tool_results=[{"order_id": 48213}])
    assert evidence.supports(one("Your order 48213 has shipped."))


def test_dataclass_and_model_dump_objects_are_walked() -> None:
    @dataclass
    class Fact:
        subject: str
        value: str

    class Model:
        def model_dump(self) -> dict[str, object]:
            return {"refund": "12.50"}

    evidence = Evidence(facts=[Fact("refund", "$7.25"), Model()])
    assert evidence.supports(one("That is $7.25."))
    assert evidence.supports(one("That is $12.50."))


def test_add_extends_evidence() -> None:
    evidence = Evidence()
    claim = one("Your refund is $12.50.")
    assert not evidence.supports(claim)
    evidence.add({"refund": "$12.50"})
    assert evidence.supports(claim)


def test_nearest_value_of_same_kind() -> None:
    evidence = Evidence(tool_results=[{"refund": 12.5, "fee": 40.0, "eta": "7:52 pm"}])
    assert evidence.nearest(one("Your refund is $18.")) == 1250
    assert evidence.nearest(one("I see order 99999.")) is None


def test_words_bag_for_scorers() -> None:
    evidence = Evidence(tool_results=[{"status": "Out for delivery"}], memory=["Prefers email"])
    assert {"delivery", "email", "status"} <= evidence.words


TYPED = {"tool": "lookup_order", "eta_minutes": 20, "items": 3, "order_id": "48213"}


@pytest.mark.parametrize(
    "sentence",
    [
        "I've refunded $20 to your card.",
        "I've issued a refund of $3.",
        "Your refund of 48,213 dollars is on its way.",
    ],
)
def test_plain_numbers_and_ids_are_not_money_or_references(sentence: str) -> None:
    from switchboard.grounding import GroundingGate

    decision = GroundingGate().review(sentence, Evidence(tool_results=[TYPED]))
    assert decision.decision == "block", sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "Your order 48213 is on the way.",
        "It has 3 items.",
        "It arrives in 20 minutes.",
    ],
)
def test_typed_values_still_match_their_own_kind(sentence: str) -> None:
    from switchboard.grounding import GroundingGate

    decision = GroundingGate().review(sentence, Evidence(tool_results=[TYPED]))
    assert decision.decision == "release", sentence


@pytest.mark.parametrize(
    ("value", "sentence"),
    [
        ({"amount": 20}, "I refunded $20."),
        ({"refund_total": 12.5}, "Your refund is $12.50."),
        ({"fee_cents": 499}, "The fee is $4.99."),
        ({"pay_dollars": 182}, "You earned $182."),
        ({"note": "Refunded $15.00 on Monday"}, "We refunded $15."),
        ({"week_total": 182.4}, "You've earned $182.40."),
    ],
)
def test_money_comes_from_money_keys_and_currency_text(value: object, sentence: str) -> None:
    assert Evidence(tool_results=[value]).supports(one(sentence)), sentence
