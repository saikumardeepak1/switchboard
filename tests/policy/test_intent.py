"""Which committed user turns open a new request for a write tool (ADR-009 revision)."""

from __future__ import annotations

import pytest

from switchboard.policy import RequestDetector

TOOLS = ["mark_delivered", "issue_refund", "update_address"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I can't mark order 48213 delivered, the customer wants it at home.", ["mark_delivered"]),
        ("Please refund the customer ten dollars.", ["issue_refund"]),
        ("Can you refund another ten dollars?", ["issue_refund"]),
        ("Mark it delivered and refund them too.", ["issue_refund", "mark_delivered"]),
        ("Thanks, is it marked now?", []),
        ("Was it delivered?", []),
        ("Did the refund go through?", []),
        ("Okay, thanks.", []),
        ("What is my pay this week?", []),
    ],
)
def test_default_cues_come_from_the_tool_name(text: str, expected: list[str]) -> None:
    assert RequestDetector().requested(text, TOOLS) == expected


def test_a_bundle_can_name_its_own_cues() -> None:
    detector = RequestDetector({"issue_refund": [["money", "back"]]})
    assert detector.requested("I want my money back.", TOOLS) == ["issue_refund"]
    assert detector.requested("Please refund me.", TOOLS) == []


ACKNOWLEDGEMENTS = [
    "Thanks for the refund, bye.",
    "Great, the refund came through, thanks",
    "Okay so it's marked delivered then, cool.",
    "Okay great, so it's marked delivered now, thanks.",
    "Yes, I asked the courier to mark it delivered at my home.",
    "Thank you for marking it delivered.",
    "Cool, the refund showed up.",
    "Perfect, it says delivered now.",
    "I see it's marked delivered.",
    "The refund was issued yesterday.",
    "You already refunded me, thanks.",
    "Alright, so the refund is done.",
    "Got it, it was marked delivered.",
    "Awesome, thanks for the refund.",
    "Oh nice, the address was updated.",
    "Did you mark it delivered?",
    "Has the refund been issued?",
    "So it's marked delivered?",
    "Is the refund on its way?",
    "Don't refund it yet.",
    "Please don't mark it delivered.",
    "I need to know if it's marked delivered.",
    "I want to check whether the refund went through.",
    "Mm-hm, refunded, okay.",
    "Yeah the customer got the refund.",
    "It's been marked delivered, right?",
]

REQUESTS = [
    ("Please refund the customer.", ["issue_refund"]),
    ("Refund the customer ten dollars.", ["issue_refund"]),
    ("Can you refund the customer?", ["issue_refund"]),
    ("Could you mark it delivered?", ["mark_delivered"]),
    ("Would you please mark order 48213 delivered?", ["mark_delivered"]),
    ("I need it marked delivered.", ["mark_delivered"]),
    ("I want a refund for this order.", ["issue_refund"]),
    ("I'd like a refund.", ["issue_refund"]),
    ("Okay, please mark it delivered.", ["mark_delivered"]),
    ("Thanks, now refund another ten dollars.", ["issue_refund"]),
    ("Great, and please update the address too.", ["update_address"]),
    ("Mark it delivered.", ["mark_delivered"]),
    ("I can't mark order 48213 delivered, the customer wants it at home.", ["mark_delivered"]),
    ("Go ahead and refund them.", ["issue_refund"]),
    ("Yes, please, refund it.", ["issue_refund"]),
    ("Can you update the address to 12 Elm?", ["update_address"]),
]


@pytest.mark.parametrize("text", ACKNOWLEDGEMENTS)
def test_acknowledgements_and_status_talk_open_no_request(text: str) -> None:
    assert RequestDetector().requested(text, TOOLS) == []


@pytest.mark.parametrize(("text", "expected"), REQUESTS)
def test_requests_need_an_imperative_or_request_cue(text: str, expected: list[str]) -> None:
    assert RequestDetector().requested(text, TOOLS) == expected
