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
