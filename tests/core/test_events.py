import pydantic
import pytest

from switchboard.core.events import (
    EVENT_TYPES,
    ConversationStarted,
    Event,
    TurnCommitted,
    parse_event,
)
from tests.core.samples import base, every_event_type

CATALOG = {
    "conversation.started",
    "leg.opened",
    "leg.closed",
    "turn.speculation_started",
    "turn.speculation_cancelled",
    "turn.committed",
    "brain.routed",
    "tool.proposed",
    "policy.verdict",
    "tool.executed",
    "tool.compensated",
    "speech.sentence_gated",
    "speech.played",
    "procedure.advanced",
    "child.result",
    "knowledge.retrieved",
    "handoff.requested",
    "conversation.ended",
}


def test_catalog_is_complete() -> None:
    assert set(EVENT_TYPES) == CATALOG
    assert {e.type for e in every_event_type()} == CATALOG


def test_events_are_frozen() -> None:
    event = ConversationStarted(**base(), tenant_id="t", customer_id="c")
    assert isinstance(event, Event)
    with pytest.raises(pydantic.ValidationError):
        event.tenant_id = "other"  # type: ignore[misc]


def test_new_event_has_unassigned_seq_and_id() -> None:
    event = TurnCommitted(**base(), turn_id="t", text="hi", endpointing_ms=10)
    assert event.seq == 0
    assert event.event_id == ""
    assert event.type == "turn.committed"


def test_extra_fields_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        ConversationStarted(**base(), tenant_id="t", customer_id="c", bogus=1)  # type: ignore[call-arg]


@pytest.mark.parametrize("event", every_event_type(), ids=lambda e: e.type)
def test_parse_event_round_trips_every_type(event: Event) -> None:
    data = event.model_dump(mode="json")
    parsed = parse_event(data)
    assert type(parsed) is type(event)
    assert parsed == event


def test_parse_event_rejects_unknown_type() -> None:
    with pytest.raises(pydantic.ValidationError):
        parse_event({**base(), "type": "nope.nope"})


def test_parse_event_rejects_bad_literal() -> None:
    data = {
        **base(),
        "type": "brain.routed",
        "turn_id": "t",
        "model_tier": "giant",
        "reason": "x",
    }
    with pytest.raises(pydantic.ValidationError):
        parse_event(data)
