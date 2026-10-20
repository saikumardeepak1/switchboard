import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from switchboard.core.events import (
    HandoffRequested,
    PolicyVerdict,
    SpeechSentenceGated,
    ToolCompensated,
    ToolExecuted,
    TurnCommitted,
)
from switchboard.core.store import InMemoryEventStore
from switchboard.telemetry.activity import (
    ActivityFilter,
    ActivityStream,
    InvalidFilter,
    describe,
    resolve_types,
    severity,
)
from tests.telemetry.fixtures import b, conv_a_events, conv_b_events, seeded_store


async def stream() -> ActivityStream:
    return ActivityStream(await seeded_store(), capture_content=True)


ALL = len(conv_a_events()) + len(conv_b_events())


async def test_default_query_is_newest_first() -> None:
    page = await (await stream()).query(ActivityFilter(limit=500))
    assert len(page.items) == ALL
    keys = [(i.at, i.conversation_id, i.seq) for i in page.items]
    assert keys == sorted(keys, reverse=True)
    assert page.next_cursor is None


async def test_ascending_order() -> None:
    page = await (await stream()).query(ActivityFilter(limit=500, order="asc"))
    keys = [(i.at, i.conversation_id, i.seq) for i in page.items]
    assert keys == sorted(keys)


@pytest.mark.parametrize("order", ["asc", "desc"])
async def test_pagination_with_ties_is_complete_and_stable(order: str) -> None:
    s = await stream()
    full = (await s.query(ActivityFilter(limit=500, order=order))).items
    # conv_b has three events at t=1.0 and conv_a has one: ties across and within conversations.
    seen = []
    cursor = None
    while True:
        page = await s.query(ActivityFilter(limit=2, order=order, cursor=cursor))
        seen.extend(page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert [i.event_id for i in seen] == [i.event_id for i in full]


@settings(max_examples=25, deadline=None)
@given(limit=st.integers(min_value=1, max_value=12), order=st.sampled_from(["asc", "desc"]))
async def test_any_page_size_reproduces_the_full_listing(limit: int, order: str) -> None:
    s = await stream()
    full = [i.event_id for i in (await s.query(ActivityFilter(limit=500, order=order))).items]
    seen: list[str] = []
    cursor = None
    for _ in range(100):
        page = await s.query(ActivityFilter(limit=limit, order=order, cursor=cursor))
        seen.extend(i.event_id for i in page.items)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert seen == full


async def test_cursor_survives_new_appends() -> None:
    store = await seeded_store()
    s = ActivityStream(store)
    first = await s.query(ActivityFilter(limit=3, order="asc"))
    await store.append(
        "conv_c",
        [TurnCommitted(**b("conv_c", 99.0), turn_id="z", text="late", endpointing_ms=1)],
        expected_seq=0,
    )
    rest = await s.query(ActivityFilter(limit=500, order="asc", cursor=first.next_cursor))
    ids = [i.event_id for i in first.items + rest.items]
    assert len(ids) == len(set(ids)) == ALL + 1


async def test_bad_cursor_is_rejected() -> None:
    with pytest.raises(InvalidFilter, match="cursor"):
        await (await stream()).query(ActivityFilter(cursor="not-a-cursor"))


def test_type_prefix_resolution() -> None:
    assert resolve_types(["policy."]) == {"policy.verdict"}
    assert resolve_types(["tool.*"]) == {
        "tool.proposed",
        "tool.executed",
        "tool.compensated",
        "tool.unresolved",
    }
    assert resolve_types(["speech.played", "handoff."]) == {"speech.played", "handoff.requested"}
    assert resolve_types([]) is None
    with pytest.raises(InvalidFilter, match="nope"):
        resolve_types(["nope."])
    with pytest.raises(InvalidFilter):
        resolve_types(["policy"])


async def test_type_prefix_filter() -> None:
    page = await (await stream()).query(ActivityFilter(types=("policy.",), limit=500))
    assert {i.type for i in page.items} == {"policy.verdict"}
    assert len(page.items) == 3


async def test_conversation_leg_time_and_text_filters() -> None:
    s = await stream()
    conv = await s.query(ActivityFilter(conversation_id="conv_b", limit=500))
    assert {i.conversation_id for i in conv.items} == {"conv_b"}
    leg = await s.query(ActivityFilter(leg_id="leg_k", limit=500))
    assert {i.leg_id for i in leg.items} == {"leg_k"}
    assert len(leg.items) == 7
    window = await s.query(ActivityFilter(since=5.0, until=5.3, limit=500))
    assert sorted(i.at for i in window.items) == [5.0, 5.02, 5.1, 5.101]
    text = await s.query(ActivityFilter(text="REFUND_LIMIT", limit=500))
    assert [i.type for i in text.items] == ["policy.verdict"]


async def test_since_after_until_is_rejected() -> None:
    with pytest.raises(InvalidFilter, match="since"):
        await (await stream()).query(ActivityFilter(since=5, until=1))


async def test_limit_bounds() -> None:
    with pytest.raises(InvalidFilter, match="limit"):
        await (await stream()).query(ActivityFilter(limit=0))


def test_severity_derivation() -> None:
    denied = PolicyVerdict(**b("c", 1), call_id="x", allowed=False, rule_ids=["r"])
    allowed = PolicyVerdict(**b("c", 1), call_id="x", allowed=True, rule_ids=[])
    blocked = SpeechSentenceGated(
        **b("c", 1), utterance_id="u", sentence="s", decision="block", tier=1, claims=[]
    )
    released = blocked.model_copy(update={"decision": "release"})
    failed = ToolExecuted(**b("c", 1), call_id="x", tool="t", result={"error": "boom"})
    ok = ToolExecuted(**b("c", 1), call_id="x", tool="t", result={"error": None})
    handoff = HandoffRequested(**b("c", 1), reason="r")
    comp = ToolCompensated(**b("c", 1), call_id="x", compensation_tool="undo", result={})
    assert severity(denied) == "warn"
    assert severity(allowed) == "info"
    assert severity(blocked) == "warn"
    assert severity(released) == "info"
    assert severity(failed) == "error"
    assert severity(ok) == "info"
    assert severity(handoff) == "warn"
    assert severity(comp) == "warn"


def test_every_event_type_has_a_one_line_summary() -> None:
    for event in conv_a_events() + conv_b_events():
        text = describe(event)
        assert text
        assert "\n" not in text
        assert len(text) <= 160
    denied = conv_a_events()[19]
    assert describe(denied) == "Denied call_2 by refund_limit: Refunds above $25 need a human."


async def test_item_view_model() -> None:
    page = await (await stream()).query(ActivityFilter(types=("tool.executed",), limit=500))
    failed = next(i for i in page.items if i.leg_id == "leg_k")
    assert failed.severity == "error"
    assert "mark_delivered" in failed.summary
    assert failed.payload == {
        "call_id": "call_3",
        "tool": "mark_delivered",
        "result": {"error": "timeout"},
        "idempotency_key": None,
        "replayed": False,
    }
    d = failed.to_dict()
    assert d["conversation_id"] == "conv_a"
    assert d["type"] == "tool.executed"


async def test_histogram_buckets() -> None:
    s = await stream()
    buckets = await s.histogram(ActivityFilter(conversation_id="conv_a"), bucket_s=2.0)
    assert [x.start for x in buckets] == [0.0, 2.0, 4.0, 6.0]
    assert [x.total for x in buckets] == [9, 6, 9, 2]
    assert buckets[0].counts_by_type["turn.committed"] == 1
    assert sum(x.total for x in buckets) == len(conv_a_events())
    assert buckets[1].end == 4.0


async def test_histogram_fills_empty_buckets_and_respects_filters() -> None:
    s = await stream()
    buckets = await s.histogram(
        ActivityFilter(conversation_id="conv_a", types=("policy.",)), bucket_s=1.0
    )
    assert [x.start for x in buckets] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [x.total for x in buckets] == [1, 0, 1, 0, 1]
    assert buckets[1].counts_by_type == {}


async def test_histogram_uses_explicit_range_and_rejects_bad_buckets() -> None:
    s = await stream()
    buckets = await s.histogram(ActivityFilter(since=0.0, until=10.0), bucket_s=5.0)
    assert [x.start for x in buckets] == [0.0, 5.0]
    assert await s.histogram(ActivityFilter(conversation_id="none"), bucket_s=1) == []
    with pytest.raises(InvalidFilter, match="bucket"):
        await s.histogram(ActivityFilter(), bucket_s=0)
    with pytest.raises(InvalidFilter, match="buckets"):
        await s.histogram(ActivityFilter(since=0, until=1e9), bucket_s=0.001)


async def test_summary() -> None:
    s = await stream()
    summary = await s.summary("conv_a")
    assert summary is not None
    assert summary.tenant_id == "acme"
    assert summary.customer_id == "cust_9"
    assert summary.bundle_id == "bundle_v1"
    assert summary.outcome == "handoff"
    assert summary.started_at == 0.0
    assert summary.ended_at == 6.1
    assert [(p.leg_id, p.party, p.role) for p in summary.parties] == [
        ("leg_c", "customer", "inbound"),
        ("leg_k", "courier", "outbound_child"),
    ]
    assert summary.parties[1].close_reason == "completed"
    assert summary.parties[1].parent_leg_id == "leg_c"
    assert summary.turns == 3
    assert summary.violations_blocked == 1
    assert summary.sentences_blocked == 1
    assert summary.tools_executed == 2
    assert summary.tool_errors == 1
    assert summary.handoffs == 1
    assert summary.voice_to_voice_p50_ms == pytest.approx(1000)
    assert summary.voice_to_voice_p95_ms == pytest.approx(1135)
    assert summary.to_dict()["parties"][0]["party"] == "customer"


async def test_summary_of_unknown_conversation_is_none() -> None:
    assert await (await stream()).summary("nope") is None


async def test_summary_works_on_empty_store() -> None:
    s = ActivityStream(InMemoryEventStore())
    assert await s.summary("x") is None
    assert (await s.query(ActivityFilter())).items == []
