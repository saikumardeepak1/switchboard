"""The runtime gives grounding evidence the conversation clock's time of day (ADR-005)."""

from __future__ import annotations

from switchboard.grounding import Evidence, GroundingGate
from switchboard.knowledge import parse_timestamp

from .harness import CALL_START, make_world

# The harness clock reads Unix time and the call starts at 11:30 in UTC-7.
LOCAL_MIDNIGHT = parse_timestamp("2026-10-20T00:00:00-07:00")
RELATIVE_ETA = "Your order should arrive in about 20 minutes."


async def test_evidence_carries_the_time_of_day_when_the_runtime_knows_it() -> None:
    world = make_world()
    world.runtime = world.new_runtime(day_start=LOCAL_MIDNIGHT)
    conv = await world.start()

    assert conv.grounding_evidence().now_minutes == 11 * 60 + 30
    await world.clock.advance(5 * 60 + 59)
    assert conv.grounding_evidence().now_minutes == 11 * 60 + 35


async def test_day_start_is_any_local_midnight_not_only_today() -> None:
    world = make_world()
    world.runtime = world.new_runtime(day_start=LOCAL_MIDNIGHT - 30 * 86_400)
    conv = await world.start()

    assert conv.grounding_evidence().now_minutes == 11 * 60 + 30


async def test_relative_eta_against_a_clock_time_releases_and_a_wrong_one_does_not() -> None:
    world = make_world()
    world.runtime = world.new_runtime(day_start=LOCAL_MIDNIGHT)
    conv = await world.start()
    eta = Evidence(tool_results=[{"eta": "11:50 AM"}])
    evidence = conv.grounding_evidence().union(eta)

    gate = GroundingGate()
    assert gate.review(RELATIVE_ETA, evidence).decision == "release"
    assert gate.review("It should arrive in 45 minutes.", evidence).decision == "block"


async def test_without_a_day_start_no_time_of_day_is_assumed() -> None:
    world = make_world()
    conv = await world.start()

    assert world.runtime.day_start is None
    assert conv.grounding_evidence().now_minutes is None
    assert world.clock.now() == CALL_START
