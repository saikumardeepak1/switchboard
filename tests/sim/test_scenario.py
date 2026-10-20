"""Full simulated calls through the public runtime API, scored by assertions."""

from __future__ import annotations

import dataclasses

from switchboard.knowledge import parse_timestamp
from switchboard.sim import (
    PERSONAS,
    AdversaryConfig,
    Persona,
    ScenarioResult,
    SimConfig,
    build_world,
    run_matrix,
    run_scenario,
)
from switchboard.sim.delivery import declined_delivery, happy_delivery, refund_delivery


def failed(result: ScenarioResult) -> list[str]:
    return [f"{r.name}: {r.detail}" for r in result.assertion_results if not r.passed]


def test_happy_delivery_passes_every_assertion() -> None:
    result = run_scenario(happy_delivery(), seed=1)
    assert result.passed, failed(result)
    assert result.outcome == "resolved"
    assert result.callers["courier"].status == "hung_up"
    assert result.callers["customer"].status == "hung_up"
    types = [e.type for e in result.events]
    assert types[0] == "conversation.started" and types[-1] == "conversation.ended"
    assert result.timeline.turns, "per-turn latency comes from the same events"
    assert result.cost.total > 0 and result.cost.minutes > 0
    assert result.voice_to_voice_ms and all(v > 0 for v in result.voice_to_voice_ms)
    assert result.failures == []


def test_sim_runtime_knows_the_local_time_of_day_of_its_clock() -> None:
    world = build_world(happy_delivery(), seed=1)
    try:
        assert world.runtime.day_start == parse_timestamp("2026-10-20T00:00:00-07:00")
        assert world.runtime.minute_of_day(world.clock.now()) == 11 * 60 + 30
    finally:
        world.close()
    assert SimConfig(day_start=None).day_start is None


def test_same_seed_gives_identical_events_and_another_seed_does_not() -> None:
    def log(seed: int) -> list[str]:
        return [e.model_dump_json() for e in run_scenario(happy_delivery(), seed=seed).events]

    assert log(3) == log(3)
    assert log(3) != log(4)


def test_declined_delivery_never_marks_the_order() -> None:
    result = run_scenario(declined_delivery(), seed=2)
    assert result.passed, failed(result)
    assert result.outcome == "unresolved"


def test_refund_flow_with_an_adversary_logs_faults_and_blocks_violations() -> None:
    scenario = dataclasses.replace(
        refund_delivery(), adversary=AdversaryConfig(p_violation=0.5, p_corrupt=0.3)
    )
    seen_faults = 0
    for seed in range(4):
        result = run_scenario(scenario, seed=seed)
        seen_faults += len(result.faults)
        names = {r.name for r in result.assertion_results}
        assert "no_corrupted_sentence_released" in names
        executed = {getattr(e, "call_id", "") for e in result.events if e.type == "tool.executed"}
        for fault in result.faults:
            if fault.kind in ("refund_over_limit", "early_mark_delivered", "second_refund"):
                assert fault.call_id not in executed, fault
    assert seen_faults > 0


def test_personas_apply_to_every_human_on_the_call() -> None:
    slow = run_scenario(happy_delivery(), seed=1, persona=PERSONAS["slow_speaker"])
    fast = run_scenario(happy_delivery(), seed=1)
    assert slow.persona == "slow_speaker" and fast.persona == "default"
    assert slow.cost.minutes > fast.cost.minutes


def test_run_matrix_groups_trials_by_scenario_and_persona() -> None:
    matrix = run_matrix(
        [happy_delivery(), declined_delivery()], seeds=[1, 2], personas=[None, Persona("p")]
    )
    assert len(matrix.results) == 8
    assert set(matrix.tasks()) == {
        ("happy_delivery", "default"),
        ("happy_delivery", "p"),
        ("declined_delivery", "default"),
        ("declined_delivery", "p"),
    }
    assert 0.0 <= matrix.pass_hat_k(2) <= matrix.pass_hat_k(1) <= 1.0


def test_script_reacts_to_a_child_result_that_is_not_the_last_message() -> None:
    from switchboard.brain import BrainRequest, Message
    from switchboard.sim.delivery import DeliveryScript

    result = 'Result of the call to the customer: {"confirmed":false,"party":"customer"}'
    request = BrainRequest(
        messages=(
            Message("system", "prefix"),
            Message("system", result),
            Message("assistant", "You've earned $182.40 this week."),
        )
    )
    reply = DeliveryScript()(request)
    assert "didn't confirm" in reply.text
    answered = BrainRequest(
        messages=(*request.messages, Message("assistant", reply.text), Message("user", "ok"))
    )
    assert "didn't confirm" not in DeliveryScript()(answered).text
