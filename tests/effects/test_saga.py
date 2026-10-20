import pytest

from switchboard.core.clock import VirtualClock
from switchboard.effects.ledger import InMemoryLedger
from switchboard.effects.runner import EffectContext, EffectRunner, ToolCall
from switchboard.effects.saga import Saga
from tests.effects.tools import Bench, build_bench

CTX = EffectContext(conversation_id="conv_1", intent=1)


@pytest.fixture
def bench() -> Bench:
    return build_bench(VirtualClock())


@pytest.fixture
def runner(bench: Bench) -> EffectRunner:
    return EffectRunner(bench.registry, InMemoryLedger(bench.clock, lease_s=60), bench.clock)


async def run_three(runner: EffectRunner, saga: Saga) -> None:
    calls = [
        (ToolCall("c1", "hold_slot", {"slot": "9am"}), 1),
        (ToolCall("c2", "issue_refund", {"order_id": "7", "amount": 5}), 2),
        (ToolCall("c3", "send_sms", {"text": "sorry"}), 3),
    ]
    for call, step in calls:
        result = await runner.run(call, EffectContext("conv_1", step))
        assert result.ok
        saga.record(result)


async def test_abort_compensates_in_reverse_order(runner: EffectRunner, bench: Bench) -> None:
    saga = Saga(runner)
    await run_three(runner, saga)
    report = await saga.abort(CTX)
    compensations = [name for name, _ in bench.log if name in {"void_refund", "release_slot"}]
    assert compensations == ["void_refund", "release_slot"]
    assert [(o.call_id, o.status) for o in report.outcomes] == [
        ("c3", "not_compensable"),
        ("c2", "compensated"),
        ("c1", "compensated"),
    ]
    assert report.ok


async def test_abort_twice_does_not_double_compensate(runner: EffectRunner, bench: Bench) -> None:
    saga = Saga(runner)
    await run_three(runner, saga)
    await saga.abort(CTX)
    again = await saga.abort(CTX)
    assert bench.calls["void_refund"] == 1 and bench.calls["release_slot"] == 1
    assert [o.status for o in again.outcomes] == ["not_compensable", "replayed", "replayed"]
    assert again.ok


async def test_failed_compensation_is_retried_without_double_running_others(
    runner: EffectRunner, bench: Bench
) -> None:
    saga = Saga(runner)
    await run_three(runner, saga)
    bench.fail_next["void_refund"] = 1
    first = await saga.abort(CTX)
    assert not first.ok
    assert [o.status for o in first.outcomes] == ["not_compensable", "failed", "compensated"]
    assert first.outcomes[1].result is not None
    assert first.outcomes[1].result.error == "RuntimeError: void_refund exploded"

    second = await saga.abort(CTX)
    assert second.ok
    assert [o.status for o in second.outcomes] == ["not_compensable", "compensated", "replayed"]
    assert bench.calls["void_refund"] == 2  # one failed attempt, one success
    assert bench.calls["release_slot"] == 1


async def test_record_ignores_failures_and_reads_and_dedupes_replays(
    runner: EffectRunner, bench: Bench
) -> None:
    saga = Saga(runner)
    read = await runner.run(ToolCall("r", "lookup_order", {"order_id": "1"}), CTX)
    bench.fail_next["hold_slot"] = 1
    failed = await runner.run(ToolCall("f", "hold_slot", {"slot": "x"}), CTX)
    held = await runner.run(ToolCall("h", "hold_slot", {"slot": "x"}), CTX)
    replay = await runner.run(ToolCall("h2", "hold_slot", {"slot": "x"}), CTX)
    for result in (read, failed, held, replay):
        saga.record(result)
    assert [r.call_id for r in saga.recorded] == ["h"]
    report = await saga.abort(CTX)
    assert [o.call_id for o in report.outcomes] == ["h"]
    assert bench.calls["release_slot"] == 1


async def test_empty_saga_aborts_cleanly(runner: EffectRunner) -> None:
    report = await Saga(runner).abort(CTX)
    assert report.ok and report.outcomes == []
