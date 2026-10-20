import asyncio
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import Any

import pytest

from switchboard.core.clock import VirtualClock
from switchboard.effects.keys import compensation_key, idempotency_key
from switchboard.effects.ledger import IdempotencyLedger, InMemoryLedger, SqliteLedger
from switchboard.effects.runner import EffectContext, EffectRunner, ToolCall, ToolResult
from switchboard.effects.spec import ToolRegistry
from tests.effects.tools import Bench, build_bench

CTX = EffectContext(conversation_id="conv_1", step_seq=4)
REFUND = {"order_id": "48213", "amount": 12.5}


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock()


@pytest.fixture
def bench(clock: VirtualClock) -> Bench:
    return build_bench(clock)


@pytest.fixture
async def ledger(clock: VirtualClock) -> AsyncIterator[IdempotencyLedger]:
    led = InMemoryLedger(clock, lease_s=60.0)
    yield led
    await led.close()


@pytest.fixture
def runner(bench: Bench, ledger: IdempotencyLedger, clock: VirtualClock) -> EffectRunner:
    return EffectRunner(bench.registry, ledger, clock)


def refund_call(call_id: str = "call_1", **overrides: Any) -> ToolCall:
    return ToolCall(call_id=call_id, tool="issue_refund", args={**REFUND, **overrides})


async def drive(clock: VirtualClock, coro: Coroutine[Any, Any, ToolResult]) -> ToolResult:
    task = asyncio.create_task(coro)
    await clock.run_until_idle()
    return await task


async def test_unknown_tool_is_an_error_result(runner: EffectRunner) -> None:
    result = await runner.run(ToolCall(call_id="c", tool="launch_rocket", args={}), CTX)
    assert not result.ok
    assert result.error == "unknown tool 'launch_rocket'"
    assert result.call_id == "c" and result.tool == "launch_rocket"


async def test_invalid_args_list_every_problem_and_do_not_run(
    runner: EffectRunner, bench: Bench, ledger: IdempotencyLedger
) -> None:
    call = ToolCall(call_id="c", tool="issue_refund", args={"amount": "lots"})
    result = await runner.run(call, CTX)
    assert not result.ok
    assert result.error is not None
    assert "args.amount: expected number, got string" in result.error
    assert "args.order_id: required property is missing" in result.error
    assert bench.calls["issue_refund"] == 0
    assert result.idempotency_key is None


async def test_read_tool_runs_without_ledger(runner: EffectRunner, bench: Bench) -> None:
    call = ToolCall(call_id="c", tool="lookup_order", args={"order_id": "1"})
    first = await runner.run(call, CTX)
    second = await runner.run(call, CTX)
    assert first.ok and first.result == {"order_id": "1", "status": "late"}
    assert first.idempotency_key is None and not first.replayed
    assert not second.replayed
    assert bench.calls["lookup_order"] == 2


async def test_read_tool_may_run_speculatively(runner: EffectRunner, bench: Bench) -> None:
    call = ToolCall(call_id="c", tool="lookup_order", args={"order_id": "1"})
    result = await runner.run(call, EffectContext("conv_1", 4, speculative=True))
    assert result.ok and bench.calls["lookup_order"] == 1


async def test_speculative_write_is_refused_and_never_runs(
    runner: EffectRunner, bench: Bench, ledger: IdempotencyLedger
) -> None:
    result = await runner.run(refund_call(), EffectContext("conv_1", 4, speculative=True))
    assert not result.ok
    assert result.error is not None and "speculative" in result.error
    assert bench.calls["issue_refund"] == 0
    key = idempotency_key("conv_1", 4, "issue_refund", REFUND, ["order_id", "amount"])
    assert await ledger.get(key) is None


async def test_write_runs_once_and_replays_identical_result(
    runner: EffectRunner, bench: Bench
) -> None:
    first = await runner.run(refund_call("call_1"), CTX)
    second = await runner.run(refund_call("call_2"), CTX)
    assert first.ok and not first.replayed
    assert second.ok and second.replayed
    assert second.result == first.result == {"refund_id": "ref_1", "amount": 12.5}
    assert second.idempotency_key == first.idempotency_key
    assert second.call_id == "call_2"
    assert bench.calls["issue_refund"] == 1


async def test_non_identity_args_do_not_change_the_intent(
    runner: EffectRunner, bench: Bench
) -> None:
    await runner.run(refund_call(note="first wording"), CTX)
    again = await runner.run(refund_call(note="second wording"), CTX)
    assert again.replayed and bench.calls["issue_refund"] == 1


async def test_same_refund_at_a_later_step_is_a_new_intent(
    runner: EffectRunner, bench: Bench
) -> None:
    await runner.run(refund_call(), CTX)
    later = await runner.run(refund_call(), EffectContext("conv_1", 9))
    assert not later.replayed and bench.calls["issue_refund"] == 2


async def test_concurrent_runs_of_one_key_execute_once(
    runner: EffectRunner, bench: Bench, clock: VirtualClock
) -> None:
    bench.delay_s["issue_refund"] = 1.0
    tasks = [asyncio.create_task(runner.run(refund_call(f"call_{i}"), CTX)) for i in range(5)]
    await clock.run_until_idle()
    results = await asyncio.gather(*tasks)
    assert bench.calls["issue_refund"] == 1
    assert all(r.ok for r in results)
    assert sorted(r.replayed for r in results) == [False, True, True, True, True]
    assert all(r.result == results[0].result for r in results)


async def test_failure_marks_failed_and_retry_executes_again(
    runner: EffectRunner, bench: Bench, ledger: IdempotencyLedger
) -> None:
    bench.fail_next["issue_refund"] = 1
    failed = await runner.run(refund_call(), CTX)
    assert not failed.ok
    assert failed.error == "RuntimeError: issue_refund exploded"
    assert failed.idempotency_key is not None
    entry = await ledger.get(failed.idempotency_key)
    assert entry is not None and entry.state == "failed"
    retried = await runner.run(refund_call(), CTX)
    assert retried.ok and not retried.replayed
    assert bench.calls["issue_refund"] == 2


async def test_lost_reply_is_recovered_without_re_executing(
    tmp_path: Path, clock: VirtualClock, bench: Bench
) -> None:
    path = tmp_path / "ledger.sqlite3"
    first_process = SqliteLedger(path, clock, lease_s=60)
    lost = await EffectRunner(bench.registry, first_process, clock).run(refund_call(), CTX)
    assert lost.ok
    # The caller crashes here and never sees `lost`. A fresh process retries the same step.
    await first_process.close()
    del lost
    second_process = SqliteLedger(path, clock, lease_s=60)
    retry = await EffectRunner(bench.registry, second_process, clock).run(refund_call(), CTX)
    await second_process.close()
    assert retry.ok and retry.replayed
    assert retry.result == {"refund_id": "ref_1", "amount": 12.5}
    assert bench.calls["issue_refund"] == 1


async def test_crash_before_record_is_reclaimed_after_lease(
    runner: EffectRunner, bench: Bench, ledger: IdempotencyLedger, clock: VirtualClock
) -> None:
    key = idempotency_key("conv_1", 4, "issue_refund", REFUND, ["order_id", "amount"])
    assert (await ledger.claim(key)).status == "acquired"  # a process that then died
    await clock.advance(61)
    result = await runner.run(refund_call(), CTX)
    assert result.ok and not result.replayed
    assert bench.calls["issue_refund"] == 1


async def test_in_flight_elsewhere_waits_then_returns_clear_error(
    runner: EffectRunner, bench: Bench, ledger: IdempotencyLedger, clock: VirtualClock
) -> None:
    key = idempotency_key("conv_1", 4, "issue_refund", REFUND, ["order_id", "amount"])
    await ledger.claim(key)
    result = await drive(clock, runner.run(refund_call(), CTX))
    assert not result.ok
    assert result.error is not None and "still in flight" in result.error
    assert result.idempotency_key == key
    assert clock.now() == pytest.approx(5.0)
    assert bench.calls["issue_refund"] == 0


async def test_in_flight_elsewhere_returns_result_once_it_lands(
    runner: EffectRunner, bench: Bench, ledger: IdempotencyLedger, clock: VirtualClock
) -> None:
    key = idempotency_key("conv_1", 4, "issue_refund", REFUND, ["order_id", "amount"])
    claim = await ledger.claim(key)
    assert claim.token is not None
    task = asyncio.create_task(runner.run(refund_call(), CTX))
    await clock.advance(1.0)
    await ledger.record(key, claim.token, {"refund_id": "elsewhere"})
    await clock.run_until_idle()
    result = await task
    assert result.ok and result.replayed and result.result == {"refund_id": "elsewhere"}
    assert bench.calls["issue_refund"] == 0


async def test_write_timeout_cancels_tool_and_marks_failed(
    runner: EffectRunner, bench: Bench, ledger: IdempotencyLedger, clock: VirtualClock
) -> None:
    bench.delay_s["issue_refund"] = 10.0
    result = await drive(clock, runner.run(refund_call(), CTX))
    assert not result.ok
    assert result.error == "issue_refund timed out after 5s"
    assert result.duration_s == pytest.approx(5.0)
    assert bench.cancelled == ["issue_refund"]
    assert result.idempotency_key is not None
    entry = await ledger.get(result.idempotency_key)
    assert entry is not None and entry.state == "failed"


async def test_read_timeout_is_enforced(
    runner: EffectRunner, bench: Bench, clock: VirtualClock
) -> None:
    bench.delay_s["lookup_order"] = 3.0
    call = ToolCall(call_id="c", tool="lookup_order", args={"order_id": "1"})
    result = await drive(clock, runner.run(call, CTX))
    assert not result.ok and result.error == "lookup_order timed out after 2s"
    assert bench.cancelled == ["lookup_order"]


async def test_duration_comes_from_the_injected_clock(
    runner: EffectRunner, bench: Bench, clock: VirtualClock
) -> None:
    bench.delay_s["issue_refund"] = 1.5
    result = await drive(clock, runner.run(refund_call(), CTX))
    assert result.ok and result.duration_s == pytest.approx(1.5)


async def test_non_dict_result_is_an_error(runner: EffectRunner) -> None:
    result = await runner.run(ToolCall(call_id="c", tool="bad_result", args={}), CTX)
    assert not result.ok
    assert result.error is not None and "must return a JSON object" in result.error


async def test_cancelling_the_caller_marks_the_claim_failed(
    runner: EffectRunner, bench: Bench, ledger: IdempotencyLedger, clock: VirtualClock
) -> None:
    bench.delay_s["issue_refund"] = 3.0
    task = asyncio.create_task(runner.run(refund_call(), CTX))
    await clock.advance(1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bench.cancelled == ["issue_refund"]
    key = idempotency_key("conv_1", 4, "issue_refund", REFUND, ["order_id", "amount"])
    entry = await ledger.get(key)
    assert entry is not None and entry.state == "failed"


async def test_compensate_runs_declared_tool_at_most_once(
    runner: EffectRunner, bench: Bench
) -> None:
    original = await runner.run(refund_call(), CTX)
    first = await runner.compensate(original, CTX)
    second = await runner.compensate(original, CTX)
    assert first.ok and first.tool == "void_refund" and not first.replayed
    assert first.result == {"voided": "ref_1"}
    assert second.ok and second.replayed and second.result == first.result
    assert bench.calls["void_refund"] == 1
    assert bench.log[-1] == ("void_refund", {"refund_id": "ref_1", "order_id": "48213"})
    assert first.idempotency_key is not None and first.idempotency_key.startswith("void_refund:")


async def test_compensate_without_declared_compensation_is_an_error(
    runner: EffectRunner,
) -> None:
    sent = await runner.run(ToolCall(call_id="c", tool="send_sms", args={"text": "hi"}), CTX)
    result = await runner.compensate(sent, CTX)
    assert not result.ok and result.error == "send_sms declares no compensation"


async def test_compensate_of_failed_result_is_an_error(runner: EffectRunner, bench: Bench) -> None:
    bench.fail_next["issue_refund"] = 1
    failed = await runner.run(refund_call(), CTX)
    result = await runner.compensate(failed, CTX)
    assert not result.ok and result.error is not None and "did not succeed" in result.error
    assert bench.calls["void_refund"] == 0


async def test_compensate_uses_recorded_result_not_caller_copy(
    runner: EffectRunner, bench: Bench
) -> None:
    original = await runner.run(refund_call(), CTX)
    forged = ToolResult(
        call_id=original.call_id,
        tool=original.tool,
        args=original.args,
        ok=True,
        result={"refund_id": "someone_elses_refund", "amount": 12.5},
        error=None,
        idempotency_key=original.idempotency_key,
        replayed=False,
        duration_s=0.0,
    )
    await runner.compensate(forged, CTX)
    assert bench.log[-1] == ("void_refund", {"refund_id": "ref_1", "order_id": "48213"})


async def test_compensate_refuses_speculative_context(runner: EffectRunner, bench: Bench) -> None:
    original = await runner.run(refund_call(), CTX)
    result = await runner.compensate(original, EffectContext("conv_1", 4, speculative=True))
    assert not result.ok and bench.calls["void_refund"] == 0


def test_runner_rejects_write_timeout_not_below_lease(clock: VirtualClock) -> None:
    bench = build_bench(clock)
    with pytest.raises(ValueError, match="lease"):
        EffectRunner(bench.registry, InMemoryLedger(clock, lease_s=5.0), clock)


def test_runner_validates_registry(clock: VirtualClock) -> None:
    registry = ToolRegistry()

    async def t() -> dict[str, Any]:
        return {}

    registry.tool(
        name="t", description="d", effect="write", params={"type": "object"}, compensate="gone"
    )(t)
    with pytest.raises(Exception, match="not registered"):
        EffectRunner(registry, InMemoryLedger(clock, lease_s=60), clock)


class FakePaymentsApi:
    """A downstream service that dedupes on an Idempotency-Key header, as Stripe does."""

    def __init__(self) -> None:
        self.applied: dict[str, dict[str, Any]] = {}
        self.requests: list[str] = []

    def refund(self, order_id: str, idempotency_key: str) -> dict[str, Any]:
        self.requests.append(idempotency_key)
        if idempotency_key not in self.applied:
            self.applied[idempotency_key] = {"refund_id": f"ext_{len(self.applied) + 1}"}
        return self.applied[idempotency_key]


def keyed_registry(clock: VirtualClock, api: FakePaymentsApi, body_runs: list[str]) -> ToolRegistry:
    registry = ToolRegistry()
    params = {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    }

    @registry.tool(
        name="refund",
        description="d",
        effect="write",
        params=params,
        compensate="unrefund",
        timeout_s=5.0,
    )
    async def refund(order_id: str, *, idempotency_key: str) -> dict[str, Any]:
        body_runs.append(idempotency_key)
        reply = api.refund(order_id, idempotency_key)
        if len(body_runs) == 1:
            await clock.sleep(30)  # the reply is slow and the runner gives up first
        return reply

    @registry.tool(
        name="unrefund",
        description="d",
        effect="write",
        params={"type": "object", "properties": {"refund_id": {"type": "string"}}},
    )
    async def unrefund(refund_id: str, *, idempotency_key: str) -> dict[str, Any]:
        body_runs.append(idempotency_key)
        return {"voided": refund_id}

    return registry


async def test_timeout_then_retry_applies_downstream_effect_exactly_once(
    clock: VirtualClock,
) -> None:
    api = FakePaymentsApi()
    body_runs: list[str] = []
    runner = EffectRunner(
        keyed_registry(clock, api, body_runs), InMemoryLedger(clock, lease_s=60), clock
    )
    call = ToolCall("c1", "refund", {"order_id": "48213"})
    first = await drive(clock, runner.run(call, CTX))
    assert not first.ok and first.error == "refund timed out after 5s"
    second = await runner.run(call, CTX)
    assert second.ok and not second.replayed
    assert len(body_runs) == 2
    assert body_runs[0] == body_runs[1] == first.idempotency_key == second.idempotency_key
    assert len(api.applied) == 1
    assert second.result == {"refund_id": "ext_1"}


async def test_compensation_receives_its_own_compensation_key(clock: VirtualClock) -> None:
    api = FakePaymentsApi()
    body_runs: list[str] = ["skip slow path"]
    runner = EffectRunner(
        keyed_registry(clock, api, body_runs), InMemoryLedger(clock, lease_s=60), clock
    )
    original = await runner.run(ToolCall("c1", "refund", {"order_id": "1"}), CTX)
    undo = await runner.compensate(original, CTX)
    assert original.idempotency_key is not None
    assert undo.ok
    assert body_runs[-1] == undo.idempotency_key
    assert undo.idempotency_key == compensation_key(original.idempotency_key, "unrefund")


async def test_tools_without_the_parameter_get_no_key(runner: EffectRunner, bench: Bench) -> None:
    result = await runner.run(refund_call(), CTX)
    assert result.ok  # issue_refund takes no idempotency_key and must not be passed one


async def test_caller_cannot_supply_the_key_itself(clock: VirtualClock) -> None:
    registry = ToolRegistry()
    runs: list[str] = []

    @registry.tool(
        name="open_ended",
        description="d",
        effect="write",
        params={"type": "object", "properties": {}, "additionalProperties": True},
    )
    async def open_ended(*, idempotency_key: str, **_: Any) -> dict[str, Any]:
        runs.append(idempotency_key)
        return {}

    runner = EffectRunner(registry, InMemoryLedger(clock, lease_s=60), clock)
    result = await runner.run(ToolCall("c", "open_ended", {"idempotency_key": "mine"}), CTX)
    assert not result.ok
    assert result.error is not None and "args.idempotency_key: reserved" in result.error
    assert runs == []
