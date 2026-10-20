"""The effect runner: validates, deduplicates, times out and compensates tool calls (ADR-009).

Read tools run directly. Write tools run at most once per idempotency key: the runner claims the
key in the ledger, runs the tool, and records the result, so a retry or replay of the same step
gets the recorded result back with `replayed=True` instead of a second side effect. A write tool
never runs on a speculative turn (ADR-006): speculation can be cancelled, and a refund cannot.

The ledger only dedupes inside this runtime. When a write times out, the tool may already have
reached the downstream service, and the retry runs the body again. A write tool that declares a
keyword-only `idempotency_key` parameter receives the derived key (or, as a compensation, the
compensation key) on every attempt, so it can forward it and let the downstream service apply
the effect once.

Every failure comes back as a `ToolResult` with `ok=False` rather than an exception, because the
caller is a conversation that has to say something either way. The one exception is
cancellation of the caller itself, which marks the claim failed and propagates.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from switchboard.core.clock import Clock
from switchboard.effects.keys import compensation_key, idempotency_key
from switchboard.effects.ledger import IdempotencyLedger
from switchboard.effects.spec import KEY_PARAM, RegisteredTool, ToolRegistry, validate_args

DEFAULT_POLL_INTERVAL_S = 0.05


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class EffectContext:
    conversation_id: str
    # The conversation step that proposed the call. Part of the idempotency key: see keys.py.
    step_seq: int
    speculative: bool = False


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    idempotency_key: str | None = None
    replayed: bool = False
    duration_s: float = 0.0


@dataclass
class _Attempt:
    call: ToolCall
    started: float
    key: str | None = None


class _TimedOut(Exception):
    pass


class _BadResult(Exception):
    pass


class EffectRunner:
    def __init__(
        self,
        registry: ToolRegistry,
        ledger: IdempotencyLedger,
        clock: Clock,
        *,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        in_flight_wait_s: float | None = None,
    ) -> None:
        registry.validate()
        for spec in registry.specs():
            # A live attempt must be cancelled by its timeout before its claim can go stale,
            # or a second attempt could reclaim the key and run the effect again.
            if spec.effect == "write" and spec.timeout_s >= ledger.lease_s:
                raise ValueError(
                    f"{spec.name}: timeout_s {spec.timeout_s:g} must be below the ledger"
                    f" lease of {ledger.lease_s:g}s"
                )
        if not poll_interval_s > 0:
            raise ValueError("poll_interval_s must be positive")
        self._registry = registry
        self._ledger = ledger
        self._clock = clock
        self._poll_s = poll_interval_s
        self._in_flight_wait_s = in_flight_wait_s

    async def run(self, call: ToolCall, ctx: EffectContext) -> ToolResult:
        attempt = _Attempt(call, self._clock.now())
        tool = self._registry.get(call.tool)
        if tool is None:
            return self._error(attempt, f"unknown tool {call.tool!r}")
        errors = validate_args(tool.spec, call.args)
        if tool.accepts_key and KEY_PARAM in call.args:
            errors.append(f"args.{KEY_PARAM}: reserved for the runner")
        if errors:
            return self._error(attempt, f"invalid arguments for {call.tool}: " + "; ".join(errors))
        if tool.spec.effect == "read":
            return await self._run_read(tool, attempt)
        if ctx.speculative:
            return self._refuse_speculative(attempt)
        key = idempotency_key(
            ctx.conversation_id, ctx.step_seq, call.tool, call.args, tool.spec.idempotency
        )
        return await self._run_write(tool, attempt, key)

    def compensation_for(self, tool_name: str) -> str | None:
        """The compensation the spec of `tool_name` declares, if any."""
        tool = self._registry.get(tool_name)
        return None if tool is None else tool.spec.compensate

    async def compensate(self, original: ToolResult, ctx: EffectContext) -> ToolResult:
        """Undo `original` with the compensation its spec declares, at most once.

        The compensation's arguments are the fields its params declare, taken from the recorded
        result first and the original arguments second. The recorded result comes from the
        ledger, not from `original`, so a caller cannot steer the compensation at another
        record.
        """
        attempt = _Attempt(ToolCall(original.call_id, original.tool, {}), self._clock.now())
        tool = self._registry.get(original.tool)
        if tool is None:
            return self._error(attempt, f"unknown tool {original.tool!r}")
        target = tool.spec.compensate
        if target is None:
            return self._error(attempt, f"{original.tool} declares no compensation")
        if not original.ok or original.idempotency_key is None:
            return self._error(
                attempt,
                f"{original.tool} call {original.call_id} did not succeed; nothing to compensate",
            )
        entry = await self._ledger.get(original.idempotency_key)
        if entry is None or entry.state != "done" or entry.result is None:
            return self._error(
                attempt,
                f"no recorded result for {original.idempotency_key}; nothing to compensate",
            )
        compensation = self._registry.get(target)
        assert compensation is not None  # registry.validate() ran in __init__
        sources = {**original.args, **entry.result}
        declared = compensation.spec.params.get("properties", {})
        args = {name: sources[name] for name in declared if name in sources}
        attempt = _Attempt(ToolCall(original.call_id, target, args), attempt.started)
        errors = validate_args(compensation.spec, args)
        if errors:
            return self._error(
                attempt, f"cannot build arguments for {target}: " + "; ".join(errors)
            )
        if ctx.speculative:
            return self._refuse_speculative(attempt)
        key = compensation_key(original.idempotency_key, target)
        return await self._run_write(compensation, attempt, key)

    async def _run_read(self, tool: RegisteredTool, attempt: _Attempt) -> ToolResult:
        try:
            result = await self._execute(tool, attempt.call.args)
        except _TimedOut as exc:
            return self._error(attempt, str(exc))
        except Exception as exc:
            return self._error(attempt, _describe(exc))
        return self._ok(attempt, result, replayed=False)

    async def _run_write(self, tool: RegisteredTool, attempt: _Attempt, key: str) -> ToolResult:
        attempt.key = key
        wait_s = (
            self._in_flight_wait_s if self._in_flight_wait_s is not None else tool.spec.timeout_s
        )
        deadline = attempt.started + wait_s
        while True:
            claim = await self._ledger.claim(key)
            if claim.status == "done":
                assert claim.result is not None
                return self._ok(attempt, claim.result, replayed=True)
            if claim.status == "acquired":
                assert claim.token is not None
                token = claim.token
                break
            now = self._clock.now()
            if now >= deadline:
                return self._error(
                    attempt,
                    f"{tool.spec.name} with key {key} is still in flight in another attempt"
                    f" after {wait_s:g}s; not running it a second time",
                )
            await self._clock.sleep(min(self._poll_s, deadline - now))

        args = dict(attempt.call.args)
        if tool.accepts_key:
            args[KEY_PARAM] = key
        try:
            result = await self._execute(tool, args)
        except (_TimedOut, Exception) as exc:
            message = str(exc) if isinstance(exc, _TimedOut) else _describe(exc)
            await self._ledger.fail(key, token, message)
            return self._error(attempt, message)
        except BaseException:
            await self._ledger.fail(key, token, "cancelled")
            raise
        # A False here means the claim went stale and another attempt took over. The effect
        # has happened either way, so report it truthfully rather than hide it.
        await self._ledger.record(key, token, result)
        return self._ok(attempt, result, replayed=False)

    async def _execute(self, tool: RegisteredTool, args: dict[str, Any]) -> dict[str, Any]:
        work = asyncio.ensure_future(tool.fn(**args))
        timer = asyncio.ensure_future(self._clock.sleep(tool.spec.timeout_s))
        try:
            await asyncio.wait({work, timer}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (work, timer):
                if not task.done():
                    task.cancel()
        if not work.done() or work.cancelled():
            raise _TimedOut(f"{tool.spec.name} timed out after {tool.spec.timeout_s:g}s")
        result = work.result()
        if not isinstance(result, dict):
            raise _BadResult(
                f"{tool.spec.name} must return a JSON object, got {type(result).__name__}"
            )
        try:
            json.dumps(result, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise _BadResult(f"{tool.spec.name} must return a JSON object: {exc}") from exc
        return result

    def _refuse_speculative(self, attempt: _Attempt) -> ToolResult:
        return self._error(
            attempt,
            f"refused: {attempt.call.tool} is a write tool and the turn is speculative;"
            " write tools run only on a committed turn",
        )

    def _ok(self, attempt: _Attempt, result: dict[str, Any], *, replayed: bool) -> ToolResult:
        return ToolResult(
            call_id=attempt.call.call_id,
            tool=attempt.call.tool,
            args=attempt.call.args,
            ok=True,
            result=result,
            idempotency_key=attempt.key,
            replayed=replayed,
            duration_s=self._clock.now() - attempt.started,
        )

    def _error(self, attempt: _Attempt, message: str) -> ToolResult:
        return ToolResult(
            call_id=attempt.call.call_id,
            tool=attempt.call.tool,
            args=attempt.call.args,
            ok=False,
            error=message,
            idempotency_key=attempt.key,
            duration_s=self._clock.now() - attempt.started,
        )


def _describe(exc: BaseException) -> str:
    if isinstance(exc, _BadResult):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"
