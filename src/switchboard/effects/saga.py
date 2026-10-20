"""Sagas: undo a sequence of committed writes in reverse order (ADR-009).

The caller records results; it never names a compensation. On `abort` the runner looks up each
tool's declared compensation in its spec. Every compensation goes through the ledger under a key
derived from the original effect, so `abort` can be called again after a partial failure: the
compensations that already ran come back replayed and only the failed ones run again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from switchboard.effects.runner import EffectContext, EffectRunner, ToolResult

OutcomeStatus = Literal["compensated", "replayed", "failed", "not_compensable"]


@dataclass(frozen=True)
class CompensationOutcome:
    call_id: str
    tool: str
    compensation_tool: str | None
    status: OutcomeStatus
    result: ToolResult | None


@dataclass(frozen=True)
class SagaReport:
    outcomes: list[CompensationOutcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(o.status != "failed" for o in self.outcomes)


class Saga:
    def __init__(self, runner: EffectRunner) -> None:
        self._runner = runner
        self._recorded: list[ToolResult] = []
        self._keys: set[str] = set()

    @property
    def recorded(self) -> list[ToolResult]:
        return list(self._recorded)

    def record(self, result: ToolResult) -> None:
        """Track a write that took effect. Reads, failures and repeat replays are ignored."""
        key = result.idempotency_key
        if not result.ok or key is None or key in self._keys:
            return
        self._keys.add(key)
        self._recorded.append(result)

    async def abort(self, ctx: EffectContext) -> SagaReport:
        outcomes: list[CompensationOutcome] = []
        for original in reversed(self._recorded):
            target = self._runner.compensation_for(original.tool)
            if target is None:
                outcomes.append(
                    CompensationOutcome(
                        original.call_id, original.tool, None, "not_compensable", None
                    )
                )
                continue
            undo = await self._runner.compensate(original, ctx)
            status: OutcomeStatus
            if not undo.ok:
                status = "failed"
            elif undo.replayed:
                status = "replayed"
            else:
                status = "compensated"
            outcomes.append(
                CompensationOutcome(original.call_id, original.tool, target, status, undo)
            )
        return SagaReport(outcomes)
