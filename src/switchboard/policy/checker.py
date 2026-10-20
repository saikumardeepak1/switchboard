"""Pre-execution policy check (ADR-004). Pure: no I/O, no clock, no mutation of inputs."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from switchboard.core.events import Event
from switchboard.policy.expr import EvalContext, Expr, evaluate_indexed, index_events
from switchboard.policy.index import EventIndex
from switchboard.policy.procedure import ProcedureGraph, ProcedureState

__all__ = [
    "CheckContext",
    "CheckReport",
    "CompiledPolicy",
    "ProposedCall",
    "Rule",
    "Verdict",
]

PROCEDURE_RULE_PREFIX = "procedure:requires:"
COMPENSATION_RULE = "runtime:compensation_only"


@dataclass(frozen=True, slots=True)
class ProposedCall:
    tool: str
    args: Mapping[str, Any]
    call_id: str = ""


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Everything a check may read besides the call itself. `args` come from the call.

    A long-running caller passes `index` (kept up to date as events are appended) and
    `completed_steps` (its folded procedure state) instead of `events`, so a check costs the
    same at turn 500 as at turn 5. Given either, the checker never scans `events`.
    """

    evidence: Mapping[str, Any] = field(default_factory=dict)
    state: Mapping[str, Any] = field(default_factory=dict)
    events: Sequence[Event] = ()
    # Overrides the compile-time tool effect map for `when: {effect: ...}` rules.
    tool_effects: Mapping[str, str] | None = None
    index: EventIndex | None = None
    completed_steps: Collection[str] | None = None


@dataclass(frozen=True, slots=True)
class Verdict:
    allowed: bool
    rule_ids: list[str]
    repair_hint: str | None


@dataclass(frozen=True, slots=True)
class CheckReport:
    verdict: Verdict
    # Rule id to the paths that did not resolve while evaluating it, for failed rules only.
    missing: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    require: Expr
    source: str
    hint: str | None = None
    tools: frozenset[str] | None = None
    effect: str | None = None

    def failure_hint(self) -> str:
        return self.hint if self.hint is not None else f"Rule {self.id} failed: {self.source}"


@dataclass(frozen=True, slots=True)
class CompiledPolicy:
    rules: tuple[Rule, ...]
    procedure: ProcedureGraph
    tool_effects: Mapping[str, str] = field(default_factory=dict)
    # Tools only the runtime runs, as compensations. Never allowed as a proposal.
    compensation_only: frozenset[str] = frozenset()
    _by_tool: Mapping[str, tuple[Rule, ...]] = field(init=False, repr=False, compare=False)
    _by_effect: tuple[Rule, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_tool: dict[str, list[Rule]] = {}
        for rule in self.rules:
            for tool in rule.tools or ():
                by_tool.setdefault(tool, []).append(rule)
        object.__setattr__(
            self, "_by_tool", MappingProxyType({k: tuple(v) for k, v in by_tool.items()})
        )
        object.__setattr__(self, "_by_effect", tuple(r for r in self.rules if r.tools is None))
        object.__setattr__(self, "tool_effects", MappingProxyType(dict(self.tool_effects)))

    def _effect_of(self, tool: str, ctx: CheckContext) -> str:
        if ctx.tool_effects is not None and tool in ctx.tool_effects:
            return ctx.tool_effects[tool]
        # An unknown tool is treated as a write: rules guarding writes fail closed.
        return self.tool_effects.get(tool, "write")

    def applicable(self, tool: str, ctx: CheckContext) -> list[Rule]:
        """Rules whose `when` matches this tool, in declaration order."""
        named = self._by_tool.get(tool, ())
        effect: str | None = None
        selected: list[Rule] = []
        for rule in self.rules if named and self._by_effect else (*named, *self._by_effect):
            if rule.tools is not None and tool not in rule.tools:
                continue
            if rule.effect is not None:
                if effect is None:
                    effect = self._effect_of(tool, ctx)
                if rule.effect != effect:
                    continue
            selected.append(rule)
        return selected

    def explain(self, call: ProposedCall, ctx: CheckContext) -> CheckReport:
        if call.tool in self.compensation_only:
            refusal = (
                f"{call.tool} is a compensation that only the runtime runs. "
                "Offer a handoff if something needs to be undone."
            )
            return CheckReport(Verdict(False, [COMPENSATION_RULE], refusal), {})
        ids: list[str] = []
        hints: list[str] = []
        missing: dict[str, tuple[str, ...]] = {}

        by_type: dict[str, list[Event]] | None = None
        if ctx.index is None and (ctx.completed_steps is None or ctx.events):
            by_type = index_events(ctx.events)
        if self.procedure.require_before.get(call.tool):
            done = self._done(ctx, by_type)
            for step in self.procedure.missing_prerequisites(call.tool, done):
                ids.append(f"{PROCEDURE_RULE_PREFIX}{step}")
                hints.append(f"Complete step {step!r} before calling {call.tool}.")

        rules = self.applicable(call.tool, ctx)
        if rules:
            eval_ctx = EvalContext(call.args, ctx.evidence, ctx.state, ctx.events, ctx.index)
            for rule in rules:
                result = evaluate_indexed(rule.require, eval_ctx, by_type or {})
                if result.value is True:
                    continue
                ids.append(rule.id)
                hints.append(rule.failure_hint())
                if result.missing:
                    missing[rule.id] = result.missing

        hint = " ".join(dict.fromkeys(hints)) if hints else None
        return CheckReport(Verdict(not ids, ids, hint), missing)

    def _done(self, ctx: CheckContext, by_type: dict[str, list[Event]] | None) -> ProcedureState:
        if ctx.completed_steps is not None:
            return ProcedureState(frozenset(ctx.completed_steps))
        if ctx.index is not None:
            return self.procedure.state_from_events(
                (*ctx.index.of_type("procedure.advanced"), *ctx.index.of_type("tool.executed"))
            )
        assert by_type is not None
        return self.procedure.state_from_events(
            (*by_type.get("procedure.advanced", ()), *by_type.get("tool.executed", ()))
        )

    def check(self, call: ProposedCall, ctx: CheckContext) -> Verdict:
        return self.explain(call, ctx).verdict
