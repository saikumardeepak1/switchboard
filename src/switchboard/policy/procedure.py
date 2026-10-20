"""Procedure graph: which steps exist and which steps a tool needs first (ADR-004).

Completion rule. A step is completed when either

* a `procedure.advanced` event names it, or
* a `tool.executed` event succeeded for a tool that `completes` maps to it. A result counts as
  failed when it has an `error` key or `ok: false`; anything else is success. Replayed results
  count, because they are the recorded outcome of the same execution.

Steps not declared in `steps` are ignored, so a typo in an emitted event cannot satisfy a gate.

`steps` order is used for reporting (open steps, missing prerequisites). Gating is exactly what
`require_before` says and is not transitive: requiring `verify_identity` does not also require
`greet` just because it comes first.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from switchboard.core.events import Event

__all__ = ["ProcedureGraph", "ProcedureState", "tool_succeeded"]


@dataclass(frozen=True, slots=True)
class ProcedureState:
    completed: frozenset[str]

    @classmethod
    def empty(cls) -> ProcedureState:
        return cls(frozenset())


def tool_succeeded(result: Mapping[str, Any]) -> bool:
    return "error" not in result and result.get("ok", True) is not False


@dataclass(frozen=True, slots=True)
class ProcedureGraph:
    steps: tuple[str, ...] = ()
    require_before: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    completes: Mapping[str, str] = field(default_factory=dict)
    _order: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        order: dict[str, int] = {}
        for i, step in enumerate(self.steps):
            if step in order:
                raise ValueError(f"duplicate step {step!r} in procedure.steps")
            order[step] = i
        for tool, needed in self.require_before.items():
            for step in needed:
                if step not in order:
                    raise ValueError(f"require_before[{tool!r}] names unknown step {step!r}")
        for tool, step in self.completes.items():
            if step not in order:
                raise ValueError(f"completes[{tool!r}] names unknown step {step!r}")
        # Prerequisites are stored in step order so reports read in procedure order.
        ordered = {
            tool: tuple(sorted(dict.fromkeys(needed), key=order.__getitem__))
            for tool, needed in self.require_before.items()
        }
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "require_before", MappingProxyType(ordered))
        object.__setattr__(self, "completes", MappingProxyType(dict(self.completes)))
        object.__setattr__(self, "_order", MappingProxyType(order))

    def state_from_events(self, events: Iterable[Event]) -> ProcedureState:
        done: set[str] = set()
        for event in events:
            step: object = None
            if event.type == "procedure.advanced":
                step = getattr(event, "step", None)
            elif event.type == "tool.executed":
                result = getattr(event, "result", None)
                if isinstance(result, Mapping) and tool_succeeded(result):
                    step = self.completes.get(getattr(event, "tool", ""))
            if isinstance(step, str) and step in self._order:
                done.add(step)
        return ProcedureState(frozenset(done))

    def open_steps(self, state: ProcedureState) -> tuple[str, ...]:
        return tuple(s for s in self.steps if s not in state.completed)

    def next_step(self, state: ProcedureState) -> str | None:
        remaining = self.open_steps(state)
        return remaining[0] if remaining else None

    def missing_prerequisites(self, tool: str, state: ProcedureState) -> tuple[str, ...]:
        needed = self.require_before.get(tool)
        if not needed:
            return ()
        return tuple(s for s in needed if s not in state.completed)
