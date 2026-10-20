"""Compile policy YAML documents (already loaded into mappings) into a `CompiledPolicy`.

Merging rules across documents:

* `procedure.steps` may be declared by at most one document, since two step orders cannot be
  merged meaningfully.
* `procedure.require_before` lists are unioned per tool.
* `procedure.completes` entries merge; the same tool mapped to two different steps is an error.
* Rules keep document order, and ids are unique across all documents. Ids starting with
  `procedure:` are reserved for prerequisite violations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from switchboard.policy.checker import CompiledPolicy, Rule
from switchboard.policy.expr import ParseError, parse
from switchboard.policy.procedure import ProcedureGraph

__all__ = ["CompileError", "compile_policies"]

EFFECTS = frozenset({"read", "write"})
_TOP_KEYS = frozenset({"procedure", "rules"})
_PROCEDURE_KEYS = frozenset({"steps", "require_before", "completes"})
_RULE_KEYS = frozenset({"id", "when", "require", "hint"})
_WHEN_KEYS = frozenset({"tool", "effect"})


class CompileError(ValueError):
    pass


def _str_list(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CompileError(f"{where!r} must be a list of strings")
    return list(value)


def _unknown_keys(data: Mapping[Any, Any], allowed: frozenset[str], where: str) -> None:
    for key in data:
        if key not in allowed:
            raise CompileError(f"unknown key {key!r} in {where}")


class _Builder:
    def __init__(self) -> None:
        self.steps: list[str] | None = None
        self.require_before: dict[str, list[str]] = {}
        self.completes: dict[str, str] = {}
        self.rules: list[Rule] = []
        self.ids: set[str] = set()

    def add_document(self, index: int, doc: object) -> None:
        if not isinstance(doc, Mapping):
            raise CompileError(f"document {index} must be a mapping, got {type(doc).__name__}")
        for key in doc:
            if key not in _TOP_KEYS:
                raise CompileError(f"document {index}: unknown top-level key {key!r}")
        if doc.get("procedure") is not None:
            self.add_procedure(doc["procedure"])
        rules = doc.get("rules")
        if rules is None:
            return
        if not isinstance(rules, list):
            raise CompileError(f"document {index}: 'rules' must be a list")
        for i, raw in enumerate(rules):
            self.add_rule(i, raw)

    def add_procedure(self, proc: object) -> None:
        if not isinstance(proc, Mapping):
            raise CompileError("'procedure' must be a mapping")
        _unknown_keys(proc, _PROCEDURE_KEYS, "'procedure'")
        if "steps" in proc:
            if self.steps is not None:
                raise CompileError("'procedure.steps' is declared more than once")
            self.steps = _str_list(proc["steps"], "procedure.steps")
        require_before = proc.get("require_before") or {}
        if not isinstance(require_before, Mapping):
            raise CompileError("'procedure.require_before' must be a mapping of tool to steps")
        for tool, steps in require_before.items():
            merged = self.require_before.setdefault(str(tool), [])
            for step in _str_list(steps, f"procedure.require_before.{tool}"):
                if step not in merged:
                    merged.append(step)
        completes = proc.get("completes") or {}
        if not isinstance(completes, Mapping):
            raise CompileError("'procedure.completes' must be a mapping of tool to step")
        for tool, step in completes.items():
            if not isinstance(step, str):
                raise CompileError(f"'procedure.completes.{tool}' must be a step name")
            previous = self.completes.setdefault(str(tool), step)
            if previous != step:
                raise CompileError(
                    f"conflicting 'completes' for tool {tool!r}: {previous!r} and {step!r}"
                )

    def add_rule(self, index: int, raw: object) -> None:
        if not isinstance(raw, Mapping):
            raise CompileError(f"rules[{index}] must be a mapping")
        rule_id = raw.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            raise CompileError(f"rules[{index}]: missing 'id'")
        where = f"rule {rule_id!r}"
        if rule_id in self.ids:
            raise CompileError(f"duplicate rule id {rule_id!r}")
        if rule_id.startswith("procedure:"):
            raise CompileError(f"{where}: ids starting with 'procedure:' are reserved")
        for key in raw:
            if key not in _RULE_KEYS:
                raise CompileError(f"{where}: unknown key {key!r}")
        tools, effect = self.parse_when(raw.get("when"), where)
        if "require" not in raw:
            raise CompileError(f"{where}: missing 'require'")
        source = raw["require"]
        if not isinstance(source, str):
            raise CompileError(f"{where}: 'require' must be a string")
        hint = raw.get("hint")
        if hint is not None and not isinstance(hint, str):
            raise CompileError(f"{where}: 'hint' must be a string")
        try:
            expr = parse(source)
        except ParseError as err:
            pointer = "\n".join(f"    {line}" for line in err.pointer().split("\n"))
            raise CompileError(f"{where}: {err}\n{pointer}") from err
        self.ids.add(rule_id)
        self.rules.append(Rule(rule_id, expr, source, hint, tools, effect))

    @staticmethod
    def parse_when(when: object, where: str) -> tuple[frozenset[str] | None, str | None]:
        if not isinstance(when, Mapping):
            raise CompileError(f"{where}: 'when' must be a mapping")
        _unknown_keys(when, _WHEN_KEYS, "'when'")
        if "tool" not in when and "effect" not in when:
            raise CompileError(f"{where}: 'when' needs 'tool' or 'effect'")
        tools: frozenset[str] | None = None
        if "tool" in when:
            raw = when["tool"]
            names = [raw] if isinstance(raw, str) else raw
            tools = frozenset(_str_list(names, f"{where} when.tool"))
            if not tools:
                raise CompileError(f"{where}: 'when.tool' must name at least one tool")
        effect = when.get("effect")
        if effect is not None and effect not in EFFECTS:
            raise CompileError(f"{where}: effect must be 'read' or 'write', got {effect!r}")
        return tools, effect

    def build(self, tool_effects: Mapping[str, str]) -> CompiledPolicy:
        try:
            graph = ProcedureGraph(
                steps=tuple(self.steps or ()),
                require_before={k: tuple(v) for k, v in self.require_before.items()},
                completes=self.completes,
            )
        except ValueError as err:
            raise CompileError(f"procedure: {err}") from err
        return CompiledPolicy(tuple(self.rules), graph, tool_effects)


def compile_policies(
    docs: Sequence[Mapping[str, Any]], *, tool_effects: Mapping[str, str] | None = None
) -> CompiledPolicy:
    """Validate and merge policy documents. Raises `CompileError` with the offending location."""
    for tool, effect in (tool_effects or {}).items():
        if effect not in EFFECTS:
            raise CompileError(f"tool_effects[{tool!r}] must be 'read' or 'write', got {effect!r}")
    builder = _Builder()
    for index, doc in enumerate(docs):
        builder.add_document(index, doc)
    return builder.build(tool_effects or {})
