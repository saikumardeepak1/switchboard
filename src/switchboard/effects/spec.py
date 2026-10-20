"""Tool specs, argument validation, and the tool registry (ADR-009).

The parameter schema is a deliberately small subset of JSON Schema: `type` (string, number,
integer, boolean, object, array), `properties`, `required`, `enum`, `minimum`, `maximum`,
`items`, `additionalProperties` and `description`. Any other keyword is rejected when the spec
is built, so a schema never looks stricter than the checks that actually run.

One intentional difference from JSON Schema: unknown properties are rejected unless
`additionalProperties` is true. An LLM inventing an argument is a bug worth surfacing before a
write tool runs, not something to pass through silently.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

Effect = Literal["read", "write"]
ToolFn = Callable[..., Awaitable[dict[str, Any]]]

_TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array"})
_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "enum",
        "minimum",
        "maximum",
        "items",
        "additionalProperties",
        "description",
    }
)

DEFAULT_TIMEOUT_S = 10.0


class RegistryError(Exception):
    """The registry is inconsistent: duplicate names or broken compensation references."""


def _check_schema(schema: object, path: str) -> None:
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path}: schema must be a mapping")
    unknown = set(schema) - _KEYWORDS
    if unknown:
        raise ValueError(f"{path}: unsupported keyword(s) {sorted(unknown)}")
    kind = schema.get("type")
    if kind not in _TYPES:
        raise ValueError(f"{path}: unsupported type {kind!r}")
    if kind == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path}.properties must be a mapping")
        for name, sub in properties.items():
            _check_schema(sub, f"{path}.{name}")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
            raise ValueError(f"{path}.required must be a list of property names")
        missing = [r for r in required if r not in properties]
        if missing:
            raise ValueError(f"{path}.required names undeclared properties {missing}")
    if kind == "array" and "items" in schema:
        _check_schema(schema["items"], f"{path}[]")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError(f"{path}.enum must be a list")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    effect: Effect
    params: Mapping[str, Any]
    # Argument names that define the intent's identity. None means every argument.
    idempotency: Sequence[str] | None = None
    compensate: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        if self.effect not in ("read", "write"):
            raise ValueError(f"{self.name}: effect must be 'read' or 'write', not {self.effect!r}")
        _check_schema(self.params, f"{self.name}.params")
        if self.params["type"] != "object":
            raise ValueError(f"{self.name}.params: top-level type must be 'object'")
        if self.idempotency is not None:
            object.__setattr__(self, "idempotency", tuple(self.idempotency))
            declared = self.params.get("properties", {})
            for field_name in self.idempotency:
                if field_name not in declared:
                    raise ValueError(
                        f"{self.name}: idempotency field {field_name!r} is not a declared param"
                    )
        if self.compensate is not None and self.effect == "read":
            raise ValueError(f"{self.name}: a read tool has nothing to compensate")
        if not self.timeout_s > 0:
            raise ValueError(f"{self.name}: timeout_s must be positive")

    def identity_fields(self, args: Mapping[str, Any]) -> list[str]:
        if self.idempotency is None:
            return sorted(args)
        return sorted(self.idempotency)


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list | tuple):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _matches_type(value: object, kind: str) -> bool:
    actual = _type_name(value)
    if kind == "number":
        return actual in ("integer", "number")
    if kind == "integer":
        # JSON has one number type, so 2.0 is a valid integer the way JSON Schema defines it.
        return actual == "integer" or (actual == "number" and float(value).is_integer())  # type: ignore[arg-type]
    return actual == kind


def _validate(value: object, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    kind: str = schema["type"]
    if not _matches_type(value, kind):
        errors.append(f"{path}: expected {kind}, got {_type_name(value)}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']!r}")
    if kind in ("number", "integer"):
        assert isinstance(value, int | float)
        if not math.isfinite(value):
            errors.append(f"{path}: expected a finite number")
            return
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value!r} is below the minimum of {schema['minimum']!r}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value!r} is above the maximum of {schema['maximum']!r}")
    elif kind == "array" and "items" in schema:
        assert isinstance(value, list | tuple)
        for i, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{i}]", errors)
    elif kind == "object":
        assert isinstance(value, Mapping)
        properties: Mapping[str, Any] = schema.get("properties", {})
        open_ended = schema.get("additionalProperties") is True
        for name in sorted(value):
            if name in properties:
                _validate(value[name], properties[name], f"{path}.{name}", errors)
            elif not open_ended:
                errors.append(f"{path}.{name}: unexpected property")
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name}: required property is missing")


def validate_args(spec: ToolSpec, args: Mapping[str, Any]) -> list[str]:
    """Return every problem with `args` against `spec.params`; empty means valid."""
    errors: list[str] = []
    _validate(args, spec.params, "args", errors)
    return errors


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    fn: ToolFn


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, fn: ToolFn) -> None:
        if spec.name in self._tools:
            raise RegistryError(f"tool {spec.name!r} is already registered")
        if not inspect.iscoroutinefunction(fn):
            raise RegistryError(f"tool {spec.name!r} must be an async function")
        self._tools[spec.name] = RegisteredTool(spec, fn)

    def tool(
        self,
        *,
        name: str,
        description: str,
        effect: Effect,
        params: Mapping[str, Any],
        idempotency: Sequence[str] | None = None,
        compensate: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> Callable[[ToolFn], ToolFn]:
        spec = ToolSpec(
            name=name,
            description=description,
            effect=effect,
            params=params,
            idempotency=idempotency,
            compensate=compensate,
            timeout_s=timeout_s,
        )

        def decorate(fn: ToolFn) -> ToolFn:
            self.register(spec, fn)
            return fn

        return decorate

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec for name in self.names()]

    def validate(self) -> None:
        """Check cross-tool references. Run once all tools are registered."""
        for spec in self.specs():
            target = spec.compensate
            if target is None:
                continue
            if target == spec.name:
                raise RegistryError(f"{spec.name!r} cannot compensate itself")
            registered = self._tools.get(target)
            if registered is None:
                raise RegistryError(
                    f"{spec.name!r} compensates with {target!r}, which is not registered"
                )
            if registered.spec.effect != "write":
                raise RegistryError(
                    f"{spec.name!r} compensates with {target!r}, but {target!r} is a read tool"
                )
