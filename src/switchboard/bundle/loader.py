"""Load an agent directory into an immutable, content-addressed release bundle (ADR-012).

Directory format::

    agent.yaml          identity, system prompt, voice, router overrides, limits, parties
    policies/*.yaml     policy documents (ADR-004), compiled together in file name order
    tools.py            defines a module-level `registry: effects.ToolRegistry`
    knowledge/*.md      optional documents for retrieval
    anything else       hashed, so helper modules imported by tools.py are part of the release

`tools.py` is trusted code from the customer's own repository. It is imported as a submodule of
a fresh synthetic package per load, so relative imports work and two loads never share state.
That package and everything imported under it live in `sys.modules` until `Bundle.close()`
removes them; a bundle is also a context manager. A failed import cleans up after itself.

The bundle id is `bnd_` plus the first 12 hex chars of a sha256 over every file (sorted relative
paths with their bytes; `__pycache__`, `*.pyc` and hidden files skipped) and the canonical JSON of
the tool specs. Specs are hashed as well as files because a registry can be built from data the
files only reference.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import itertools
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml  # type: ignore[import-untyped]  # no stubs in the dev environment

from switchboard.bundle.config import AgentConfig, ConfigError, parse_agent_config
from switchboard.bundle.runtime_tools import RUNTIME_TOOLS
from switchboard.effects import RegistryError, ToolRegistry, ToolSpec, canonical_json
from switchboard.policy import CompiledPolicy, CompileError, compile_policies
from switchboard.policy.expr import And, Compare, Count, Expr, ListExpr, Literal, Not, Or

_load_counter = itertools.count(1)


class BundleError(ValueError):
    """The agent directory is not a valid bundle. The message names the file and the problem."""


@dataclass(frozen=True)
class KnowledgeDoc:
    doc_id: str
    title: str
    body: str


@dataclass(frozen=True)
class Bundle:
    bundle_id: str
    root: Path
    config: AgentConfig
    policy: CompiledPolicy
    # Canonical text of the policy documents, for the stable prompt prefix.
    policies_text: str
    registry: ToolRegistry
    runtime_tools: tuple[ToolSpec, ...]
    documents: tuple[KnowledgeDoc, ...]
    tools_module: ModuleType
    # The synthetic package `tools.py` was imported under; see `close`.
    module_namespace: str = ""
    # Compensation targets: run by the runtime only, never offered to or callable by the model.
    compensation_tools: frozenset[str] = frozenset()

    def close(self) -> None:
        """Drop this load's modules from `sys.modules`. Idempotent.

        Conversations pinned to the bundle keep working: they hold the registry and the tools
        module directly. Only a later import by name would miss.
        """
        _unload(self.module_namespace)

    def __enter__(self) -> Bundle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def tool_specs(self) -> list[ToolSpec]:
        """Every tool the model may call, by name: registry tools except compensations, and
        declared runtime tools."""
        specs = [s for s in self.registry.specs() if s.name not in self.compensation_tools]
        return sorted([*specs, *self.runtime_tools], key=lambda s: s.name)

    def tool_effects(self) -> dict[str, str]:
        return {spec.name: spec.effect for spec in self.tool_specs()}


def load_bundle(path: str | Path) -> Bundle:
    root = Path(path).resolve()
    if not root.is_dir():
        raise BundleError(f"{root} is not a directory")
    config = _load_config(root)
    module, namespace = _import_tools(root)
    try:
        return _build(root, config, module, namespace)
    except BaseException:
        _unload(namespace)
        raise


def _build(root: Path, config: AgentConfig, module: ModuleType, namespace: str) -> Bundle:
    registry = getattr(module, "registry", None)
    if not isinstance(registry, ToolRegistry):
        raise BundleError(f"{root}/tools.py must define a module-level 'registry' ToolRegistry")
    try:
        registry.validate()
    except RegistryError as err:
        raise BundleError(f"tools.py: {err}") from err
    runtime_tools = tuple(RUNTIME_TOOLS[name] for name in config.runtime_tools)
    for spec in runtime_tools:
        if registry.get(spec.name) is not None:
            raise BundleError(f"tools.py: {spec.name!r} is a runtime tool and cannot be redefined")

    specs = sorted([*registry.specs(), *runtime_tools], key=lambda s: s.name)
    effects = {spec.name: spec.effect for spec in specs}
    compensations = frozenset(s.compensate for s in registry.specs() if s.compensate)
    docs = _load_policy_docs(root)
    for rel, doc in docs:
        _check_references(rel, doc, set(effects))
    try:
        policy = compile_policies(
            [doc for _, doc in docs], tool_effects=effects, compensation_only=compensations
        )
    except CompileError as err:
        raise BundleError(f"policies: {err}") from err

    return Bundle(
        bundle_id=_bundle_id(root, specs),
        root=root,
        config=config,
        policy=policy,
        policies_text="\n".join(
            f"## {rel}\n{yaml.safe_dump(doc, sort_keys=True).strip()}" for rel, doc in docs
        ),
        registry=registry,
        runtime_tools=runtime_tools,
        documents=_load_documents(root),
        tools_module=module,
        module_namespace=namespace,
        compensation_tools=compensations,
    )


def _load_yaml(path: Path, rel: str) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise BundleError(f"{rel}: invalid YAML: {err}") from err


def _load_config(root: Path) -> AgentConfig:
    path = root / "agent.yaml"
    if not path.is_file():
        raise BundleError(f"{root}: agent.yaml is missing")
    try:
        return parse_agent_config(_load_yaml(path, "agent.yaml"))
    except ConfigError as err:
        raise BundleError(str(err)) from err


def _import_tools(root: Path) -> tuple[ModuleType, str]:
    if not (root / "tools.py").is_file():
        raise BundleError(f"{root}: tools.py is missing")
    package = f"_switchboard_bundle_{next(_load_counter)}"
    spec = importlib.util.spec_from_loader(package, loader=None, is_package=True)
    assert spec is not None
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [str(root)]
    sys.modules[package] = pkg
    try:
        return importlib.import_module(f"{package}.tools"), package
    except Exception as err:
        _unload(package)
        raise BundleError(f"{root}/tools.py failed to import: {err}") from err


def _unload(package: str) -> None:
    if not package:
        return
    prefix = f"{package}."
    for name in [n for n in sys.modules if n == package or n.startswith(prefix)]:
        del sys.modules[name]


def _load_policy_docs(root: Path) -> list[tuple[str, Mapping[str, Any]]]:
    docs: list[tuple[str, Mapping[str, Any]]] = []
    for path in sorted((root / "policies").glob("*.yaml")):
        rel = path.relative_to(root).as_posix()
        doc = _load_yaml(path, rel)
        if doc is None:
            continue
        if not isinstance(doc, Mapping):
            raise BundleError(f"{rel}: a policy document must be a mapping")
        try:
            compile_policies([doc])
        except CompileError as err:
            raise BundleError(f"{rel}: {err}") from err
        docs.append((rel, doc))
    return docs


def _check_references(rel: str, doc: Mapping[str, Any], tools: set[str]) -> None:
    procedure = doc.get("procedure") or {}
    for section in ("require_before", "completes"):
        for tool in procedure.get(section) or {}:
            if tool not in tools:
                raise BundleError(f"{rel}: procedure.{section} names unknown tool {tool!r}")
    for rule in doc.get("rules") or []:
        rule_id = rule.get("id")
        when = rule.get("when") or {}
        named = when.get("tool")
        for tool in [named] if isinstance(named, str) else named or []:
            if tool not in tools:
                raise BundleError(f"{rel}: rule {rule_id!r} names unknown tool {tool!r}")
        # The document compiled on its own already, so the expression parses.
        policy = compile_policies([{"rules": [rule]}])
        for tool in _counted_tools(policy.rules[0].require):
            if tool not in tools:
                raise BundleError(f"{rel}: rule {rule_id!r} counts unknown tool {tool!r}")


def _counted_tools(expr: Expr) -> Iterator[str]:
    """Tool names used as `tool=...` filters in `count(tool.*, ...)` terms."""
    if isinstance(expr, Count):
        if expr.event_type.startswith("tool."):
            for key, value in expr.filters:
                if key == "tool" and isinstance(value, Literal) and isinstance(value.value, str):
                    yield value.value
    elif isinstance(expr, Compare):
        yield from _counted_tools(expr.left)
        yield from _counted_tools(expr.right)
    elif isinstance(expr, Not):
        yield from _counted_tools(expr.operand)
    elif isinstance(expr, And | Or):
        for operand in expr.operands:
            yield from _counted_tools(operand)
    elif isinstance(expr, ListExpr):
        for item in expr.items:
            yield from _counted_tools(item)


def _load_documents(root: Path) -> tuple[KnowledgeDoc, ...]:
    out: list[KnowledgeDoc] = []
    for path in sorted((root / "knowledge").glob("*.md")):
        body = path.read_text(encoding="utf-8")
        title = next(
            (line.lstrip("#").strip() for line in body.splitlines() if line.startswith("#")),
            path.stem,
        )
        out.append(KnowledgeDoc(path.relative_to(root).as_posix(), title, body))
    return tuple(out)


def _hashed_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        if path.is_file() and path.suffix != ".pyc":
            files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _bundle_id(root: Path, specs: list[ToolSpec]) -> str:
    digest = hashlib.sha256()
    for path in _hashed_files(root):
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\x00" + str(len(data)).encode() + b"\x00")
        digest.update(data)
    tools = [
        {
            "name": s.name,
            "description": s.description,
            "effect": s.effect,
            "params": s.params,
            "idempotency": list(s.idempotency) if s.idempotency is not None else None,
            "compensate": s.compensate,
            "timeout_s": s.timeout_s,
        }
        for s in specs
    ]
    digest.update(b"tools\x00" + canonical_json(tools).encode())
    return f"bnd_{digest.hexdigest()[:12]}"
