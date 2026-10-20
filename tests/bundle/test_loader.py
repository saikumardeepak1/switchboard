import re
import shutil
from pathlib import Path

import pytest

from switchboard.bundle import CALL_PARTY, Bundle, BundleError, load_bundle
from switchboard.policy import CheckContext, ProposedCall

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "delivery_support"

TOOLS_PY = """
from typing import Any

from switchboard.effects import ToolRegistry

registry = ToolRegistry()


@registry.tool(
    name="lookup",
    description="Read a thing.",
    effect="read",
    params={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
)
async def lookup(id: str) -> dict[str, Any]:
    return {"id": id}


@registry.tool(
    name="charge",
    description="Charge a card.",
    effect="write",
    params={"type": "object", "properties": {"amount": {"type": "number"}}},
    compensate="refund",
)
async def charge(amount: float) -> dict[str, Any]:
    return {"charged": amount}


@registry.tool(
    name="refund",
    description="Undo a charge.",
    effect="write",
    params={"type": "object", "properties": {"amount": {"type": "number"}}},
)
async def refund(amount: float) -> dict[str, Any]:
    return {"refunded": amount}
"""

AGENT_YAML = """
name: mini
description: A small test agent.
system_prompt: Be brief.
"""

POLICY_YAML = """
procedure:
  steps: [verify, act]
  require_before:
    charge: [verify]
  completes:
    lookup: verify
rules:
  - id: small_charges
    when: {tool: charge}
    require: "args.amount <= 10"
"""


def write_bundle(root: Path, **overrides: str) -> Path:
    files = {
        "agent.yaml": AGENT_YAML,
        "tools.py": TOOLS_PY,
        "policies/main.yaml": POLICY_YAML,
        "knowledge/faq.md": "# Charges\n\nWe charge at most ten dollars.\n",
    }
    files.update(overrides)
    for rel, text in files.items():
        if text is None:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_loads_minimal_bundle(tmp_path: Path) -> None:
    bundle = load_bundle(write_bundle(tmp_path / "mini"))

    assert isinstance(bundle, Bundle)
    assert re.fullmatch(r"bnd_[0-9a-f]{12}", bundle.bundle_id)
    assert bundle.config.name == "mini"
    assert bundle.config.system_prompt == "Be brief."
    assert bundle.registry.names() == ["charge", "lookup", "refund"]
    assert [d.doc_id for d in bundle.documents] == ["knowledge/faq.md"]
    assert bundle.documents[0].title == "Charges"
    assert bundle.runtime_tools == ()
    assert bundle.config.max_policy_retries == 2


def test_policies_compile_with_the_tool_effect_map(tmp_path: Path) -> None:
    bundle = load_bundle(write_bundle(tmp_path / "mini"))

    assert dict(bundle.policy.tool_effects) == {
        "charge": "write",
        "lookup": "read",
        "refund": "write",
    }
    verdict = bundle.policy.check(ProposedCall("charge", {"amount": 50}), CheckContext())
    assert not verdict.allowed
    assert verdict.rule_ids == ["procedure:requires:verify", "small_charges"]


def test_bundle_id_is_stable_and_content_addressed(tmp_path: Path) -> None:
    first = load_bundle(write_bundle(tmp_path / "a"))
    second = load_bundle(write_bundle(tmp_path / "b"))
    assert first.bundle_id == second.bundle_id

    changed = load_bundle(write_bundle(tmp_path / "c", **{"agent.yaml": AGENT_YAML + "\n# x\n"}))
    assert changed.bundle_id != first.bundle_id


def test_bundle_id_ignores_caches_and_hidden_files(tmp_path: Path) -> None:
    root = write_bundle(tmp_path / "mini")
    before = load_bundle(root).bundle_id
    (root / "__pycache__").mkdir(exist_ok=True)
    (root / "__pycache__" / "tools.cpython-312.pyc").write_bytes(b"\x00junk")
    (root / ".DS_Store").write_bytes(b"junk")
    assert load_bundle(root).bundle_id == before


def test_each_load_imports_a_fresh_tools_module(tmp_path: Path) -> None:
    root = write_bundle(tmp_path / "mini")
    one, two = load_bundle(root), load_bundle(root)
    assert one.tools_module is not two.tools_module
    assert one.registry is not two.registry


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (
            "rules:\n  - id: r\n    when: {tool: nope}\n    require: 'true'\n",
            "rule 'r' names unknown tool 'nope'",
        ),
        (
            "procedure:\n  steps: [s]\n  require_before:\n    nope: [s]\n",
            "procedure.require_before names unknown tool 'nope'",
        ),
        (
            "procedure:\n  steps: [s]\n  completes:\n    nope: s\n",
            "procedure.completes names unknown tool 'nope'",
        ),
        (
            "rules:\n  - id: r\n    when: {tool: charge}\n"
            "    require: 'count(tool.executed, tool=nope) < 1'\n",
            "rule 'r' counts unknown tool 'nope'",
        ),
        (
            "rules:\n  - id: r\n    when: {tool: call_party}\n    require: 'true'\n",
            "rule 'r' names unknown tool 'call_party'",
        ),
    ],
)
def test_policy_references_must_resolve(tmp_path: Path, policy: str, message: str) -> None:
    root = write_bundle(tmp_path / "mini", **{"policies/main.yaml": policy})
    with pytest.raises(BundleError, match=re.escape(message)):
        load_bundle(root)


def test_compensation_targets_must_exist(tmp_path: Path) -> None:
    broken = TOOLS_PY.replace('compensate="refund"', 'compensate="void"')
    root = write_bundle(tmp_path / "mini", **{"tools.py": broken})
    with pytest.raises(BundleError, match="'charge' compensates with 'void'"):
        load_bundle(root)


def test_policy_compile_errors_name_the_file(tmp_path: Path) -> None:
    bad = "rules:\n  - id: r\n    when: {tool: charge}\n    require: 'args.amount <'\n"
    root = write_bundle(tmp_path / "mini", **{"policies/main.yaml": bad})
    with pytest.raises(BundleError, match=r"policies/main\.yaml"):
        load_bundle(root)


@pytest.mark.parametrize(
    ("agent", "message"),
    [
        ("description: no name\nsystem_prompt: x\n", "agent.yaml: missing 'name'"),
        ("name: a\n", "agent.yaml: missing 'system_prompt'"),
        ("name: a\nsystem_prompt: x\ncolour: red\n", "agent.yaml: unknown key 'colour'"),
        ("name: a\nsystem_prompt: x\nruntime_tools: [teleport]\n", "unknown runtime tool"),
        ("name: a\nsystem_prompt: x\neager_threshold: 2\n", "eager_threshold"),
        (
            "name: a\nsystem_prompt: x\nrouter:\n  rules:\n    - {name: r, tier: genius}\n",
            "tier",
        ),
        ("- just\n- a list\n", "agent.yaml must be a mapping"),
    ],
)
def test_agent_yaml_is_validated(tmp_path: Path, agent: str, message: str) -> None:
    root = write_bundle(tmp_path / "mini", **{"agent.yaml": agent})
    with pytest.raises(BundleError, match=re.escape(message)):
        load_bundle(root)


def test_tools_module_must_expose_a_registry(tmp_path: Path) -> None:
    root = write_bundle(tmp_path / "mini", **{"tools.py": "registry = 42\n"})
    with pytest.raises(BundleError, match="tools.py must define a module-level 'registry'"):
        load_bundle(root)


def test_missing_files_are_reported(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="not a directory"):
        load_bundle(tmp_path / "absent")
    root = write_bundle(tmp_path / "mini")
    (root / "agent.yaml").unlink()
    with pytest.raises(BundleError, match="agent.yaml is missing"):
        load_bundle(root)


def test_runtime_tools_join_the_effect_map_and_tool_specs(tmp_path: Path) -> None:
    agent = AGENT_YAML + "runtime_tools: [call_party]\nparties:\n  customer:\n    confirms: ok\n"
    policy = POLICY_YAML + "  - id: party_ok\n    when: {tool: call_party}\n    require: 'true'\n"
    root = write_bundle(tmp_path / "mini", **{"agent.yaml": agent, "policies/main.yaml": policy})
    bundle = load_bundle(root)
    assert bundle.runtime_tools == (CALL_PARTY,)
    assert bundle.policy.tool_effects["call_party"] == "write"
    assert bundle.config.parties["customer"].confirms == "ok"
    assert [s.name for s in bundle.tool_specs()] == ["call_party", "charge", "lookup", "refund"]


def test_parties_need_call_party(tmp_path: Path) -> None:
    agent = AGENT_YAML + "parties:\n  customer:\n    confirms: ok\n"
    root = write_bundle(tmp_path / "mini", **{"agent.yaml": agent})
    with pytest.raises(BundleError, match="'parties' needs runtime tool 'call_party'"):
        load_bundle(root)


def test_example_bundle_loads(tmp_path: Path) -> None:
    bundle = load_bundle(EXAMPLE)
    assert bundle.config.name == "delivery_support"
    assert "call_party" in bundle.policy.tool_effects
    assert bundle.config.parties["customer"].confirms == "customer_confirmed_address"
    assert bundle.config.router_rules[0].name == "pay_question"
    assert bundle.config.voice.chars_per_s == 16.0
    assert {d.doc_id for d in bundle.documents} == {"knowledge/delivery_policy.md"}

    copy = tmp_path / "copy"
    shutil.copytree(EXAMPLE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert load_bundle(copy).bundle_id == bundle.bundle_id
