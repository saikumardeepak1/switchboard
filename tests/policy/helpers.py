"""Shared builders for policy tests."""

from pathlib import Path
from typing import Any

import yaml

from switchboard.core.events import AnyEvent, ProcedureAdvanced, ToolExecuted, ToolProposed

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    data = yaml.safe_load((FIXTURES / name).read_text())
    assert isinstance(data, dict)
    return data


def _base(seq: int) -> dict[str, Any]:
    return {
        "conversation_id": "conv_1",
        "leg_id": "leg_1",
        "seq": seq,
        "at": float(seq),
        "bundle_id": "bundle_abc",
        "event_id": f"evt_{seq}",
    }


def executed(
    tool: str, seq: int = 1, result: dict[str, Any] | None = None, call_id: str = "call_1"
) -> AnyEvent:
    return ToolExecuted(**_base(seq), call_id=call_id, tool=tool, result=result or {"ok": True})


def proposed(tool: str, args: dict[str, Any], seq: int = 1) -> AnyEvent:
    return ToolProposed(**_base(seq), call_id=f"call_{seq}", tool=tool, args=args)


def advanced(step: str, seq: int = 1) -> AnyEvent:
    return ProcedureAdvanced(**_base(seq), step=step, state={})
