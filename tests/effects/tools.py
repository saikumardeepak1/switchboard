"""A small registry of instrumented tools shared by the runner and saga tests."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from switchboard.core.clock import VirtualClock
from switchboard.effects.spec import ToolRegistry


@dataclass
class Bench:
    clock: VirtualClock
    registry: ToolRegistry
    calls: Counter[str] = field(default_factory=Counter)
    # Tool name -> how many upcoming calls should raise.
    fail_next: Counter[str] = field(default_factory=Counter)
    delay_s: dict[str, float] = field(default_factory=dict)
    cancelled: list[str] = field(default_factory=list)
    log: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def enter(self, name: str, args: dict[str, Any]) -> None:
        self.calls[name] += 1
        self.log.append((name, args))
        delay = self.delay_s.get(name, 0.0)
        if delay:
            try:
                await self.clock.sleep(delay)
            except BaseException:
                self.cancelled.append(name)
                raise
        if self.fail_next[name] > 0:
            self.fail_next[name] -= 1
            raise RuntimeError(f"{name} exploded")


def build_bench(clock: VirtualClock) -> Bench:
    registry = ToolRegistry()
    bench = Bench(clock=clock, registry=registry)

    @registry.tool(
        name="lookup_order",
        description="Read an order.",
        effect="read",
        params={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        timeout_s=2.0,
    )
    async def lookup_order(order_id: str) -> dict[str, Any]:
        await bench.enter("lookup_order", {"order_id": order_id})
        return {"order_id": order_id, "status": "late"}

    @registry.tool(
        name="issue_refund",
        description="Refund an order.",
        effect="write",
        params={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number", "minimum": 0},
                "note": {"type": "string"},
            },
            "required": ["order_id", "amount"],
        },
        idempotency=["order_id", "amount"],
        compensate="void_refund",
        timeout_s=5.0,
    )
    async def issue_refund(order_id: str, amount: float, note: str = "") -> dict[str, Any]:
        await bench.enter("issue_refund", {"order_id": order_id, "amount": amount})
        n = bench.calls["issue_refund"]
        return {"refund_id": f"ref_{n}", "amount": amount}

    @registry.tool(
        name="void_refund",
        description="Void a refund.",
        effect="write",
        params={
            "type": "object",
            "properties": {"refund_id": {"type": "string"}, "order_id": {"type": "string"}},
            "required": ["refund_id", "order_id"],
        },
        timeout_s=5.0,
    )
    async def void_refund(refund_id: str, order_id: str) -> dict[str, Any]:
        await bench.enter("void_refund", {"refund_id": refund_id, "order_id": order_id})
        return {"voided": refund_id}

    @registry.tool(
        name="hold_slot",
        description="Hold a delivery slot.",
        effect="write",
        params={
            "type": "object",
            "properties": {"slot": {"type": "string"}},
            "required": ["slot"],
        },
        compensate="release_slot",
    )
    async def hold_slot(slot: str) -> dict[str, Any]:
        await bench.enter("hold_slot", {"slot": slot})
        return {"hold_id": f"hold_{slot}"}

    @registry.tool(
        name="release_slot",
        description="Release a held slot.",
        effect="write",
        params={
            "type": "object",
            "properties": {"hold_id": {"type": "string"}},
            "required": ["hold_id"],
        },
    )
    async def release_slot(hold_id: str) -> dict[str, Any]:
        await bench.enter("release_slot", {"hold_id": hold_id})
        return {"released": hold_id}

    @registry.tool(
        name="send_sms",
        description="Send a text. Cannot be unsent.",
        effect="write",
        params={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    async def send_sms(text: str) -> dict[str, Any]:
        await bench.enter("send_sms", {"text": text})
        return {"sent": True}

    @registry.tool(
        name="bad_result",
        description="Returns something that is not a dict.",
        effect="write",
        params={"type": "object", "properties": {}},
    )
    async def bad_result() -> dict[str, Any]:
        await bench.enter("bad_result", {})
        return ["not", "a", "dict"]  # type: ignore[return-value]

    return bench
