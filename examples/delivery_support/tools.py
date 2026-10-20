"""Tools for the delivery support agent. The runtime imports `registry` from this module."""

from __future__ import annotations

from typing import Any

from switchboard.effects import ToolRegistry

from .backend import seed

backend = seed()
registry = ToolRegistry()

_ORDER_ID = {"type": "string", "description": "Order number, digits only."}
_COURIER_ID = {"type": "string", "description": "Courier id such as C-77."}


@registry.tool(
    name="verify_courier",
    description="Verify the calling courier by id. Completes identity verification.",
    effect="read",
    params={
        "type": "object",
        "properties": {"courier_id": _COURIER_ID},
        "required": ["courier_id"],
    },
)
async def verify_courier(courier_id: str) -> dict[str, Any]:
    courier = backend.couriers.get(courier_id)
    if courier is None:
        return {"error": f"no courier {courier_id}"}
    return {"courier_id": courier_id, "name": courier["name"], "verified": True}


@registry.tool(
    name="lookup_order",
    description="Read an order, its delivery pin and the order chat between courier and customer.",
    effect="read",
    params={
        "type": "object",
        "properties": {"order_id": _ORDER_ID},
        "required": ["order_id"],
    },
)
async def lookup_order(order_id: str) -> dict[str, Any]:
    order = backend.orders.get(order_id)
    if order is None:
        return {"error": f"no order {order_id}"}
    chat = backend.chats.get(order_id, [])
    moved = any(
        m["from"] == "customer" and "address" in m["text"].lower() and "change" in m["text"].lower()
        for m in chat
    )
    return {
        "order_id": order.order_id,
        "status": order.status,
        "pin_address": order.pin_address,
        "customer_name": order.customer_name,
        "chat": chat,
        "evidence": {"chat_shows_address_change": moved},
    }


@registry.tool(
    name="mark_delivered",
    description="Mark an order delivered at the given location.",
    effect="write",
    params={
        "type": "object",
        "properties": {
            "order_id": _ORDER_ID,
            "location": {"type": "string", "description": "Where the order was handed over."},
        },
        "required": ["order_id", "location"],
    },
    idempotency=["order_id"],
    compensate="unmark_delivered",
)
async def mark_delivered(order_id: str, location: str, *, idempotency_key: str) -> dict[str, Any]:
    def apply() -> dict[str, Any]:
        order = backend.orders[order_id]
        order.status = "delivered"
        order.delivered_at = location
        return {"order_id": order_id, "status": "delivered", "location": location}

    if order_id not in backend.orders:
        return {"error": f"no order {order_id}"}
    return backend.downstream.call("mark_delivered", idempotency_key, apply)


@registry.tool(
    name="unmark_delivered",
    description="Undo mark_delivered. Run by the runtime as a compensation, never by the model.",
    effect="write",
    params={"type": "object", "properties": {"order_id": _ORDER_ID}, "required": ["order_id"]},
)
async def unmark_delivered(order_id: str, *, idempotency_key: str) -> dict[str, Any]:
    def apply() -> dict[str, Any]:
        order = backend.orders[order_id]
        order.status = "out_for_delivery"
        order.delivered_at = None
        return {"order_id": order_id, "status": order.status}

    return backend.downstream.call("unmark_delivered", idempotency_key, apply)


@registry.tool(
    name="issue_refund",
    description="Refund the customer for an order, in dollars.",
    effect="write",
    params={
        "type": "object",
        "properties": {"order_id": _ORDER_ID, "amount": {"type": "number", "minimum": 0}},
        "required": ["order_id", "amount"],
    },
    idempotency=["order_id", "amount"],
    compensate="reverse_refund",
)
async def issue_refund(order_id: str, amount: float, *, idempotency_key: str) -> dict[str, Any]:
    def apply() -> dict[str, Any]:
        refund_id = f"rf-{len(backend.refunds) + 1}"
        backend.refunds[refund_id] = {"order_id": order_id, "amount": amount, "reversed": False}
        return {"refund_id": refund_id, "order_id": order_id, "amount": amount}

    return backend.downstream.call("issue_refund", idempotency_key, apply)


@registry.tool(
    name="reverse_refund",
    description="Undo issue_refund. Run by the runtime as a compensation, never by the model.",
    effect="write",
    params={
        "type": "object",
        "properties": {"refund_id": {"type": "string"}, "order_id": _ORDER_ID},
        "required": ["refund_id", "order_id"],
    },
)
async def reverse_refund(refund_id: str, order_id: str, *, idempotency_key: str) -> dict[str, Any]:
    def apply() -> dict[str, Any]:
        backend.refunds[refund_id]["reversed"] = True
        return {"refund_id": refund_id, "reversed": True}

    return backend.downstream.call("reverse_refund", idempotency_key, apply)


@registry.tool(
    name="get_courier_pay",
    description="Read the courier's pay for the current week, in dollars.",
    effect="read",
    params={
        "type": "object",
        "properties": {"courier_id": _COURIER_ID},
        "required": ["courier_id"],
    },
)
async def get_courier_pay(courier_id: str) -> dict[str, Any]:
    courier = backend.couriers.get(courier_id)
    if courier is None:
        return {"error": f"no courier {courier_id}"}
    return {
        "courier_id": courier_id,
        "week_total": courier["week_total"],
        "deliveries": courier["deliveries"],
    }
