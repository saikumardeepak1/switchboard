"""In-memory stand-in for the delivery platform the tools talk to.

`DownstreamAPI` behaves like a real service that honours an `Idempotency-Key` header: the first
request with a key applies the effect, any later request with the same key gets the stored
response back. Tests read `requests` and `applied` to prove an effect landed exactly once.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Order:
    order_id: str
    customer_id: str
    courier_id: str
    status: str
    pin_address: str
    customer_name: str
    delivered_at: str | None = None


@dataclass
class DownstreamAPI:
    requests: list[tuple[str, str]] = field(default_factory=list)
    applied: Counter[str] = field(default_factory=Counter)
    _responses: dict[str, dict[str, Any]] = field(default_factory=dict)

    def call(self, op: str, key: str, apply: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        self.requests.append((op, key))
        if key in self._responses:
            return dict(self._responses[key])
        response = apply()
        self._responses[key] = dict(response)
        self.applied[op] += 1
        return response


@dataclass
class Backend:
    couriers: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    chats: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    refunds: dict[str, dict[str, Any]] = field(default_factory=dict)
    downstream: DownstreamAPI = field(default_factory=DownstreamAPI)


def seed() -> Backend:
    backend = Backend()
    backend.couriers["C-77"] = {
        "courier_id": "C-77",
        "name": "Sam Rivera",
        "week_total": 182.40,
        "deliveries": 23,
    }
    backend.orders["48213"] = Order(
        order_id="48213",
        customer_id="cust_ana",
        courier_id="C-77",
        status="out_for_delivery",
        pin_address="Harbor Office Park, 4th floor lobby",
        customer_name="Ana",
    )
    backend.chats["48213"] = [
        {"from": "customer", "text": "Please hand it to the front desk."},
        {"from": "courier", "text": "On my way."},
    ]
    return backend
