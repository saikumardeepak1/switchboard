"""Per-tenant concurrent conversation caps, checked when a call is set up (ADR-013).

A call holds its slot for minutes while using few tokens, so a token budget alone does not
protect capacity. Slots are counted per tenant and released when the conversation ends.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: str | None
    active: int
    cap: int


class AdmissionRejected(Exception):
    def __init__(self, decision: AdmissionDecision) -> None:
        self.decision = decision
        super().__init__(decision.reason)


class AdmissionController:
    def __init__(self, tenant_caps: Mapping[str, int] | None = None) -> None:
        """`tenant_caps` overrides the cap a bundle declares, per tenant."""
        for tenant, cap in (tenant_caps or {}).items():
            if cap < 1:
                raise ValueError(f"cap for tenant {tenant!r} must be at least 1")
        self._caps = dict(tenant_caps or {})
        self._active: dict[str, set[str]] = {}

    def active(self, tenant_id: str) -> int:
        return len(self._active.get(tenant_id, ()))

    def admit(self, tenant_id: str, conversation_id: str, default_cap: int) -> AdmissionDecision:
        cap = self._caps.get(tenant_id, default_cap)
        held = self._active.setdefault(tenant_id, set())
        if conversation_id in held:
            return AdmissionDecision(True, None, len(held), cap)
        if len(held) >= cap:
            reason = (
                f"tenant {tenant_id!r} is at its cap of {cap} concurrent conversations"
                f" ({len(held)} active)"
            )
            return AdmissionDecision(False, reason, len(held), cap)
        held.add(conversation_id)
        return AdmissionDecision(True, None, len(held), cap)

    def release(self, tenant_id: str, conversation_id: str) -> None:
        held = self._active.get(tenant_id)
        if held is not None:
            held.discard(conversation_id)
