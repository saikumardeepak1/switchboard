"""Per-tenant concurrent conversation caps and token-per-minute budgets (ADR-013).

A call holds its slot for minutes while using few tokens, so a token budget alone does not
protect capacity. Slots are counted per tenant and released when the conversation ends.

The token budget is separate: model usage is recorded per tenant with the clock time it was
spent, and a tenant whose last 60 seconds exceed its budget is over budget until old usage slides
out of the window. The runtime then routes that tenant's turns to the talker tier. Time is passed
in, so this stays pure.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

WINDOW_S = 60.0


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
    def __init__(
        self,
        tenant_caps: Mapping[str, int] | None = None,
        *,
        token_budgets: Mapping[str, int] | None = None,
    ) -> None:
        """`tenant_caps` and `token_budgets` override what a bundle declares, per tenant."""
        for tenant, cap in (tenant_caps or {}).items():
            if cap < 1:
                raise ValueError(f"cap for tenant {tenant!r} must be at least 1")
        for tenant, budget in (token_budgets or {}).items():
            if budget < 1:
                raise ValueError(f"token budget for tenant {tenant!r} must be at least 1")
        self._caps = dict(tenant_caps or {})
        self._budgets = dict(token_budgets or {})
        self._active: dict[str, set[str]] = {}
        self._spent: dict[str, deque[tuple[float, int]]] = {}
        self._totals: dict[str, int] = {}

    def record_tokens(self, tenant_id: str, tokens: int, *, now: float) -> None:
        if tokens > 0:
            self._spent.setdefault(tenant_id, deque()).append((now, tokens))
            self._totals[tenant_id] = self._totals.get(tenant_id, 0) + tokens

    def tokens_used(self, tenant_id: str, *, now: float) -> int:
        """Tokens the tenant spent in the last `WINDOW_S` seconds."""
        spent = self._spent.get(tenant_id)
        while spent and spent[0][0] <= now - WINDOW_S:
            _, tokens = spent.popleft()
            self._totals[tenant_id] -= tokens
        return self._totals.get(tenant_id, 0)

    def over_budget(self, tenant_id: str, *, now: float, default_budget: int | None = None) -> bool:
        budget = self._budgets.get(tenant_id, default_budget)
        return budget is not None and self.tokens_used(tenant_id, now=now) >= budget

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
