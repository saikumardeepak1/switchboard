"""Tool registry, idempotency ledger, effect runner and compensations (ADR-009)."""

from switchboard.effects.keys import canonical_json, compensation_key, idempotency_key
from switchboard.effects.ledger import (
    ClaimResult,
    IdempotencyLedger,
    InMemoryLedger,
    LedgerEntry,
    SqliteLedger,
)
from switchboard.effects.runner import EffectContext, EffectRunner, ToolCall, ToolResult
from switchboard.effects.saga import CompensationOutcome, Saga, SagaReport
from switchboard.effects.spec import (
    RegisteredTool,
    RegistryError,
    ToolRegistry,
    ToolSpec,
    validate_args,
)

__all__ = [
    "ClaimResult",
    "CompensationOutcome",
    "EffectContext",
    "EffectRunner",
    "IdempotencyLedger",
    "InMemoryLedger",
    "LedgerEntry",
    "RegisteredTool",
    "RegistryError",
    "Saga",
    "SagaReport",
    "SqliteLedger",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "canonical_json",
    "compensation_key",
    "idempotency_key",
    "validate_args",
]
