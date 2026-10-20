"""Deterministic idempotency keys (ADR-009).

A key names one intent to cause a side effect. It is derived from:

- `conversation_id`, so identical calls in two conversations never collide.
- the tool name, which also prefixes the key so ledgers stay readable.
- the canonical form of the identity arguments.
- `intent`, an explicit counter per tool that the policy layer moves only when the user makes a
  new request for that tool (`switchboard.policy.RequestDetector`, recorded on
  `turn.committed` and folded into conversation state).

Why an intent counter and not the step that proposed the call. The first version keyed on the
proposing step's sequence number. That dedupes retries and crash replays of one proposal, but a
model that proposes the same `mark_delivered` again in a later turn ("thanks, is it marked
now?") gets a new step, a new key, and a second downstream write. Calls do get re-proposed
across turns; models are not reliable about remembering they already acted. With the counter,
re-proposals within one request share a key and replay, while a real second request ("refund
another ten dollars") moves the counter and runs. The counter lives in the log, so recovery
derives the same key. A random id per attempt would never dedupe at all, which is the
double-refund this module exists to prevent; what counts as "the same request" is a policy
question, so the policy layer answers it, not the model.

Canonical JSON sorts keys, drops whitespace, writes integral floats as integers (so `1.0` and
`1` agree, the way they do once they have passed through any JSON hop), applies Unicode NFC to
strings, and rejects NaN and infinity. `bool` stays distinct from `int`. The result depends only
on its inputs, so every process derives the same key.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot canonicalize non-finite float {value!r}")
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"cannot canonicalize non-string key {k!r}")
            out[unicodedata.normalize("NFC", k)] = _normalize(v)
        return out
    if isinstance(value, list | tuple):
        return [_normalize(v) for v in value]
    raise ValueError(f"cannot canonicalize value of type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def idempotency_key(
    conversation_id: str,
    intent: int,
    tool: str,
    args: Mapping[str, Any],
    identity_fields: Sequence[str] | None,
) -> str:
    """Key for one intent. `identity_fields` None means every argument defines the intent."""
    fields = sorted(args) if identity_fields is None else sorted(identity_fields)
    identity = {name: args[name] for name in fields if name in args}
    payload = {"conversation": conversation_id, "intent": intent, "tool": tool, "args": identity}
    return f"{tool}:{_digest(payload)}"


def compensation_key(original_key: str, compensation_tool: str) -> str:
    """Key for undoing `original_key`. One original effect is compensated at most once."""
    payload = {"compensates": original_key, "tool": compensation_tool}
    return f"{compensation_tool}:{_digest(payload)}"
