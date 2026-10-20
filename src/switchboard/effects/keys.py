"""Deterministic idempotency keys (ADR-009).

A key names one intent to cause a side effect. It is derived from:

- `conversation_id`, so identical calls in two conversations never collide.
- `step_seq`, the conversation step that proposed the call. The same refund issued twice in
  one conversation at different steps is two intents and must run twice; a retry or replay of
  the same step is one intent and must run once. Without the step, a legitimate second refund
  would be swallowed as a duplicate. With a random id per attempt instead, retries would never
  dedupe at all, which is the double-refund this module exists to prevent.
- the tool name, which also prefixes the key so ledgers stay readable.
- the canonical form of the identity arguments.

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
    step_seq: int,
    tool: str,
    args: Mapping[str, Any],
    identity_fields: Sequence[str] | None,
) -> str:
    """Key for one intent. `identity_fields` None means every argument defines the intent."""
    fields = sorted(args) if identity_fields is None else sorted(identity_fields)
    identity = {name: args[name] for name in fields if name in args}
    payload = {"conversation": conversation_id, "step": step_seq, "tool": tool, "args": identity}
    return f"{tool}:{_digest(payload)}"


def compensation_key(original_key: str, compensation_tool: str) -> str:
    """Key for undoing `original_key`. One original effect is compensated at most once."""
    payload = {"compensates": original_key, "tool": compensation_tool}
    return f"{compensation_tool}:{_digest(payload)}"
