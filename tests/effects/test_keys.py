import subprocess
import sys
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from switchboard.effects import EffectContext
from switchboard.effects.keys import canonical_json, compensation_key, idempotency_key


def key(args: dict[str, Any], fields: list[str] | None = None, intent: int = 3) -> str:
    return idempotency_key("conv_1", intent, "issue_refund", args, fields)


def test_key_is_prefixed_with_tool_name_and_sha256() -> None:
    k = key({"order_id": "1"})
    prefix, digest = k.split(":")
    assert prefix == "issue_refund"
    assert len(digest) == 64
    int(digest, 16)


def test_dict_ordering_does_not_change_key() -> None:
    a = key({"order_id": "1", "amount": 5, "meta": {"x": 1, "y": 2}})
    b = key({"meta": {"y": 2, "x": 1}, "amount": 5, "order_id": "1"})
    assert a == b


def test_integral_float_equals_int() -> None:
    assert key({"amount": 1.0}) == key({"amount": 1})
    assert key({"amount": -0.0}) == key({"amount": 0})
    assert key({"amount": 1.5}) != key({"amount": 1})


def test_bool_is_not_confused_with_int() -> None:
    assert key({"flag": True}) != key({"flag": 1})


def test_canonical_json_has_no_whitespace_and_sorted_keys() -> None:
    assert canonical_json({"b": [1.0, 2.5], "a": {"d": None, "c": "x"}}) == (
        '{"a":{"c":"x","d":null},"b":[1,2.5]}'
    )


def test_canonical_json_rejects_non_finite_floats() -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json({"a": float("nan")})


def test_the_intent_counter_separates_intents() -> None:
    assert key({"order_id": "1"}, intent=3) != key({"order_id": "1"}, intent=4)


def test_the_key_does_not_depend_on_which_step_proposed_the_call() -> None:
    # Nothing but conversation, intent, tool and identity args goes in: the same call proposed
    # at a later step of the same intent gets the same key and replays.
    ctx_a = EffectContext("conv_1", intent=2)
    ctx_b = EffectContext("conv_1", intent=2)
    assert ctx_a == ctx_b
    assert idempotency_key("conv_1", 2, "t", {"x": 1}, None) == idempotency_key(
        "conv_1", 2, "t", {"x": 1}, None
    )


def test_conversation_and_tool_separate_keys() -> None:
    base = idempotency_key("conv_1", 1, "a", {"x": 1}, None)
    assert base != idempotency_key("conv_2", 1, "a", {"x": 1}, None)
    assert base != idempotency_key("conv_1", 1, "b", {"x": 1}, None)


def test_identity_fields_limit_what_defines_the_intent() -> None:
    fields = ["order_id"]
    assert key({"order_id": "1", "note": "a"}, fields) == key(
        {"order_id": "1", "note": "b"}, fields
    )
    assert key({"order_id": "1", "note": "a"}) != key({"order_id": "1", "note": "b"})


def test_key_is_stable_across_processes() -> None:
    code = (
        "from switchboard.effects.keys import idempotency_key;"
        "print(idempotency_key('conv_1', 3, 'issue_refund', {'b': 1.0, 'a': [True, None]}, None))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == key({"a": [True, None], "b": 1})


def test_compensation_key_is_derived_and_distinct() -> None:
    original = key({"order_id": "1"})
    first = compensation_key(original, "void_refund")
    assert first == compensation_key(original, "void_refund")
    assert first.startswith("void_refund:")
    assert first != compensation_key(key({"order_id": "2"}), "void_refund")


json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=20,
)


@given(st.dictionaries(st.text(), json_values))
def test_key_ignores_insertion_order(args: dict[str, Any]) -> None:
    reversed_args = dict(reversed(list(args.items())))
    assert key(args) == key(reversed_args)
