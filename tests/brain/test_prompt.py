import hashlib

from switchboard.brain.prompt import VolatileContext, assemble_prompt, canonical_json
from switchboard.brain.types import Message, ToolSchema

LOOKUP = ToolSchema(
    name="lookup_order",
    description="Look up an order.",
    parameters={"type": "object", "properties": {"order_id": {"type": "string"}}},
)
REFUND = ToolSchema(
    name="issue_refund",
    description="Refund an order.",
    parameters={"properties": {"amount": {"type": "number"}}, "type": "object"},
)
SYSTEM = "You are a delivery support agent."
POLICIES = "Refunds above $25 need a human."


def volatile_a() -> VolatileContext:
    return VolatileContext(
        evidence={"eta": "19:45", "order_id": "48213"},
        memory=("prefers text updates",),
        knowledge=("Storm delays in Seattle",),
        open_steps=("verify_identity",),
        messages=(Message(role="user", content="where is my order"),),
    )


def volatile_b() -> VolatileContext:
    return VolatileContext(
        evidence={"refund_amount": 12.5},
        open_steps=("resolve", "close"),
        messages=(
            Message(role="user", content="where is my order"),
            Message(role="assistant", content="It arrives at 7:45 pm."),
            Message(role="user", content="can I get a refund"),
        ),
    )


def test_prefix_is_byte_identical_across_volatile_contexts_and_tool_order() -> None:
    a = assemble_prompt(SYSTEM, POLICIES, [LOOKUP, REFUND], volatile_a())
    b = assemble_prompt(SYSTEM, POLICIES, [REFUND, LOOKUP], volatile_b())
    assert a.prefix_hash == b.prefix_hash
    assert a.messages[: a.prefix_len] == b.messages[: b.prefix_len]
    assert a.messages[0].content.encode() == b.messages[0].content.encode()
    assert a.messages != b.messages


def test_prefix_hash_is_sha256_of_prefix_bytes() -> None:
    p = assemble_prompt(SYSTEM, POLICIES, [LOOKUP], VolatileContext())
    prefix_bytes = "".join(m.content for m in p.messages[: p.prefix_len]).encode()
    assert p.prefix_hash == hashlib.sha256(prefix_bytes).hexdigest()


def test_prefix_changes_when_bundle_changes() -> None:
    base = assemble_prompt(SYSTEM, POLICIES, [LOOKUP], VolatileContext()).prefix_hash
    assert assemble_prompt(SYSTEM + "!", POLICIES, [LOOKUP], VolatileContext()).prefix_hash != base
    assert (
        assemble_prompt(SYSTEM, POLICIES, [LOOKUP, REFUND], VolatileContext()).prefix_hash != base
    )


def test_tools_sorted_by_name_and_canonical() -> None:
    p = assemble_prompt(SYSTEM, POLICIES, [LOOKUP, REFUND], VolatileContext())
    assert [t.name for t in p.tools] == ["issue_refund", "lookup_order"]
    prefix = p.messages[0].content
    assert prefix.index('"issue_refund"') < prefix.index('"lookup_order"')
    # REFUND was declared with "properties" before "type"; canonical form sorts keys
    assert canonical_json(REFUND.parameters) in prefix
    assert canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == '{"a":[1,{"c":3,"d":2}],"b":1}'


def test_volatile_context_comes_after_prefix_then_conversation() -> None:
    p = assemble_prompt(SYSTEM, POLICIES, [LOOKUP], volatile_b())
    assert p.prefix_len == 1
    assert p.messages[0].role == "system"
    context = p.messages[1]
    assert context.role == "system"
    assert '"refund_amount":12.5' in context.content
    assert "resolve" in context.content
    assert list(p.messages[2:]) == list(volatile_b().messages)
    assert SYSTEM not in context.content


def test_volatile_context_is_deterministic_for_same_input() -> None:
    v1 = VolatileContext(evidence={"b": 2, "a": 1})
    v2 = VolatileContext(evidence={"a": 1, "b": 2})
    assert (
        assemble_prompt(SYSTEM, POLICIES, [], v1).messages
        == assemble_prompt(SYSTEM, POLICIES, [], v2).messages
    )


def test_empty_volatile_context_omits_context_message() -> None:
    p = assemble_prompt(SYSTEM, POLICIES, [], VolatileContext(messages=(Message("user", "hi"),)))
    assert [m.role for m in p.messages] == ["system", "user"]
