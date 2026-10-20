from typing import Any

import pytest

from switchboard.effects.spec import RegistryError, ToolRegistry, ToolSpec, validate_args

REFUND_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "amount": {"type": "number", "minimum": 0, "maximum": 25},
        "count": {"type": "integer", "minimum": 1},
        "reason": {"type": "string", "enum": ["late", "damaged"]},
        "notify": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "address": {
            "type": "object",
            "properties": {"zip": {"type": "string"}},
            "required": ["zip"],
        },
    },
    "required": ["order_id", "amount"],
}


def spec(params: dict[str, Any] = REFUND_PARAMS) -> ToolSpec:
    return ToolSpec(name="issue_refund", description="d", effect="write", params=params)


@pytest.mark.parametrize(
    ("args", "errors"),
    [
        ({"order_id": "48213", "amount": 12.5}, []),
        ({"order_id": "48213", "amount": 12}, []),
        ({"order_id": "1", "amount": 1, "count": 2.0}, []),
        ({"amount": 1}, ["args.order_id: required property is missing"]),
        ({"order_id": 48213, "amount": 1}, ["args.order_id: expected string, got integer"]),
        ({"order_id": "1", "amount": "12"}, ["args.amount: expected number, got string"]),
        ({"order_id": "1", "amount": True}, ["args.amount: expected number, got boolean"]),
        ({"order_id": "1", "amount": 26}, ["args.amount: 26 is above the maximum of 25"]),
        ({"order_id": "1", "amount": -1}, ["args.amount: -1 is below the minimum of 0"]),
        (
            {"order_id": "1", "amount": 1, "count": 1.5},
            ["args.count: expected integer, got number"],
        ),
        ({"order_id": "1", "amount": 1, "count": 0}, ["args.count: 0 is below the minimum of 1"]),
        (
            {"order_id": "1", "amount": 1, "reason": "angry"},
            ["args.reason: 'angry' is not one of ['late', 'damaged']"],
        ),
        (
            {"order_id": "1", "amount": 1, "notify": "yes"},
            ["args.notify: expected boolean, got string"],
        ),
        (
            {"order_id": "1", "amount": 1, "tags": ["a", 2]},
            ["args.tags[1]: expected string, got integer"],
        ),
        ({"order_id": "1", "amount": 1, "tags": "a"}, ["args.tags: expected array, got string"]),
        (
            {"order_id": "1", "amount": 1, "address": {}},
            ["args.address.zip: required property is missing"],
        ),
        ({"order_id": "1", "amount": 1, "coupon": "X"}, ["args.coupon: unexpected property"]),
        (
            {"amount": "x", "extra": 1},
            [
                "args.amount: expected number, got string",
                "args.extra: unexpected property",
                "args.order_id: required property is missing",
            ],
        ),
    ],
)
def test_validate_args(args: dict[str, Any], errors: list[str]) -> None:
    assert validate_args(spec(), args) == errors


def test_additional_properties_true_allows_unknown_args() -> None:
    params = {"type": "object", "properties": {}, "additionalProperties": True}
    assert validate_args(spec(params), {"anything": 1}) == []


def test_args_must_be_an_object() -> None:
    assert validate_args(spec(), [1, 2]) == ["args: expected object, got array"]  # type: ignore[arg-type]


def test_spec_rejects_unsupported_schema_type() -> None:
    with pytest.raises(ValueError, match="unsupported type 'date'"):
        spec({"type": "object", "properties": {"when": {"type": "date"}}})


def test_spec_rejects_identity_field_not_in_params() -> None:
    with pytest.raises(ValueError, match="idempotency field 'nope'"):
        ToolSpec(
            name="t", description="d", effect="write", params=REFUND_PARAMS, idempotency=["nope"]
        )


def test_spec_rejects_compensate_on_read_tool() -> None:
    with pytest.raises(ValueError, match="read tool"):
        ToolSpec(name="t", description="d", effect="read", params=REFUND_PARAMS, compensate="undo")


def test_identity_fields_default_to_all_args() -> None:
    assert spec().identity_fields({"order_id": "1", "amount": 2}) == ["amount", "order_id"]
    narrow = ToolSpec(
        name="t", description="d", effect="write", params=REFUND_PARAMS, idempotency=["order_id"]
    )
    assert narrow.identity_fields({"order_id": "1", "amount": 2}) == ["order_id"]


async def _noop(**_: Any) -> dict[str, Any]:
    return {}


def test_registry_registers_with_decorator() -> None:
    registry = ToolRegistry()

    @registry.tool(name="lookup_order", description="d", effect="read", params=REFUND_PARAMS)
    async def lookup_order(**_: Any) -> dict[str, Any]:
        return {}

    assert registry.get("lookup_order") is not None
    assert registry.get("lookup_order").spec.effect == "read"  # type: ignore[union-attr]
    assert registry.get("missing") is None
    assert registry.names() == ["lookup_order"]


def test_registry_rejects_duplicates() -> None:
    registry = ToolRegistry()
    registry.tool(name="t", description="d", effect="read", params=REFUND_PARAMS)(_noop)
    with pytest.raises(RegistryError, match="already registered"):
        registry.tool(name="t", description="d", effect="read", params=REFUND_PARAMS)(_noop)


def test_registry_rejects_sync_callables() -> None:
    registry = ToolRegistry()

    def sync(**_: Any) -> dict[str, Any]:
        return {}

    with pytest.raises(RegistryError, match="async"):
        registry.tool(name="t", description="d", effect="read", params=REFUND_PARAMS)(sync)  # type: ignore[arg-type]


def test_validate_requires_compensation_target_to_exist() -> None:
    registry = ToolRegistry()
    registry.tool(name="t", description="d", effect="write", params=REFUND_PARAMS, compensate="u")(
        _noop
    )
    with pytest.raises(RegistryError, match="'u', which is not registered"):
        registry.validate()


def test_validate_requires_compensation_target_to_be_a_write_tool() -> None:
    registry = ToolRegistry()
    registry.tool(name="t", description="d", effect="write", params=REFUND_PARAMS, compensate="u")(
        _noop
    )
    registry.tool(name="u", description="d", effect="read", params=REFUND_PARAMS)(_noop)
    with pytest.raises(RegistryError, match="'u' is a read tool"):
        registry.validate()


def test_validate_rejects_self_compensation() -> None:
    registry = ToolRegistry()
    registry.tool(name="t", description="d", effect="write", params=REFUND_PARAMS, compensate="t")(
        _noop
    )
    with pytest.raises(RegistryError, match="compensate itself"):
        registry.validate()


def test_validate_passes_for_consistent_registry() -> None:
    registry = ToolRegistry()
    registry.tool(name="t", description="d", effect="write", params=REFUND_PARAMS, compensate="u")(
        _noop
    )
    registry.tool(name="u", description="d", effect="write", params=REFUND_PARAMS)(_noop)
    registry.validate()
