import pytest

from switchboard.memory.store import (
    DEFAULT_TTL_S,
    MemoryKind,
    MemoryStore,
    MemoryWrite,
)

DAY = 86_400.0


def test_default_ttls_per_kind() -> None:
    assert DEFAULT_TTL_S == {
        MemoryKind.PROFILE: 365 * DAY,
        MemoryKind.OPEN_ISSUE: 14 * DAY,
        MemoryKind.PREFERENCE: 180 * DAY,
        MemoryKind.INTERACTION_SUMMARY: 90 * DAY,
    }


def test_write_then_read_returns_items_oldest_first() -> None:
    store = MemoryStore()
    a = store.write("t1", "c1", MemoryKind.PROFILE, {"name": "Ana"}, now=0.0)
    b = store.write("t1", "c1", MemoryKind.PREFERENCE, {"channel": "sms"}, now=5.0)
    assert store.read("t1", "c1", now=10.0) == [a, b]
    assert a.expires_at == 365 * DAY
    assert b.written_at == 5.0
    assert a.memory_id != b.memory_id


def test_expired_items_are_excluded_from_reads() -> None:
    store = MemoryStore()
    store.write("t1", "c1", MemoryKind.OPEN_ISSUE, {"issue": "late"}, now=0.0)
    store.write("t1", "c1", MemoryKind.PROFILE, {"name": "Ana"}, now=0.0)
    assert len(store.read("t1", "c1", now=14 * DAY - 1)) == 2
    kinds = [m.kind for m in store.read("t1", "c1", now=14 * DAY)]
    assert kinds == [MemoryKind.PROFILE]


def test_explicit_ttl_overrides_the_default() -> None:
    store = MemoryStore()
    store.write("t1", "c1", MemoryKind.PROFILE, {}, now=0.0, ttl_s=10.0)
    assert store.read("t1", "c1", now=9.0)
    assert store.read("t1", "c1", now=10.0) == []
    with pytest.raises(ValueError):
        store.write("t1", "c1", MemoryKind.PROFILE, {}, now=0.0, ttl_s=0.0)


def test_purge_expired_removes_rows() -> None:
    store = MemoryStore()
    store.write("t1", "c1", MemoryKind.OPEN_ISSUE, {}, now=0.0)
    store.write("t1", "c1", MemoryKind.PROFILE, {}, now=0.0)
    assert store.count("t1", "c1", now=15 * DAY) == 1
    assert store.count("t1", "c1") == 2
    assert store.purge_expired(now=15 * DAY) == 1
    assert store.count("t1", "c1") == 1


def test_reads_never_cross_tenants() -> None:
    store = MemoryStore()
    store.write("tenant_a", "cust_1", MemoryKind.PROFILE, {"name": "Ana"}, now=0.0)
    store.write("tenant_b", "cust_2", MemoryKind.PROFILE, {"name": "Bo"}, now=0.0)
    # Same customer id under another tenant is a different customer.
    assert store.read("tenant_b", "cust_1", now=1.0) == []
    assert store.read("tenant_a", "cust_2", now=1.0) == []
    (only,) = store.read("tenant_a", "cust_1", now=1.0)
    assert only.content == {"name": "Ana"}
    assert only.tenant_id == "tenant_a"


def test_key_parts_cannot_be_combined_to_collide() -> None:
    store = MemoryStore()
    store.write("a:b", "c", MemoryKind.PROFILE, {"who": 1}, now=0.0)
    assert store.read("a", "b:c", now=1.0) == []


def test_empty_scope_ids_are_rejected() -> None:
    store = MemoryStore()
    with pytest.raises(ValueError):
        store.write("", "c1", MemoryKind.PROFILE, {}, now=0.0)
    with pytest.raises(ValueError):
        store.read("t1", "", now=0.0)
    with pytest.raises(ValueError):
        store.forget("", "c1")


def test_forget_hard_deletes_everything_for_that_customer_only() -> None:
    store = MemoryStore()
    for kind in MemoryKind:
        store.write("t1", "c1", kind, {"k": kind.value}, now=0.0)
    store.write("t1", "c1", MemoryKind.OPEN_ISSUE, {}, now=0.0, ttl_s=1.0)
    store.write("t1", "c2", MemoryKind.PROFILE, {}, now=0.0)
    store.write("t2", "c1", MemoryKind.PROFILE, {}, now=0.0)

    assert store.forget("t1", "c1") == 5
    # Hard delete: no row remains, expired ones included.
    assert store.count("t1", "c1") == 0
    assert store.read("t1", "c1", now=0.5) == []
    assert store.count("t1", "c2") == 1
    assert store.count("t2", "c1") == 1
    assert store.forget("t1", "c1") == 0


def test_delete_kind_removes_only_that_kind() -> None:
    store = MemoryStore()
    store.write("t1", "c1", MemoryKind.OPEN_ISSUE, {"n": 1}, now=0.0)
    store.write("t1", "c1", MemoryKind.OPEN_ISSUE, {"n": 2}, now=1.0)
    store.write("t1", "c1", MemoryKind.PROFILE, {}, now=0.0)
    assert store.delete_kind("t1", "c1", MemoryKind.OPEN_ISSUE) == 2
    assert [m.kind for m in store.read("t1", "c1", now=2.0)] == [MemoryKind.PROFILE]


def test_capacity_evicts_oldest_interaction_summaries_first() -> None:
    store = MemoryStore(max_items_per_customer=3)
    store.write("t1", "c1", MemoryKind.PROFILE, {"p": 1}, now=0.0)
    store.write("t1", "c1", MemoryKind.INTERACTION_SUMMARY, {"n": 1}, now=1.0)
    store.write("t1", "c1", MemoryKind.INTERACTION_SUMMARY, {"n": 2}, now=2.0)
    store.write("t1", "c1", MemoryKind.INTERACTION_SUMMARY, {"n": 3}, now=3.0)
    contents = [m.content for m in store.read("t1", "c1", now=4.0)]
    assert contents == [{"p": 1}, {"n": 2}, {"n": 3}]
    assert store.count("t1", "c1") == 3


def test_capacity_drops_expired_before_evicting_live_items() -> None:
    store = MemoryStore(max_items_per_customer=2)
    store.write("t1", "c1", MemoryKind.INTERACTION_SUMMARY, {"n": 1}, now=0.0)
    store.write("t1", "c1", MemoryKind.OPEN_ISSUE, {"o": 1}, now=0.0, ttl_s=1.0)
    store.write("t1", "c1", MemoryKind.PREFERENCE, {"q": 1}, now=5.0)
    assert [m.content for m in store.read("t1", "c1", now=6.0)] == [{"n": 1}, {"q": 1}]


def test_capacity_falls_back_to_the_oldest_item_when_no_summaries_remain() -> None:
    store = MemoryStore(max_items_per_customer=2)
    store.write("t1", "c1", MemoryKind.PREFERENCE, {"a": 1}, now=0.0)
    store.write("t1", "c1", MemoryKind.PREFERENCE, {"a": 2}, now=1.0)
    store.write("t1", "c1", MemoryKind.PREFERENCE, {"a": 3}, now=2.0)
    assert [m.content for m in store.read("t1", "c1", now=3.0)] == [{"a": 2}, {"a": 3}]


def test_apply_runs_writes_and_deletes_in_order() -> None:
    store = MemoryStore()
    store.write("t1", "c1", MemoryKind.OPEN_ISSUE, {"old": True}, now=0.0)
    store.apply(
        [
            MemoryWrite.delete("t1", "c1", MemoryKind.OPEN_ISSUE),
            MemoryWrite.put("t1", "c1", MemoryKind.INTERACTION_SUMMARY, {"text": "x"}, at=5.0),
        ]
    )
    (item,) = store.read("t1", "c1", now=6.0)
    assert item.kind is MemoryKind.INTERACTION_SUMMARY
    assert item.written_at == 5.0


def test_memory_ids_are_deterministic() -> None:
    def run() -> list[str]:
        s = MemoryStore()
        return [
            s.write("t1", "c1", MemoryKind.PROFILE, {}, now=0.0).memory_id,
            s.write("t1", "c1", MemoryKind.PROFILE, {}, now=0.0).memory_id,
        ]

    assert run() == run()
