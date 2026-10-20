import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from switchboard.core.clock import VirtualClock
from switchboard.effects.ledger import IdempotencyLedger, InMemoryLedger, SqliteLedger

LedgerFactory = Callable[[Path, VirtualClock], IdempotencyLedger]


def _memory(_: Path, clock: VirtualClock) -> IdempotencyLedger:
    return InMemoryLedger(clock, lease_s=30.0)


def _sqlite(tmp: Path, clock: VirtualClock) -> IdempotencyLedger:
    return SqliteLedger(tmp / "ledger.sqlite3", clock, lease_s=30.0)


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock(start=100.0)


@pytest.fixture(params=[_memory, _sqlite], ids=["memory", "sqlite"])
async def ledger(
    request: pytest.FixtureRequest, tmp_path: Path, clock: VirtualClock
) -> AsyncIterator[IdempotencyLedger]:
    factory: LedgerFactory = request.param
    led = factory(tmp_path, clock)
    yield led
    await led.close()


async def test_first_claim_acquires(ledger: IdempotencyLedger) -> None:
    claim = await ledger.claim("k")
    assert claim.status == "acquired"
    assert claim.token is not None
    entry = await ledger.get("k")
    assert entry is not None and entry.state == "pending" and entry.attempts == 1


async def test_second_claim_while_pending_is_in_flight(ledger: IdempotencyLedger) -> None:
    await ledger.claim("k")
    again = await ledger.claim("k")
    assert again.status == "in_flight"
    assert again.token is None


async def test_concurrent_claims_have_one_winner(ledger: IdempotencyLedger) -> None:
    claims = await asyncio.gather(*(ledger.claim("k") for _ in range(10)))
    assert sorted(c.status for c in claims) == ["acquired"] + ["in_flight"] * 9


async def test_claim_after_done_returns_recorded_result(ledger: IdempotencyLedger) -> None:
    claim = await ledger.claim("k")
    assert claim.token is not None
    assert await ledger.record("k", claim.token, {"refund_id": "r1", "amount": 12.5})
    later = await ledger.claim("k")
    assert later.status == "done"
    assert later.result == {"refund_id": "r1", "amount": 12.5}


async def test_failed_claim_can_be_retried(ledger: IdempotencyLedger) -> None:
    first = await ledger.claim("k")
    assert first.token is not None
    assert await ledger.fail("k", first.token, "boom")
    entry = await ledger.get("k")
    assert entry is not None and entry.state == "failed" and entry.error == "boom"
    retry = await ledger.claim("k")
    assert retry.status == "acquired"
    assert retry.token != first.token
    entry = await ledger.get("k")
    assert entry is not None and entry.attempts == 2


async def test_stale_pending_claim_is_reclaimed_after_lease(
    ledger: IdempotencyLedger, clock: VirtualClock
) -> None:
    first = await ledger.claim("k")
    await clock.advance(29.9)
    assert (await ledger.claim("k")).status == "in_flight"
    await clock.advance(0.1)
    second = await ledger.claim("k")
    assert second.status == "acquired"
    assert second.token != first.token


async def test_stale_claimer_cannot_record_after_reclaim(
    ledger: IdempotencyLedger, clock: VirtualClock
) -> None:
    first = await ledger.claim("k")
    await clock.advance(31)
    second = await ledger.claim("k")
    assert first.token is not None and second.token is not None
    assert not await ledger.record("k", first.token, {"from": "first"})
    assert not await ledger.fail("k", first.token, "late")
    assert await ledger.record("k", second.token, {"from": "second"})
    assert (await ledger.claim("k")).result == {"from": "second"}


async def test_record_is_final(ledger: IdempotencyLedger) -> None:
    claim = await ledger.claim("k")
    assert claim.token is not None
    assert await ledger.record("k", claim.token, {"v": 1})
    assert not await ledger.record("k", claim.token, {"v": 2})
    assert not await ledger.fail("k", claim.token, "x")
    assert (await ledger.claim("k")).result == {"v": 1}


async def test_keys_are_independent(ledger: IdempotencyLedger) -> None:
    assert (await ledger.claim("a")).status == "acquired"
    assert (await ledger.claim("b")).status == "acquired"
    assert await ledger.get("c") is None


async def test_sqlite_ledger_survives_reopen(tmp_path: Path, clock: VirtualClock) -> None:
    path = tmp_path / "ledger.sqlite3"
    first = SqliteLedger(path, clock, lease_s=30)
    claim = await first.claim("k")
    assert claim.token is not None
    await first.record("k", claim.token, {"v": 1})
    await first.close()
    second = SqliteLedger(path, clock, lease_s=30)
    again = await second.claim("k")
    assert again.status == "done" and again.result == {"v": 1}
    await second.close()


async def test_sqlite_ledgers_on_one_file_share_claims(tmp_path: Path, clock: VirtualClock) -> None:
    path = tmp_path / "ledger.sqlite3"
    a = SqliteLedger(path, clock, lease_s=30)
    b = SqliteLedger(path, clock, lease_s=30)
    assert (await a.claim("k")).status == "acquired"
    assert (await b.claim("k")).status == "in_flight"
    await a.close()
    await b.close()


def test_lease_must_be_positive(clock: VirtualClock) -> None:
    with pytest.raises(ValueError, match="lease"):
        InMemoryLedger(clock, lease_s=0)
