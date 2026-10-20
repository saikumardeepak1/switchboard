"""One smoke test per benchmark at a tiny sample size, in process."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks import (
    common,
    cost,
    endpointing,
    grounding_gate,
    idempotency,
    multiparty,
    policy_gate,
    run_all,
    speculation_latency,
)


def test_policy_gate_blocks_before_execution_and_the_baseline_does_not() -> None:
    data = policy_gate.run(n=6, workers=1)
    r = data["results"]
    a, b = r["baseline_post_hoc"], r["switchboard_pre_execution"]
    assert a["violating_proposed"] > 0 and b["violating_proposed"] > 0
    assert a["violating_executed"] == a["violating_proposed"]
    assert b["violating_executed_by_kind"].get("refund_over_limit", 0) == 0
    assert b["false_blocks"] == 0 and b["verdict_replay_mismatches"] == 0
    assert b["check_latency_us"]["p99"] is not None
    assert "baseline_post_hoc" in policy_gate.render(data)


def test_grounding_gate_reports_catch_and_false_block_rates() -> None:
    r = grounding_gate.run(n=400, workers=1)["results"]
    assert r["corrupted"] > 0
    assert 0.0 <= r["tier1_catch_rate"] <= 1.0
    assert r["baseline_verifier_calls"] == r["sentences"]
    assert r["with_scorer"]["verifier_calls"] <= r["sentences"]


def test_speculation_latency_sweeps_the_eager_threshold() -> None:
    data = speculation_latency.run(n=2, workers=1)
    rows = data["results"]["configs"]
    assert {row["eager_threshold"] for row in rows} >= {"off", 0.5}
    off = next(row for row in rows if row["eager_threshold"] == "off")
    assert off["speculations_started"] == 0
    assert all(row["v2v_ms"]["p50"] is not None for row in rows)


def test_endpointing_compares_three_controllers() -> None:
    r = endpointing.run(n=60, workers=1)["results"]
    assert set(r["controllers"]) == {"hybrid", "fixed_800ms", "text_only"}
    for c in r["controllers"].values():
        assert 0.0 <= c["premature_cutoff_rate"] <= 1.0
        assert c["mean_end_latency_ms"] is not None


def test_idempotency_counts_double_applies_per_configuration() -> None:
    r = idempotency.run(n=8, workers=1)["results"]
    assert r["switchboard"]["double_applied"] == 0
    assert r["switchboard"]["crashed_runs"] > 0
    assert set(r) >= {"switchboard", "switchboard_ledger_lost", "naive_retry"}


def test_multiparty_pass_hat_k_per_persona() -> None:
    r = multiparty.run(n=2, workers=1, ks=(1, 2))["results"]
    assert set(r["personas"]) == {"clean", "wer_5", "wer_12", "slow_speaker", "barge_in_heavy"}
    assert 0.0 <= r["overall"]["pass^2"] <= r["overall"]["pass^1"] <= 1.0


def test_cost_compares_routing_and_cache() -> None:
    r = cost.run(n=2, workers=1)["results"]
    configs = {(c["routing"], c["cache"]) for c in r["configs"]}
    assert ("talker_thinker", "always_hit") in configs and ("thinker_only", "never") in configs
    assert all(c["per_minute"]["total"] > 0 for c in r["configs"])


def test_run_all_writes_every_result_and_a_summary(tmp_path: Path) -> None:
    run_all.run_all(out_dir=tmp_path, quick=True, workers=1)
    names = {p.stem for p in tmp_path.glob("*.json")}
    assert names == set(run_all.BENCHMARKS)
    summary = (tmp_path / "summary.md").read_text()
    assert "policy_gate" in summary and "\u2014" not in summary
    data = json.loads((tmp_path / "policy_gate.json").read_text())
    assert data["simulated"] is True and data["note"] == common.SIMULATED_NOTE


async def test_serve_populates_a_store_the_api_can_read() -> None:
    from fastapi.testclient import TestClient

    from benchmarks.serve import populate
    from switchboard.api import Principal, StaticTokenAuthenticator
    from switchboard.api.app import create_app
    from switchboard.core import InMemoryEventStore

    store = InMemoryEventStore()
    ids = await populate(store, calls=2)
    assert len(ids) == 2
    # Reads need a principal of the tenant that owns the calls.
    tenant = (await store.read(ids[0]))[0].tenant_id  # type: ignore[union-attr]
    auth = StaticTokenAuthenticator({"sim-token": Principal(tenant)})
    client = TestClient(
        create_app(store, authenticator=auth), headers={"Authorization": "Bearer sim-token"}
    )
    body = client.get(f"/v1/conversations/{ids[0]}").json()
    assert body["timeline"]["turns"]
    assert client.get("/v1/activity", params={"conversation_id": ids[1]}).json()["items"]
