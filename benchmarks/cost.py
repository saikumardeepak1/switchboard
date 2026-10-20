"""Cost per minute (ADR-008 routing, prompt caching), from events plus metered model usage.

The happy delivery call, with the courier adding the short acknowledgements real callers make
("Okay, thanks.", "Great.") so the router has easy turns to send to the talker. Two routings:
talker plus thinker (the router decides per turn) and thinker only. Three cache conditions:
`auto` (one conversation alone: the first call writes the prefix, later calls hit), `always_hit`
(a warm prefix shared with other calls on the same release bundle) and `never` (no caching).

Every price is an assumption documented in `switchboard.sim.cost.CostConfig`. Token counts come
from the scripted model (one token per word) scaled by `tokens_per_word`; the real system
prompt, tool schemas and replies of a production agent are longer, so treat the LLM share as a
floor, not an estimate.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from benchmarks.common import cli, envelope, mean, pmap, ratio, table
from switchboard.sim import (
    CallerSpec,
    CostConfig,
    Say,
    Scenario,
    SimConfig,
    WaitForAgent,
    run_scenario,
)
from switchboard.sim.delivery import BYE, DELIVERED, PAY, REPORT, VERIFY, happy_delivery

NAME = "cost"
N_DEFAULT = 30
ROUTINGS = ("talker_thinker", "thinker_only")
CACHES = ("auto", "always_hit", "never")
ACK = (Say("Okay, thanks."), WaitForAgent())
GREAT = (Say("Great."), WaitForAgent())


def chatty() -> Scenario:
    base = happy_delivery()
    courier = (*VERIFY, *ACK, *REPORT, *ACK, *PAY, WaitForAgent(), *GREAT, *DELIVERED, *ACK, *BYE)
    return dataclasses.replace(
        base, name="chatty_delivery", callers={**base.callers, "courier": CallerSpec(courier)}
    )


def one(job: tuple[str, str, int]) -> dict[str, Any]:
    routing, cache, seed = job
    config = SimConfig(use_talker=routing == "talker_thinker", cache_mode=cache)  # type: ignore[arg-type]
    result = run_scenario(dataclasses.replace(chatty(), config=config), seed)
    c = result.cost
    return {
        "minutes": c.minutes,
        "stt": c.stt,
        "tts": c.tts,
        "telephony": c.telephony,
        "llm": c.llm,
        "total": c.total,
        "llm_calls": dict(c.llm_calls),
        "routed": [getattr(e, "model_tier", "") for e in result.events if e.type == "brain.routed"],
        "passed": result.passed,
    }


def run(n: int | None = None, workers: int | None = None) -> dict[str, Any]:
    n = n or N_DEFAULT
    jobs = [(r, c, s) for r in ROUTINGS for c in CACHES for s in range(n)]
    rows = pmap(one, jobs, workers)
    configs = []
    for r in ROUTINGS:
        for c in CACHES:
            group = [x for (jr, jc, _), x in zip(jobs, rows, strict=True) if (jr, jc) == (r, c)]
            minutes = sum(x["minutes"] for x in group)
            calls = {m: sum(x["llm_calls"].get(m, 0) for x in group) for m in ("talker", "thinker")}
            routed = [t for x in group for t in x["routed"]]
            configs.append(
                {
                    "routing": r,
                    "cache": c,
                    "runs": len(group),
                    "mean_minutes": mean([x["minutes"] for x in group]),
                    # Pooled: total dollars over total minutes across runs.
                    "per_minute": {
                        k: sum(x[k] for x in group) / minutes
                        for k in ("stt", "tts", "telephony", "llm", "total")
                    },
                    "llm_share": ratio(
                        sum(x["llm"] for x in group), sum(x["total"] for x in group)
                    ),
                    "llm_calls": calls,
                    "turns_routed_to_talker": ratio(routed.count("talker"), len(routed)),
                    "pass_rate": ratio(sum(1 for x in group if x["passed"]), len(group)),
                }
            )
    return envelope(
        NAME,
        {"n": n, "scenario": "chatty_delivery", "prices": _prices(CostConfig())},
        {"configs": configs},
    )


def _prices(cfg: CostConfig) -> dict[str, Any]:
    out = {k: v for k, v in vars(cfg).items() if k != "models"}
    out["models"] = {k: vars(v) for k, v in cfg.models.items()}
    out["note"] = "assumptions; see switchboard.sim.cost for sources"
    return out


def render(data: dict[str, Any]) -> str:
    head = [
        "routing",
        "cache",
        "total_$/min",
        "llm_$/min",
        "media_$/min",
        "llm_share",
        "talker_turns",
    ]
    rows = []
    for c in data["results"]["configs"]:
        pm = c["per_minute"]
        rows.append(
            [
                c["routing"],
                c["cache"],
                pm["total"],
                pm["llm"],
                pm["stt"] + pm["tts"] + pm["telephony"],
                c["llm_share"],
                c["turns_routed_to_talker"],
            ]
        )
    return table(head, rows)


def main() -> None:
    cli(NAME, run, render)


if __name__ == "__main__":
    main()
