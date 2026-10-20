"""Multi-party reliability (ADR-003, ADR-014): pass^k of the courier and customer flow.

Two tasks per persona: the customer confirms the address change (the order must be marked
delivered exactly once, only after the confirmation) and the customer declines (it must never
be marked). Every human on the call gets the persona: clean, 5% and 12% word error rate, a slow
speaker who pauses mid-sentence, and an impatient caller who interrupts any agent utterance
longer than 1.5 s.

A task is one scenario under one persona; pass^k is the tau-bench estimator, the mean over tasks
of C(c, k) / C(n, k) with n seeds per task.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from benchmarks.common import cli, envelope, pmap, table
from switchboard.sim import PERSONAS, pass_hat_k, run_scenario
from switchboard.sim.delivery import declined_delivery, happy_delivery

NAME = "multiparty"
N_DEFAULT = 32
KS = (1, 4, 8)
SCENARIOS = {"happy_delivery": happy_delivery, "declined_delivery": declined_delivery}


def one(job: tuple[str, str, int]) -> dict[str, Any]:
    scenario, persona, seed = job
    result = run_scenario(SCENARIOS[scenario](), seed, PERSONAS[persona])
    return {
        "passed": result.passed,
        "failed": [r.name for r in result.assertion_results if not r.passed],
        "outcome": result.outcome,
    }


def run(n: int | None = None, workers: int | None = None, ks: Sequence[int] = KS) -> dict[str, Any]:
    n = n or N_DEFAULT
    jobs = [(s, p, seed) for p in PERSONAS for s in SCENARIOS for seed in range(n)]
    rows = pmap(one, jobs, workers)
    tasks: dict[tuple[str, str], list[bool]] = {}
    failures: dict[str, Counter[str]] = {p: Counter() for p in PERSONAS}
    for (s, p, _), r in zip(jobs, rows, strict=True):
        tasks.setdefault((s, p), []).append(r["passed"])
        failures[p].update(r["failed"])
    personas = {}
    for p in PERSONAS:
        mine = {k: v for k, v in tasks.items() if k[1] == p}
        personas[p] = {
            **{f"pass^{k}": pass_hat_k(mine, k) for k in ks},
            "by_scenario": {s: sum(v) / len(v) for (s, _), v in mine.items()},
            "assertion_failures": dict(failures[p].most_common()),
        }
    overall = {f"pass^{k}": pass_hat_k(tasks, k) for k in ks}
    return envelope(
        NAME,
        {
            "n_per_task": n,
            "ks": list(ks),
            "scenarios": list(SCENARIOS),
            "personas": {p: vars(v) for p, v in PERSONAS.items()},
        },
        {"personas": personas, "overall": overall},
    )


def render(data: dict[str, Any]) -> str:
    r = data["results"]
    ks = data["params"]["ks"]
    head = ["persona", *(f"pass^{k}" for k in ks), "top_failure"]
    rows = []
    for p, v in [*r["personas"].items(), ("overall", r["overall"])]:
        top = next(iter(v.get("assertion_failures", {})), "")
        rows.append([p, *(v[f"pass^{k}"] for k in ks), top])
    return table(head, rows)


def main() -> None:
    cli(NAME, run, render)


if __name__ == "__main__":
    main()
