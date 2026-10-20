"""Run every benchmark, write each JSON result and `results/summary.md`.

`--quick` runs tiny samples, for checking the plumbing; the committed results use the defaults.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks import (
    cost,
    endpointing,
    grounding_gate,
    idempotency,
    multiparty,
    policy_gate,
    speculation_latency,
)
from benchmarks.common import RESULTS, SIMULATED_NOTE, fmt, markdown_table, write_result

Runner = Callable[..., dict[str, Any]]

BENCHMARKS: dict[str, tuple[Runner, int]] = {
    # name: (run, quick n)
    "policy_gate": (policy_gate.run, 6),
    "grounding_gate": (grounding_gate.run, 400),
    "speculation_latency": (speculation_latency.run, 2),
    "endpointing": (endpointing.run, 60),
    "idempotency": (idempotency.run, 8),
    "multiparty": (multiparty.run, 8),
    "cost": (cost.run, 2),
}


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def summarize(data: dict[str, dict[str, Any]], timings: dict[str, float]) -> str:
    out = [
        "# Benchmark summary",
        "",
        SIMULATED_NOTE,
        "Method, assumptions and limits are in `benchmarks/README.md`. Wall time per benchmark is",
        "listed at the end.",
        "",
    ]

    r = data["policy_gate"]["results"]
    a, b = r["baseline_post_hoc"], r["switchboard_pre_execution"]
    n = data["policy_gate"]["params"]["n"]
    out += [
        f"## policy_gate ({n} adversarial conversations per mode)",
        "",
        markdown_table(
            [
                "mode",
                "violating proposed",
                "violating executed",
                "flagged after execution",
                "false blocks on legit writes",
                "check latency",
            ],
            [
                [
                    "post-hoc detector (baseline)",
                    a["violating_proposed"],
                    a["violating_executed"],
                    a["detected_post_hoc"],
                    a["false_blocks"],
                    f"{a['check_latency_ms_modeled']:.0f} ms modeled, after execution",
                ],
                [
                    "pre-execution checker",
                    b["violating_proposed"],
                    b["violating_executed"],
                    "n/a",
                    b["false_blocks"],
                    f"p50 {b['check_latency_us']['p50']:.1f} us, p99 "
                    f"{b['check_latency_us']['p99']:.1f} us (measured)",
                ],
            ],
        ),
        "",
        markdown_table(
            [
                "violation kind",
                "baseline proposed",
                "baseline executed",
                "checker proposed",
                "checker executed",
            ],
            [
                [
                    k,
                    a["violating_proposed_by_kind"].get(k, 0),
                    a["violating_executed_by_kind"].get(k, 0),
                    b["violating_proposed_by_kind"].get(k, 0),
                    b["violating_executed_by_kind"].get(k, 0),
                ]
                for k in sorted(
                    set(a["violating_proposed_by_kind"]) | set(b["violating_proposed_by_kind"])
                )
            ],
        ),
        "",
        f"Logged verdicts recomputed from the log: {b['check_latency_us']['samples']}, "
        f"mismatches: {b['verdict_replay_mismatches']}.",
        "",
    ]

    g = data["grounding_gate"]["results"]
    out += [
        f"## grounding_gate ({g['sentences']} sentences, {g['corrupted']} corrupted)",
        "",
        markdown_table(
            [
                "config",
                "verifier calls",
                "corrupted caught",
                "corrupted released",
                "false blocks on clean",
            ],
            [
                [
                    "tier 1 only (runtime default)",
                    0,
                    pct(g["tier1_catch_rate"]),
                    g["tier1_only_released_corrupt"],
                    pct(g["false_block_rate_clean"]),
                ],
                [
                    "tier 1 + lexical scorer + tier 3",
                    g["with_scorer"]["verifier_calls"],
                    "n/a",
                    g["with_scorer"]["released_corrupt"],
                    pct(
                        g["with_scorer"]["false_blocks_clean"]
                        / max(1, g["sentences"] - g["corrupted"])
                    ),
                ],
                [
                    "every sentence to tier 3 (baseline)",
                    g["baseline_verifier_calls"],
                    pct(g["baseline_catch_rate"]),
                    g["baseline_released_corrupt"],
                    "0.0% (assumed)",
                ],
            ],
        ),
        "",
        "False blocks on clean sentences by category:",
        "",
        markdown_table(
            ["category", "clean sentences", "tier 1 only", "rate", "with lexical scorer", "rate"],
            [
                [
                    k,
                    v["sentences"] - v["corrupted"],
                    v["false_blocks"],
                    pct(v["false_block_rate"]),
                    v["scored_false_blocks"],
                    pct(v["scored_false_block_rate"]),
                ]
                for k, v in g["by_category"].items()
            ],
        ),
        "",
        f"Tier 1 latency: p50 {g['tier1_latency_us']['p50']:.1f} us, "
        f"p99 {g['tier1_latency_us']['p99']:.1f} us. "
        f"Tier 1 only escalations, failed closed with no verifier and counted as blocks: "
        f"{pct(g['tier1_only']['escalated_share'])} of sentences. "
        f"Share of sentences needing tier 3 with the lexical scorer: "
        f"{pct(g['with_scorer']['tier3_share'])}.",
        "",
    ]

    s = data["speculation_latency"]["results"]["configs"]
    out += [
        "## speculation_latency (voice to voice from TurnTimeline, end of speech to first audio)",
        "",
        markdown_table(
            [
                "eager threshold",
                "persona",
                "v2v p50 ms",
                "v2v p95 ms",
                "caller-measured p50 ms",
                "speculations",
                "cancelled share",
                "wasted output tokens",
            ],
            [
                [
                    c["eager_threshold"],
                    c["persona"],
                    c["v2v_ms"]["p50"],
                    c["v2v_ms"]["p95"],
                    c["caller_v2v_ms"]["p50"],
                    c["speculations_started"],
                    pct(c["wasted_speculation_ratio"]),
                    pct(c["wasted_output_token_ratio"]),
                ]
                for c in s
            ],
        ),
        "",
    ]

    e = data["endpointing"]["results"]
    rows = []
    for key, label in (
        ("controllers", "bare partials"),
        ("noisy_punctuation", "noisy punctuation"),
        ("punctuated_partials", "punctuated partials"),
    ):
        for name, v in e[key].items():
            rows.append(
                [
                    name,
                    label,
                    pct(v["premature_cutoff_rate"]),
                    pct(v["premature_cutoff_rate_paused_turns"]),
                    fmt(v["mean_end_latency_ms"]),
                    fmt(v["p95_end_latency_ms"]),
                ]
            )
    out += [
        f"## endpointing ({data['endpointing']['params']['turns']} turns, "
        f"{e['turns_with_a_pause']} with a mid-sentence pause)",
        "",
        markdown_table(
            [
                "controller",
                "transcripts",
                "premature cutoffs",
                "on paused turns",
                "mean end latency ms",
                "p95 end latency ms",
            ],
            rows,
        ),
        "",
    ]

    i = data["idempotency"]["results"]
    out += [
        f"## idempotency ({data['idempotency']['params']['n']} crash seeds per config)",
        "",
        markdown_table(
            ["config", "crashed runs", "intended writes", "applied", "double applies", "missed"],
            [
                [
                    k,
                    v["crashed_runs"],
                    v["intended_writes"],
                    v["applied_writes"],
                    v["double_applied"],
                    v["missed"],
                ]
                for k, v in i.items()
            ],
        ),
        "",
    ]

    m = data["multiparty"]
    ks = m["params"]["ks"]
    out += [
        f"## multiparty (courier and customer flow, {m['params']['n_per_task']} seeds per task)",
        "",
        markdown_table(
            ["persona", *(f"pass^{k}" for k in ks)],
            [
                [p, *(fmt(v[f"pass^{k}"]) for k in ks)]
                for p, v in [
                    *m["results"]["personas"].items(),
                    ("overall", m["results"]["overall"]),
                ]
            ],
        ),
        "",
    ]

    c = data["cost"]["results"]["configs"]
    out += [
        "## cost (per conversation minute, prices are assumptions)",
        "",
        markdown_table(
            ["routing", "cache", "total $/min", "LLM $/min", "LLM share", "turns to talker"],
            [
                [
                    x["routing"],
                    x["cache"],
                    f"{x['per_minute']['total']:.4f}",
                    f"{x['per_minute']['llm']:.4f}",
                    pct(x["llm_share"]),
                    pct(x["turns_routed_to_talker"]),
                ]
                for x in c
            ],
        ),
        "",
        "## wall time",
        "",
        markdown_table(["benchmark", "seconds"], [[k, f"{v:.1f}"] for k, v in timings.items()]),
        "",
    ]
    return "\n".join(out)


def run_all(out_dir: Path | None = None, *, quick: bool = False, workers: int | None = None) -> str:
    out_dir = out_dir or RESULTS
    data: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {}
    for name, (runner, quick_n) in BENCHMARKS.items():
        start = time.perf_counter()
        data[name] = runner(n=quick_n if quick else None, workers=workers)
        timings[name] = time.perf_counter() - start
        write_result(name, data[name], out_dir)
        print(f"{name}: {timings[name]:.1f} s", flush=True)
    timings["total"] = sum(timings.values())
    summary = summarize(data, timings)
    (out_dir / "summary.md").write_text(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.run_all")
    parser.add_argument("--quick", action="store_true", help="tiny samples, plumbing only")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(run_all(args.out, quick=args.quick, workers=args.workers))


if __name__ == "__main__":
    main()
