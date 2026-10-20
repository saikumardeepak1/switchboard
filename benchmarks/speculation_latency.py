"""Speculative turns (ADR-006): voice-to-voice latency with speculation on and off.

The happy delivery call is run under two personas (a clean speaker, and a slow speaker who
pauses mid-sentence, which is what makes speculation guess wrong), sweeping the bundle's eager
threshold. "off" sets the eager threshold equal to the end threshold, so `eager_end` never fires
before `end` and every turn starts generating only at commit.

Voice to voice is measured where the caller is: from the moment they stop speaking to the first
agent audio chunk that reaches them. It is not taken from `TurnTimeline`, whose `first_audio_at`
is the `speech.played` event, appended when an utterance finishes playing.

Wasted generation: speculations cancelled over speculations started, and the share of model
output tokens produced by calls that were cancelled.

Latency here is the simulator's: STT final lag, model time to first token and TTS first byte
come from configured distributions (`SimConfig`), so this measures scheduling, not providers.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from benchmarks.common import cli, envelope, mean, pct, pmap, ratio, table
from switchboard.sim import PERSONAS, SPECULATION_OFF, SimConfig, run_scenario
from switchboard.sim.delivery import happy_delivery

NAME = "speculation_latency"
N_DEFAULT = 40
THRESHOLDS: tuple[float | str, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, "off")
PERSONA_NAMES = ("clean", "slow_speaker")


def one(job: tuple[float | str, str, int]) -> dict[str, Any]:
    threshold, persona, seed = job
    eager = SPECULATION_OFF if threshold == "off" else float(threshold)
    scenario = dataclasses.replace(happy_delivery(), config=SimConfig(eager_threshold=eager))
    result = run_scenario(scenario, seed, PERSONAS[persona])
    types = [e.type for e in result.events]
    usages = result.usages
    return {
        "v2v": result.voice_to_voice_ms,
        "started": types.count("turn.speculation_started"),
        "cancelled": types.count("turn.speculation_cancelled"),
        "calls": len(usages),
        "cancelled_calls": sum(1 for u in usages if u.cancelled),
        "output": sum(u.output_tokens for u in usages),
        "wasted_output": sum(u.output_tokens for u in usages if u.cancelled),
        "endpointing_ms": [
            float(getattr(e, "endpointing_ms", 0.0))
            for e in result.events
            if e.type == "turn.committed"
        ],
        "passed": result.passed,
    }


def run(n: int | None = None, workers: int | None = None) -> dict[str, Any]:
    n = n or N_DEFAULT
    jobs = [(t, p, s) for t in THRESHOLDS for p in PERSONA_NAMES for s in range(n)]
    rows = pmap(one, jobs, workers)
    configs = []
    for t in THRESHOLDS:
        for p in PERSONA_NAMES:
            group = [r for (jt, jp, _), r in zip(jobs, rows, strict=True) if jt == t and jp == p]
            v2v = [v for r in group for v in r["v2v"]]
            started = sum(r["started"] for r in group)
            configs.append(
                {
                    "eager_threshold": t,
                    "persona": p,
                    "runs": len(group),
                    "turns": len(v2v),
                    "v2v_ms": {"p50": pct(v2v, 50), "p95": pct(v2v, 95), "mean": mean(v2v)},
                    "endpointing_ms_mean": mean([x for r in group for x in r["endpointing_ms"]]),
                    "speculations_started": started,
                    "speculations_cancelled": sum(r["cancelled"] for r in group),
                    "wasted_speculation_ratio": ratio(sum(r["cancelled"] for r in group), started),
                    "llm_calls": sum(r["calls"] for r in group),
                    "llm_calls_cancelled": sum(r["cancelled_calls"] for r in group),
                    "wasted_output_token_ratio": ratio(
                        sum(r["wasted_output"] for r in group), sum(r["output"] for r in group)
                    ),
                    "pass_rate": ratio(sum(1 for r in group if r["passed"]), len(group)),
                }
            )
    return envelope(
        NAME,
        {
            "n": n,
            "thresholds": list(THRESHOLDS),
            "personas": list(PERSONA_NAMES),
            "sim_config": _describe(SimConfig()),
        },
        {"configs": configs},
    )


def _describe(cfg: SimConfig) -> dict[str, Any]:
    return {
        "thinker_ttft_s": {"p50": cfg.thinker_ttft.p50, "p95": cfg.thinker_ttft.p95},
        "talker_ttft_s": {"p50": cfg.talker_ttft.p50, "p95": cfg.talker_ttft.p95},
        "tokens_per_s": cfg.tokens_per_s,
        "tts_ttfb_s": "from the bundle voice settings (0.15)",
        "stt_final_lag_s": "from the persona (clean p50 0.08 p95 0.12, slow p50 0.1 p95 0.2)",
    }


def render(data: dict[str, Any]) -> str:
    head = [
        "eager",
        "persona",
        "v2v_p50_ms",
        "v2v_p95_ms",
        "spec_started",
        "wasted_spec",
        "wasted_tokens",
        "pass_rate",
    ]
    rows = [
        [
            c["eager_threshold"],
            c["persona"],
            c["v2v_ms"]["p50"],
            c["v2v_ms"]["p95"],
            c["speculations_started"],
            c["wasted_speculation_ratio"],
            c["wasted_output_token_ratio"],
            c["pass_rate"],
        ]
        for c in data["results"]["configs"]
    ]
    return table(head, rows)


def main() -> None:
    cli(NAME, run, render)


if __name__ == "__main__":
    main()
