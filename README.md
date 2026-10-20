# Switchboard

A runtime for enterprise voice agents that can hold several live calls at once, refuse an unsafe
action before it runs, and refuse to say a number it cannot back up.

```
courier ──call──►  ┌──────────────────────────── conversation (one writer, one log) ───────────────────────────┐
                   │  leg: courier   "I delivered it to their home, can you mark it delivered?"                │
                   │     ├─ verify_courier ✓   lookup_order ✓   chat shows no address change                   │
                   │     ├─ mark_delivered  ✗  policy: delivered_needs_confirmation                            │
                   │     └─ call_party(customer) ──► leg: customer  "Did you ask for delivery to your home?"    │
                   │                                         └─ "Yes, I did."  → child.result confirmed        │
                   │     ├─ mark_delivered  ✓  (exactly once, idempotency key in the ledger)                   │
                   │     └─ "All set, it's marked delivered. Your pay this week is $182.40."              │
                   │           ← gated before TTS: $182.40 matches the get_courier_pay result                  │
                   └───────────────────────────────────────────────────────────────────────────────────────────┘
```

That flow is the example agent in [`examples/delivery_support`](examples/delivery_support), and it
runs end to end in the test suite on a virtual clock in milliseconds.

## See it run

```
make demo
```

About a second, no API keys. It prints the call above as a transcript, then the same call with a
model that misbehaves. From Part 2 ([full output](docs/demo-output.txt)): the courier asks about
pay while the customer leg is still open, a wrong dollar amount is blocked before TTS, an early
`mark_delivered` is blocked by policy, and the real one runs once after the customer confirms.

```
 0:25.3  COURIER  While I wait, what is my pay this week?
 0:25.9    tool  get_courier_pay(courier_id=C-77) -> week_total=182.4, deliveries=23
 0:26.0  AGENT -> courier  Let me check.
 0:26.7    BLOCKED speech "You've earned $112.40 this week across 23 deliveries.": unsupported $112.40 (evidence: $182.40)
 0:27.2    BLOCKED mark_delivered: delivered_needs_confirmation
 0:27.2            hint to model: The address change is not confirmed. Check the chat, or call the customer with call_party to confirm it first.
 0:27.3  AGENT -> courier  You've earned $182.40 this week across 23 deliveries.
 0:27.7  CUSTOMER Yes, I asked the courier to bring it to my home.
 0:28.2    tool  report_result(confirmed=true, notes=Asked for home delivery.) -> recorded=true
 0:28.2    result customer confirmed
 0:28.3  AGENT -> customer  Thanks, I'll let the courier know. Goodbye.
 0:28.7    tool  mark_delivered(order_id=48213, location=customer's home address) -> delivered  [key mark_delivered:41e9ad9c...]
```

`make serve` serves the same simulated calls over the API, with an activity stream page at the
`/activity` URL it prints.

## Why this exists

I built this after studying how production voice support agents are put together, in particular
Giga's public engineering posts and their YC interview about the DoorDash deployment. Their
approach (a reasoning-model hallucination detector that races TTS playback, a per-policy classifier
that picks which rules to load, agents defined as code) is strong. Switchboard asks a narrower
question: **which of those checks can move from "an LLM looks at it afterwards" to "a deterministic
check before it happens"**, and what does the runtime need to look like for that to work across
multi-party calls, crashes, and retries.

The answers became fifteen decision records in [docs/DECISIONS.md](docs/DECISIONS.md). The short version:

| Problem | Common approach | Switchboard |
|---|---|---|
| Agent marks delivered or refunds when it should not | LLM detector reviews the turn about 3 s later | Policies compiled to a procedure graph and rule checker; the call never executes, the model gets a repair hint |
| Agent says the wrong refund amount or ETA | Reasoning model checks each sentence during playback | Every money, time, duration and ID span is matched against tool results before TTS; only ambiguous sentences reach an LLM verifier |
| Two parties on two calls | Sub-agent with its own state | Child legs inside one conversation; one writer, one event log, results arrive as events |
| Retries and crashes | Checkpoints | Deterministic idempotency keys per intent, a claim ledger, spec-defined compensations, replay from the log |
| Latency | Wait for end of turn | Speculative generation on an eager end-of-turn signal; writes wait for the committed turn |
| Testing | Simulations | Virtual-clock simulation with adversarial models and ASR noise, pass^k gates, bit-identical replay |

## Results

All numbers come from `make bench` on a simulated stack (virtual clock, simulated STT/TTS with
latency distributions, scripted and adversarial LLMs). They measure the architecture, not real
providers. Method and limits are in [benchmarks/README.md](benchmarks/README.md); full tables in
[benchmarks/results/summary.md](benchmarks/results/summary.md).

**Unsafe actions** (1,000 adversarial conversations per mode; the model is made to propose early
`mark_delivered`, refunds over the limit, second refunds, and repeated writes)

| | violating writes proposed | executed | false blocks on legitimate writes | check latency |
|---|---|---|---|---|
| Post-hoc detector baseline (95% recall, 3 s after execution) | 1,677 | **1,437** | 0 | 3 s, after the fact |
| Pre-execution checker | 1,732 | **0** | 0 | p50 5.6 µs, p99 23.7 µs |

All 7,732 logged policy verdicts were recomputed from the event log with zero mismatches.

**Wrong numbers in speech** (20,000 sentences, 520 with a corrupted amount, time, ETA or order ID)

| | LLM verifier calls | corrupted caught | false blocks on clean sentences |
|---|---|---|---|
| Every sentence to an LLM verifier (baseline, 94% recall) | 20,000 | 94.2% | assumed 0% |
| Tier 1 deterministic check (runtime default) | 0 | 100% | 0.0% |

Tier 1 runs at p50 42.5 µs. About 0.5% of sentences are ambiguous and escalate (they fail closed
when no verifier is configured).

**Latency** (voice to voice, end of caller speech to first agent audio)

| | p50 | p95 |
|---|---|---|
| No speculation | 1,050 ms | 1,580 ms |
| Speculative turns (eager threshold 0.5) | **776 ms** | **1,300 ms** |

Wasted generation from cancelled speculation stayed at or under 0.1% of output tokens.

**Turn-taking** (3,000 turns, 1,360 with a mid-sentence pause)

| Controller | premature cutoffs | cutoffs on paused turns |
|---|---|---|
| Fixed 800 ms silence timer | 14.1% | 31.2% |
| Hybrid (silence + completeness + adaptive pauses) | **2.4%** | **5.3%** |

**Exactly-once writes** (500 crash points injected during write tools, then recovery from the log)

| | intended writes | applied | double applies |
|---|---|---|---|
| Naive retry | 745 | 993 | 248 |
| Switchboard | 745 | 745 | **0** |

**Multi-party reliability** (courier plus customer flow, pass^k across seeds)

| Caller persona | pass^1 | pass^4 | pass^8 |
|---|---|---|---|
| Clean, 5% WER, slow speaker, barge-in heavy | 1.00 | 1.00 | 1.00 |
| 12% word error rate | 0.94 | 0.76 | 0.56 |

The 12% WER row is the honest weak spot: misheard digits in an order ID break verification. That is
what a production deployment would attack next (confirmation read-back, DTMF fallback).

## Architecture

```
 media plane                          control plane (this repo)
 Twilio / WebRTC ──► Leg ── endpointing controller ── speculative turn engine
                      │
                      ▼
               Conversation actor ── append-only event log ──► replay, activity stream, traces
                      │   procedure state, evidence, pinned bundle id
                      ▼
               router ─► talker | thinker LLM  (stable prompt prefix for caching, deadline fallback)
                      │ tool call                         │ sentences
                      ▼                                   ▼
               policy checker (µs)                 grounding gate (tier 1 µs, tier 3 async)
                      │ allowed                           │ released
                      ▼                                   ▼
               effect runner + idempotency ledger    TTS ─► playback cursor (history = what was heard)
                      │
        tools · bitemporal knowledge · scoped memory · child legs
```

| Package | What it does |
|---|---|
| `core` | Injectable clock (virtual and real), typed events, event stores (memory, SQLite) with optimistic concurrency, keyed single-writer actors |
| `policy` | Hand-written parser for the rule language (no `eval`), procedure graph, compiler, checker, intent detection |
| `grounding` | Claim extraction for money, times, dates, durations, IDs including spoken forms; typed evidence; tiered gate |
| `turn` | Hybrid endpointing, speculative turn engine with typed generation context, playback cursor, barge-in policy |
| `effects` | Tool specs and argument validation, idempotency keys, claim ledger with leases, saga compensations |
| `knowledge` | Bitemporal fact store, Atlassian Statuspage ingestor (incidents valid only while open), BM25 with a recency prior |
| `memory` | Tenant and customer scoped memory, typed with TTLs, hard delete, consolidation after the call |
| `brain` | LLM protocol, talker and thinker router, cache-stable prompt assembly, deadline fallback chain, Anthropic adapter |
| `runtime` | Conversation actor, legs, child calls, admission control, token budgets, Twilio media session, crash recovery |
| `telemetry` | Per-turn latency decomposition with OpenTelemetry GenAI attribute names, OTLP JSON export, activity stream projection |
| `api` | FastAPI: activity stream and histogram, conversation timeline and trace, Prometheus metrics, authenticated Twilio media WebSocket |
| `sim` | Simulated callers, adversarial LLM, scenarios, pass^k, cost model |

## Running it

```bash
uv sync
make check     # ruff, format, mypy --strict, 1,025 tests, about 20 s, no network or API keys
make bench     # all benchmarks, about 60 s
make serve     # API plus the activity stream page, seeded with simulated calls
```

To use a real model, install the extra and pass `AnthropicLLM` to the runtime:

```bash
uv sync --extra anthropic --extra api
```

## How it was built

- Decisions first: each ADR cites the paper or production write-up behind it, and two of them
  carry a dated revision where a benchmark proved the original reasoning wrong.
- Test first: every module started with a failing test. Property tests cover the rule parser,
  number normalization, mu-law codec and event store concurrency.
- Reviewed: an independent review pass found twelve defects (a hang-up that could wedge a call, a
  verifier error that silenced the speaker, a path to a double write, an unauthenticated media
  socket). Each was reproduced, fixed with a regression test, and re-verified.

## Limits

- Everything is measured in simulation. Real STT, TTS and LLM latency and error profiles will move
  the numbers; the scheduling advantages should hold, the absolute values will not.
- The media plane is an adapter. A production deployment needs an SFU or SIP edge pinned per call so
  an agent restart never drops caller audio.
- The rule language covers argument bounds, evidence facts, ordering and counts. Policies that need
  judgement stay in the prompt and fall to the speech gate.

MIT licensed.
