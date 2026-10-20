# Architecture Decision Records

Each record states the decision, the evidence behind it, and what it costs. Decisions were made
before any code was written. Numbers in "Evidence" come from the cited papers or vendor posts and
are directional, not reproduced here.

The reference point throughout is the public description of Giga's production stack (Giga
engineering posts, 2025 to 2026): a per-policy LoRA classifier that selects which policies to
load per turn, a fine-tuned main model, a reasoning-model hallucination detector that races TTS
playback, agents defined as code in git, and a full-duplex voice front end being adopted in 2026.

---

## ADR-001: Text brain behind a swappable voice front end

**Decision.** Every decision the agent makes is a text LLM call with structured tool calls,
logged per turn. The voice front end (STT plus TTS today, a full-duplex model later) sits behind
one interface, `VoiceFrontEnd`, and talks to the brain through a delegate protocol: the front end
hands over a committed user turn and receives speakable text plus control signals.

**Evidence.** On Sierra's tau-voice benchmark, voice agents keep only about 79% of text-agent task
success; tool reliability is the gap. arXiv 2609.19334 and SALMONN-duo (2609.34247) both converge
on "duplex front, text brain behind" and reach parity with end-to-end realtime models on tool tasks.

**Cost.** A cascade loses prosody on input. Accepted, because enterprise support is graded on
correct actions, and the delegate interface lets a duplex front end replace the cascade without
touching policy, tools, or memory.

## ADR-002: A conversation is a single-writer actor over an append-only event log

**Decision.** Each conversation is a keyed actor with one mailbox and exactly one writer. All
state changes are typed events appended to a log with optimistic concurrency (expected sequence
number). Current state is a fold over events. Snapshots are an optimization, never the source of
truth.

**Evidence.** Quo runs each call as a Temporal workflow with transcripts and tool results as
signals; Restate Virtual Objects give one instance per key with queued calls. Both get crash replay
and a replayable transcript from the same mechanism.

**What it buys.** Crash recovery by replay, deterministic regression tests from production logs,
the Activity Stream as a projection of the log, and trace-to-config attribution because every event
carries the release bundle hash.

**Cost.** We implement a small kernel instead of adopting Temporal. The event and store interfaces
are deliberately Temporal-shaped (signals in, activities out) so a port is mechanical.

## ADR-003: Multi-party calls are legs owned by one conversation

**Decision.** A conversation owns N call legs. Each leg has its own media pipeline and turn state.
A leg can spawn a child leg (an outbound sub-agent call) with a goal and a result schema. Cross-leg
messages are events in the parent conversation's log, so shared state has exactly one owner.

**Evidence.** Giga won DoorDash on a two-call verification flow. LiveKit models every caller,
callee and agent as a participant and coordinates transfers with consultation rooms; the failure
mode of independent loops is competing writers on shared state.

**Cost.** A slow child leg must not block the parent leg. The parent leg keeps talking (bridge
speech, other questions) and receives the child result as an event.

## ADR-004: Transactional policy is checked before a tool runs, not after speech plays

**Decision.** Policies compile into two deterministic artifacts:

1. **Procedure graph.** Required steps and their order (for example `verify_identity` before any
   account read). Graph state is part of conversation state.
2. **Action rules.** Preconditions on tool calls written in a small rule language: argument bounds,
   required facts in evidence, temporal ordering, counts per conversation.

Every proposed tool call passes the checker before execution. A violation never executes; the
model gets a structured repair hint and proposes again, with a bounded retry budget before handoff.

**Evidence.** Agent-C (2512.23738) enforces ordering rules with SMT and constrained repair: 100%
conformance on tau-bench retail and airline from a 77 to 84% baseline, with task utility up.
PolicyGuide (2608.19861) compiles policies to workflow graphs: Pass^4 0.42 to 0.62. Giga's detector
runs about 3 seconds after generation, after a refund would already have executed.

**Cost.** Policies need to be written in the rule language. Natural-language policies that cannot
be compiled stay in the prompt and fall to ADR-005's speech gate.

## ADR-005: Spoken claims pass a tiered grounding gate before TTS

**Decision.** The brain streams sentences. Each sentence passes three tiers before it is released
to TTS:

1. **Deterministic claims (microseconds).** Extract checkable spans: money, numbers, dates and
   times, durations, order and ticket IDs. Each must match a value in the turn's evidence set (tool
   results, retrieved facts, memory). Mismatch blocks the sentence.
2. **Small grounding scorer (milliseconds).** Optional pluggable model for uncheckable sentences.
3. **Async verifier racing playback.** Only sentences the first two tiers cannot decide go to an
   LLM verifier that runs while earlier audio plays, as Giga does, with block-or-self-correct.

**Evidence.** Giga reports 4 to 5% hallucination down to under 1% with the reasoning-model
detector alone. LettuceDetect and HHEM show small models catch most retrieval-grounding errors.
Most support hallucinations that hurt (wrong refund amount, wrong ETA) are numeric and checkable.

**Cost.** Tier 1 has false positives on paraphrased numbers ("a couple of dollars"). Normalization
handles common forms; ambiguous spans escalate rather than block.

## ADR-006: Speculative turns with playback as the source of truth

**Decision.** The turn engine starts brain generation on an eager end-of-turn signal and holds the
output in a private buffer. If the user keeps talking, the speculation is cancelled and discarded.
If the turn commits, the buffer is released. Read-only tools may run speculatively; write tools only
run on a committed turn. Conversation history records only audio that was actually played: on
barge-in the assistant message is truncated at the playback cursor.

**Evidence.** PredGen (2506.15556) about 2x lower latency from speculation; Deepgram Flux exposes
EagerEndOfTurn and TurnResumed for exactly this; Voice-Light (2609.20995) treats played audio as
history. arXiv 2605.13360: speculative tool calls give 1.3 to 1.7x end-to-end, but can repeat
actions, which is why write tools are excluded.

**Cost.** Wasted brain calls on cancelled speculation. Capped per tenant by an eager threshold.

## ADR-007: Hybrid endpointing controller

**Decision.** End-of-turn is a calibrated combination of a silence timer, a text-completeness
score, and an optional audio model score, with per-caller adaptive pause statistics. No single
model decides.

**Evidence.** Voice-Light: a learned end-of-turn checkpoint alone reached 12.5% recall while a
timing policy reached 95.6%; their deployed system is hybrid. A fixed 800 ms silence timeout adds
nearly a second to every reply.

## ADR-008: Talker and thinker models with a router

**Decision.** A router classifies each committed turn. Easy turns (greetings, confirmations,
status reads already in evidence) go to a small talker model. Hard turns go to the thinker. While
the thinker or a tool runs, the talker produces bridge speech that is marked non-substantive and
exempt from the grounding gate only if it contains no claims.

**Evidence.** SALMONN-duo delegates 16% of easy and 61% of hard turns at equal accuracy;
RouteLLM cut cost by more than 2x at 95% quality. The LLM is the largest variable cost per minute.

## ADR-009: Side effects are idempotent activities with compensations in the spec

**Decision.** A tool with side effects declares `effect: write`, an idempotency key derivation
(conversation id, step sequence, action name, canonical arguments), and optionally a compensating
tool. The effect runner checks the key in a ledger before running, records the result, and
returns the recorded result on retry or replay. Compensations are run by the runtime, never chosen
by the LLM.

**Evidence.** Checkpoints alone still double-refund when the reply is lost after the side effect;
deterministic keys plus lookup-before-retry fixes it. Random UUIDs per retry defeat dedupe.

**Revision (2026-10).** The step sequence in the key was the wrong unit. It dedupes retries and
crash replays of one proposal, but models re-propose writes in later turns ("thanks, is it
marked now?"), and each re-proposal got a new step, a new key and a second downstream write. The
key is now conversation id, tool, canonical identity arguments and an explicit intent counter
per tool. The policy layer moves the counter only when a committed user turn makes a new request
for that tool (`RequestDetector`, cue words per tool plus a request cue or imperative at clause
start; thanks, status questions and statements about what already happened are excluded), and
records it on `turn.committed` so recovery derives the same key. Only the inbound caller's turns
count; a child call's party answers questions and opens nothing. A proposal identical to a call
that already ran replays it unless a request was opened after that call ran. A re-proposal in the same request replays
from the ledger; a real second request ("refund another ten dollars") runs. The cost: a request
the detector misses reuses the old key and replays instead of running, which fails safe for
writes, and a bundle can name its own cue words when the tool name is a poor cue.

## ADR-010: Knowledge is bitemporal; incidents expire on their own

**Decision.** Every fact has valid time (when it is true in the world) and transaction time (when
we learned it). Retrieval runs as of the call's current time. A status-page ingestor turns open
incidents into facts valid while open; resolution closes the interval without deleting history.

**Evidence.** Graphiti invalidates instead of deleting; TEMPRAGEVAL shows standard retrievers fail
time-sensitive questions. Maps to Giga's open "dynamic knowledge" work item.

## ADR-011: Memory is scoped, typed, expiring, and written off the hot path

**Decision.** Cross-session memory is keyed by tenant and customer, typed (profile, open issue,
preference, interaction summary), each type with a TTL, with hard delete. Writes happen in a
consolidation step after the call ends. Reads happen once at call setup and are prefetched.

**Evidence.** Mem0 shows 91% lower p95 than full context. MEXTRA and SPORE extract 42 to 80% of
stored records from unscoped memory. Letta's sleep-time compute moves consolidation off the live
path. Maps to Giga's open "agent memory" work item.

## ADR-012: Agents are code, compiled to immutable content-hashed release bundles

**Decision.** An agent is a directory: `agent.yaml`, `policies/*.yaml`, `tools.py`, `tests/`. A
compiler validates references, compiles policies (ADR-004), and emits a bundle whose id is the hash
of its canonical content. A call is pinned to one bundle at dispatch and cannot change mid-call.
Every event carries the bundle id.

**Evidence.** Sierra's Agent OS and ElevenLabs versioning both pin calls to immutable releases.
Giga's CLI keeps agents in git. Pinning gives exact trace-to-config attribution.

## ADR-013: Admission control at ring time, fallback chains on time-to-first-token

**Decision.** Calls are admitted at setup against a per-tenant concurrent-call cap, separate from a
token-per-minute budget. Each provider stage (STT, brain, TTS) has an ordered fallback chain that
fires on a deadline, not only on errors, ending in a human-handoff tier.

**Evidence.** A call holds a slot for minutes with low token use, so token limits alone do not
protect capacity. Slow providers cause dead air, the failure callers notice first (Vapi, Retell,
Sierra all use deadline-triggered fallback).

## ADR-014: Everything runs on an injectable clock; evaluation is deterministic simulation

**Decision.** No component reads wall time directly. A virtual clock drives simulated providers
with latency distributions, simulated callers, and ASR error injection, so a full multi-party call
runs in milliseconds and is bit-for-bit reproducible from a seed. Release gates use pass^k across
several simulated callers, plus replay of recorded event logs.

**Evidence.** Simulator choice swings agent scores by up to 9 points; pass@1 overstates
reliability; tau-bench uses pass^k. Barge-in and speculation are race conditions that only a
virtual clock can test deterministically.

## ADR-015: Per-turn latency is decomposed, not summed

**Decision.** Each turn records endpointing wait, STT final, router decision, brain first token,
gate decision, TTS first byte, and first audio out as separate attributes, with OpenTelemetry GenAI
attribute names. Content capture is off by default.

**Evidence.** Endpointing is the largest hidden cost and is invisible in an end-to-end number.

---

## Where this is meant to beat the reference design

| Concern | Reference (public description) | This design |
|---|---|---|
| Wrong action (refund, delivered) | LLM detector about 3 s after generation | Deterministic check before execution, repair loop |
| Wrong spoken number | Reasoning model per sentence | Exact match against evidence first, LLM only for leftovers |
| Policy selection | Separate LoRA classifier hop | Procedure graph state already says which steps are open |
| Multi-party state | Sub-agent protocol | One log owner, child legs as events, replayable |
| Regression testing | Simulation suites | Virtual-clock simulation plus replay of production logs, pass^k |
| Retries | Not described | Idempotency ledger plus spec-defined compensations |
