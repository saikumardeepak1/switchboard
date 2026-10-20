# Switchboard Architecture

Switchboard is a runtime for enterprise voice agents. One conversation can hold several live call
legs at once (a courier on one line, the customer on another), every action is checked against
compiled policy before it runs, and every spoken number is checked against evidence before it is
heard. The decisions behind this layout are in [DECISIONS.md](DECISIONS.md).

## Planes

```
             MEDIA PLANE                         CONTROL PLANE (this repo)
 caller ──► transport (Twilio WS / WebRTC) ──► Leg pipeline
            mu-law 8k frames                     │  endpointing controller (ADR-007)
                                                 │  speculative turn engine (ADR-006)
                                                 ▼
                                         Conversation actor (single writer, ADR-002)
                                                 │  event log  ◄── replay, activity stream
                                                 │  procedure graph state (ADR-004)
                                                 ▼
                                         Brain: router ► talker | thinker (ADR-008)
                                                 │ proposed tool call          │ sentence stream
                                                 ▼                             ▼
                                         Policy checker (ADR-004)      Grounding gate (ADR-005)
                                                 │ allowed                     │ released
                                                 ▼                             ▼
                                         Effect runner (ADR-009)        TTS ► playback cursor
                                                 │
                                   tools, knowledge (ADR-010), memory (ADR-011), child legs (ADR-003)
```

The media plane is out of scope beyond a transport adapter: a production deployment puts an SFU or
SIP edge in front, pinned per call, so an agent crash never drops the caller's audio.

## Package layout

```
src/switchboard/
  core/        clock, ids, events, event store, actor runtime          (no deps on other packages)
  policy/      rule language, procedure graph, compiler, checker        (core)
  grounding/   claim extraction, normalization, tiered gate             (core)
  turn/        endpointing controller, speculative turn engine, playback cursor  (core)
  knowledge/   bitemporal fact store, status-page ingestor, as-of retrieval      (core)
  memory/      scoped typed memory, TTL, consolidation                   (core)
  effects/     tool registry, idempotency ledger, compensations          (core)
  bundle/      agent-as-code loader and compiler, content hash           (core, policy, effects)
  brain/       LLM interface, router, prompt assembly, scripted fake LLM (core)
  runtime/     Conversation actor, Leg, child-leg protocol, admission    (everything above)
  providers/   STT/TTS/LLM/transport interfaces, simulated providers, optional real adapters
  sim/         virtual-time simulator, simulated callers, scenarios, pass^k, latency report
  telemetry/   per-turn spans, activity stream projection
  api/         FastAPI app: activity stream, conversations, Twilio media stream endpoint
examples/delivery_support/   reference agent bundle (courier and customer multi-party flow)
benchmarks/                  reproducible runs that produce the numbers in README
```

Rule: a package may import only from packages listed to its right above. `core` imports nothing
from the project. Tests live in `tests/<package>/`.

## Core contracts (`switchboard.core`)

These are the shared types every other package builds on. They are the only cross-package
contract, so they change only with a decision record.

### Clock

```python
class Clock(Protocol):
    def now(self) -> float: ...                       # seconds, monotonic within a run
    async def sleep(self, seconds: float) -> None: ...

class VirtualClock(Clock):   # deterministic; advance() runs due sleepers in time order
    def __init__(self, start: float = 0.0): ...
    async def advance(self, seconds: float) -> None: ...
    async def run_until_idle(self) -> None: ...       # advance to each next timer until none remain

class RealClock(Clock): ...
```

No module calls `time.time()`, `time.monotonic()` or `asyncio.sleep()` with a non-zero delay
directly. They take a `Clock`.

### Ids

`new_id(prefix: str, rng: random.Random) -> str` gives readable, seedable ids such as
`conv_3f9a1c2e`. Determinism matters because simulation runs must be byte-identical per seed.

### Events

All events are frozen pydantic v2 models deriving from `Event`:

```python
class Event(BaseModel, frozen=True):
    type: str                 # discriminator, e.g. "turn.committed"
    conversation_id: str
    leg_id: str | None
    seq: int                  # assigned by the store on append, 1-based, gapless per conversation
    at: float                 # clock time
    bundle_id: str            # release bundle the conversation is pinned to
    causation_id: str | None  # id of the event that caused this one
    event_id: str
```

Event catalog (type string, payload fields):

| type | payload |
|---|---|
| `conversation.started` | tenant_id, customer_id |
| `leg.opened` | role (`inbound` / `outbound_child`), party (e.g. `courier`), parent_leg_id, goal |
| `leg.closed` | reason |
| `turn.speculation_started` | turn_id, partial_text |
| `turn.speculation_cancelled` | turn_id |
| `turn.committed` | turn_id, text, endpointing_ms |
| `brain.routed` | turn_id, model_tier (`talker`/`thinker`), reason |
| `tool.proposed` | call_id, tool, args |
| `policy.verdict` | call_id, allowed, rule_ids, repair_hint |
| `tool.executed` | call_id, tool, result, idempotency_key, replayed (bool) |
| `tool.compensated` | call_id, compensation_tool, result |
| `speech.sentence_gated` | utterance_id, sentence, decision (`release`/`block`/`escalate`), tier, claims |
| `speech.played` | utterance_id, text_played, interrupted (bool) |
| `procedure.advanced` | step, state |
| `child.result` | child_leg_id, result |
| `knowledge.retrieved` | fact_ids, as_of, texts (empty with content capture off) |
| `memory.loaded` | count, texts (empty with content capture off) |
| `handoff.requested` | reason |
| `conversation.ended` | outcome |

Events are a discriminated union on `type` so the store can deserialize any of them.

### Event store

```python
class EventStore(Protocol):
    async def append(self, conversation_id: str, events: Sequence[NewEvent],
                     expected_seq: int) -> list[Event]: ...   # raises ConcurrencyError on mismatch
    async def read(self, conversation_id: str, after_seq: int = 0) -> list[Event]: ...
    async def read_all(self, *, types: set[str] | None = None,
                       since: float | None = None) -> AsyncIterator[Event]: ...
```

`NewEvent` is an event without `seq` and `event_id`; the store assigns both. Implementations:
`InMemoryEventStore` and `SqliteEventStore` (stdlib `sqlite3` run in a thread, one table with
`UNIQUE(conversation_id, seq)`). Both pass the same contract test suite.

### Actor runtime

```python
class Actor(Protocol[M]):
    async def handle(self, message: M) -> None: ...

class ActorRuntime:
    def __init__(self, factory: Callable[[str], Actor], clock: Clock): ...
    async def send(self, key: str, message: object) -> None: ...   # FIFO per key, one handler at a time
    async def ask(self, key: str, message: object) -> object: ...   # send plus await reply
    async def stop(self, key: str) -> None: ...
```

Messages for one key are processed strictly one at a time in arrival order; different keys run
concurrently. An exception in `handle` is recorded and does not kill other keys.

## Package contracts

### policy

Rule file format (`policies/*.yaml`):

```yaml
procedure:
  steps: [greet, verify_identity, resolve, close]
  require_before:
    lookup_order: [verify_identity]
    mark_delivered: [verify_identity]
rules:
  - id: refund_limit
    when: {tool: issue_refund}
    require: "args.amount <= 25"
    hint: "Refunds above $25 need a human. Offer a handoff."
  - id: delivered_needs_confirmation
    when: {tool: mark_delivered}
    require: "evidence.customer_confirmed_address == true"
    hint: "Confirm the address change with the customer first, by calling them if needed."
  - id: one_refund_per_conversation
    when: {tool: issue_refund}
    require: "count(tool.executed, tool=issue_refund) < 1"
```

The `require` expression language is a small safe grammar parsed by the project (no `eval`):
comparisons, boolean `and`/`or`/`not`, dotted paths into `args`, `evidence`, `state`, numbers,
strings, booleans, and `count(event_type, field=value)` over the conversation's events.

API: `compile_policies(docs) -> CompiledPolicy`; `CompiledPolicy.check(call, ctx) -> Verdict`
where `Verdict(allowed: bool, rule_ids: list[str], repair_hint: str | None)`. Checking is pure and
runs in microseconds; a benchmark asserts p99 under 1 ms for 50 rules.

### grounding

`extract_claims(sentence) -> list[Claim]` where `Claim(kind, raw, value, span)` and kind is one of
`money`, `number`, `time`, `date`, `duration`, `identifier`. Normalization handles "$12.50",
"twelve fifty", "7:45 pm", "in 20 minutes", "order 48213".

`GroundingGate.review(sentence, evidence: Evidence) -> GateDecision` with tiers per ADR-005.
`Evidence` is the set of values from tool results, retrieved facts and memory for the current turn.
Tier 3 is an async `Verifier` protocol; the gate returns `escalate` and the caller races it against
playback.

### turn

`EndpointingController.observe(signal) -> TurnSignal | None` consumes VAD frames, partial and final
transcripts on a clock, and emits `eager_end`, `end`, or `resumed`.
`SpeculativeTurnEngine` subscribes to those signals, starts a brain call on `eager_end` into a
private buffer, cancels on `resumed`, and releases on `end`.
`PlaybackCursor` tracks which characters of an utterance have been played, so barge-in truncates
history to `text_played`.

### knowledge

`BitemporalStore.assert_fact(subject, predicate, value, valid_from, valid_to=None, recorded_at)`,
`close(fact_id, valid_to)`, `query(subject=None, predicate=None, as_of: float, known_at: float | None)`.
`StatusPageIngestor.ingest(payload)` parses the Atlassian Statuspage `incidents.json` shape and
opens or closes incident facts. `retrieve(query, as_of, k)` ranks by lexical match plus a recency prior.

### memory

`MemoryStore.write(tenant_id, customer_id, kind, content, ttl_s)`, `read(tenant_id, customer_id,
now)` filtered by TTL, `forget(tenant_id, customer_id)` hard delete. `consolidate(events) ->
list[MemoryWrite]` turns a finished conversation's events into memory writes.

### effects

`ToolSpec(name, effect: read|write, params schema, idempotency: list[str], compensate: str | None)`.
`EffectRunner.run(call, ctx) -> ToolResult` checks the ledger for the derived key, runs at most once,
records, and returns the recorded result on repeat with `replayed=True`.

### brain

```python
class LLM(Protocol):
    async def stream(self, request: BrainRequest) -> AsyncIterator[BrainChunk]: ...
```

`BrainChunk` is either text delta or a complete tool call. `ScriptedLLM` is a deterministic fake
that plays back scripted responses with configured latency on the injected clock, including
deliberately wrong numbers and policy-violating tool calls for testing. `Router.route(turn,
state) -> Tier`. `assemble_prompt` emits a stable prefix (system, policies, tool schemas in a fixed
order) followed by the volatile suffix, so provider prefix caches hit.

### runtime

`Conversation` is the actor. It owns legs, the event log, procedure state, and the evidence set.
`Leg` wires transport, endpointing, the speculative engine, the gate and playback for one party.
`spawn_child(goal, party, result_schema)` opens an outbound child leg whose result arrives as a
`child.result` event to the parent. `AdmissionController` enforces per-tenant concurrent-call caps.

### sim

`Simulation(seed)` builds a virtual clock, simulated STT/TTS with latency distributions and ASR
word-error injection, and `SimulatedCaller` scripts (including barge-in and pauses). Scenarios
return a `ScenarioResult` with pass/fail per assertion and per-turn latency. `pass_hat_k(results,
k)` computes pass^k.

### telemetry and api

`ActivityStream` is a projection over the event store with filters (conversation, leg, type, time
range) and frequency buckets for charts. The API serves it at `GET /v1/activity` and
`GET /v1/activity/histogram`.

## Testing rules

- Test first, always. A feature lands with the test that failed before it.
- Unit tests per package, contract tests for every Protocol with more than one implementation,
  property tests (hypothesis) for the rule parser and claim normalization, and scenario tests in
  `tests/sim/` that run full multi-party calls on the virtual clock.
- `pytest -q` runs everything offline in seconds. No network, no API keys.
