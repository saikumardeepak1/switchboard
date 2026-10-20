# Benchmarks

Every number here comes from simulation. Calls run on a virtual clock (ADR-014) with simulated
callers, a simulated recognizer and synthesizer, and a scripted model. Nothing talks to a real
provider, and nothing here is a measurement of production traffic.

What that means for each kind of number:

- **Latency** (voice to voice, end of turn) comes from configured provider latency distributions
  (`switchboard.sim.SimConfig`, the personas in `switchboard.sim.caller`). It measures how the
  architecture schedules work around those distributions, not how fast any provider is.
- **Policy check and tier 1 gate latency** are real `time.perf_counter` measurements of the pure
  checking code on this machine, not model calls.
- **Counts** (violations executed, double applies, false blocks) are exact for the simulated
  workload, against ground truth logged by the fault injector, never by the component scored.
- **Cost** uses list prices that are assumptions, with sources in `switchboard/sim/cost.py`.

Run everything with `make bench` (about a minute on a 10-core laptop; benchmarks fan out over
worker processes and results are aggregated in seed order, so they do not depend on worker count).
Run one with `uv run python -m benchmarks.<name> [--n N] [--workers W]`. Results land in
`results/<name>.json` and `results/summary.md`. The committed results came from `make bench`.

## The workload

All call-level benchmarks drive the reference agent in `examples/delivery_support` through the
public runtime API: `Runtime.start`, `Runtime.recover`, `LegIO`, `replay`. A courier calls in
because they cannot mark order 48213 delivered; the agent verifies them, reads the order and
chat, calls the customer on a child leg while the courier keeps talking (a pay question), and
marks the order delivered only after the customer confirms. Some scenarios add a legitimate $10
refund.

The model is `switchboard.sim.delivery.DeliveryScript`, a deterministic function of the request
(it reads the last message, tool results and evidence the way a model would). It asks again when
it cannot place what it heard. A scripted model is the main limit of every call-level result:
real models fail in ways no script anticipates. The adversary (`switchboard.sim.adversarial`)
is how model misbehaviour enters: it wraps the script and injects policy-violating writes,
corrupted numbers and repeated writes at seeded rates, logging each fault.

Callers (`switchboard.sim.caller`) follow beat scripts (say, pause, wait for a phrase, barge in,
backchannel, hang up) with persona knobs: speaking rate, mid-sentence pauses, word error rate,
recognizer lag, background noise, patience, and a habit of interrupting long agent turns.

## Benchmarks

### policy_gate (ADR-004)

1000 adversarial conversations per mode. The adversary proposes, per eligible model call, a
violating write with probability 0.25 (mark delivered before the customer confirmed, refund over
$25, second refund) and repeats an executed write with probability 0.10.

- **Baseline (post-hoc detector).** Modeled on the reference design: the bundle is copied with
  its rules and procedure prerequisites removed, so nothing is checked before execution. A
  detector flags each executed violating action with recall 0.95 about 3 s after it ran. Its
  false-positive rate is not public; it is set to 0, which favours the baseline.
- **Switchboard.** The bundle as written, checked by the runtime before execution.

Check latency: every logged verdict is recomputed from the log prefix with the same context the
runtime builds (folded evidence, incremental event index, procedure state) and timed. The
recomputed verdicts must equal the logged ones; the mismatch count is reported.

Limits: the baseline runs on this runtime with policy removed, so it inherits the runtime's
intent-keyed write dedupe (a repeated identical write replays instead of executing). That is why
the baseline's "executed" is below "proposed" for repeats and for second refunds with the same
amount as the legitimate one. The reference design has no such dedupe as far as is public, so
the baseline is, if anything, flattered.

### grounding_gate (ADR-005)

20,000 sentences generated from agent templates (pay totals, refunds, order ids, clock ETAs,
minute ETAs, claimless bridge speech) each paired with its tool result and distractor values.
Some templates say numbers the way people do: "twelve fifty", "a quarter to eight", "in about 20
minutes". 5% of sentences that carry a money amount, ETA or order id get one value corrupted.
Evidence is built the way the runtime builds it: tool results plus the time of day the sentence
is spoken at (`Runtime.day_start`, ADR-005 Revision). The relative ETA template fixes that time
(the ETA minus the minutes said); every other sentence draws one from a separate stream, so the
corpus itself is unchanged.

Configurations: the runtime default gate (tier 1 only; a sentence with no claims is released;
there is no verifier, so an escalation fails closed and is counted as a block), the gate with the
reference `LexicalScorer` as tier 2 plus a simulated tier 3 verifier, and the baseline that sends
every sentence to tier 3. Tier 2 scores only claimless sentences that make a factual assertion (a
capitalized word not opening the sentence, or a policy word such as refund or deliver); other
filler releases unscored. The simulated verifier is an oracle with recall 0.95 and no false
positives, which favours the baseline and the tier 3 path.

Limits: this is a gate-level benchmark, not a call-level one. The corpus is template generated,
so the paraphrase categories are only as varied as the templates.

### speculation_latency (ADR-006)

The happy delivery call, 40 seeds per cell, under a clean speaker and a slow speaker who pauses
mid-sentence, sweeping the bundle's eager threshold from 0.3 to 0.7 and "off" (eager threshold
equal to the end threshold, so nothing starts before the turn commits).

Voice to voice comes from `TurnTimeline`: from the end of the caller's speech (commit time minus
the endpointing wait) to the first TTS chunk sent to them (`speech.played.first_audio_at`). The
caller's own measurement, from the end of what they said to the first agent audio that reached
them, is reported beside it as a cross-check. They agree per turn to well under a millisecond,
except when the agent commits during a mid-sentence pause: the timeline then counts a turn from
the pause, which the caller does not, so slow-speaker percentiles differ by about 1 ms. Wasted generation is reported two ways: speculations cancelled over
started, and model output tokens produced by calls that were later cancelled.

Configured latencies: model time to first token lognormal (thinker p50 0.35 s, p95 0.8 s;
talker p50 0.15 s, p95 0.3 s), 60 tokens per second, TTS first byte 0.15 s, recognizer final lag
p50 0.08 s (clean) to 0.1 s (slow).

### endpointing (ADR-007)

3000 generated turns from 300 simulated callers, each with their own speaking rate and pause
habit. Templates mark hesitation points; at each the caller pauses with probability 0.7. Some
pauses follow text that already looks complete, the hard case for a text model. Three
controllers see the same stream: the hybrid controller, a fixed 800 ms silence timer, and a
text-only rule (end when the heuristic completeness score reaches 0.8, 3 s fallback).

Three transcript conditions: bare partials with punctuated finals (the headline), recognizer
style noisy punctuation on finals, and punctuation on every partial (flatters text models).

Limits: the text model in both "hybrid" and "text_only" is the same heuristic scorer, and the
transcripts have no word errors. This benchmark says nothing about a learned turn model.

### idempotency (ADR-009)

500 seeds per configuration of the refund flow (two writes). Each seed picks one write and one
crash point (before the downstream request, after the downstream applied it, after the ledger
recorded it but before the event log did, after the event log did), kills the process there
(every later append fails), abandons the conversation, recovers it with `Runtime.recover` on a
new runtime, and lets the caller say hello on a fresh line. Applied writes downstream are
compared with intended writes (allowed write proposals in the final log).

Configurations cross ledger durability with whether the downstream service honours the
idempotency key. `naive_retry` (no ledger, no key at the downstream) is the baseline.

Limit: a killed tool is modeled by raising inside it, so the runner marks the ledger claim failed
instead of leaving it pending until its lease runs out. Recovery reaches the same state; a real
recovery would first wait out the lease.

### multiparty (ADR-003, ADR-014)

pass^k for k = 1, 4, 8 with the tau-bench estimator (mean over tasks of C(c, k) / C(n, k)),
32 seeds per task. Tasks: the customer confirms (mark delivered exactly once, only after the
confirmation) and the customer declines (never mark it), each under five personas applied to
both humans: clean, 5% WER, 12% WER, slow speaker, barge-in heavy.

### cost (ADR-008)

Per-minute cost for talker plus thinker routing against thinker only, under three prompt-cache
conditions, on a chattier version of the happy call (the courier says "Okay, thanks." and
"Great." between requests, which is what the router sends to the talker). Token counts come from
the scripted model, one token per word, scaled by 1.3. The scripted prompts and replies are far
shorter than a production agent's, so the LLM share here is a floor.

## Findings worth reading before the numbers

These come out of the committed run. They are reported as measured, not tuned away. The results
committed before this run predated the review fixes (cost and slow-speaker latency had already
moved on the code they were committed with), so compare against this run, not that one.

1. **Relative ETAs ground only when the runtime knows the time of day.** "Your order should
   arrive in about 20 minutes" against a tool result that says 7:45 PM was blocked 100% of the
   time, because the runtime built evidence with no current time. With `Runtime.day_start` set
   (the simulator sets it) there are no false blocks in that category. `day_start` defaults to
   None because `RealClock` is monotonic and has no date; a deployment that leaves it unset still
   fails these sentences closed.
2. **The lexical scorer is now harmless, not useful.** It blocked about two thirds of claimless
   bridge speech. Tier 2 now scores only sentences that assert something (a name, or a policy
   word like refund), so filler releases unscored and false blocks with the scorer drop to 0;
   3.3% of sentences go to the verifier. Whether the scorer catches anything a verifier would not
   is untested: the corpus has no claimless corrupted sentences, so this benchmark cannot show a
   tier 2 catch.
3. **Hybrid endpointing is not faster than a fixed timer here; it is safer.** With bare partials
   its mean end latency is close to the 800 ms timer, because its adaptive pause scale grows for
   callers who pause, but it cuts callers off far less often. ADR-007 motivates the hybrid partly
   by the latency of fixed timeouts; in this simulation the win is premature cutoffs, not latency.
   A text-only rule is much faster when transcripts are clean and loses its edge once final
   punctuation is noisy.
4. **`TurnTimeline` voice to voice now is voice to voice.** It used to end at the first
   `speech.played` event, which is appended when an utterance finishes playing, and reported 1.3
   to 3.5 s where the caller heard audio after 0.65 to 0.9 s. It now ends at
   `speech.played.first_audio_at`, and speculative turns keep their routing tier and what they did
   before the commit. The speculation benchmark reads the timeline.
5. **Caching matters more than routing at these prices.** About a third of turns go to the
   talker in the chatty call. Routing cuts the per-minute total by about 3% (0.0747 to 0.0727
   $/min with a cold cache per call); a warm shared prefix against no caching moves it by about 16
   to 20%. Telephony, STT and TTS stay the majority of per-minute cost either way, and the scripted
   prompts are short, so a production agent's LLM share would be larger.

## Reproducing

`make bench` regenerates everything. Seeds are fixed, so counts and simulated latencies
reproduce exactly; `perf_counter` latencies and wall times vary with the machine.
