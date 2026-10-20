# Benchmark summary

Simulated on a virtual clock. Latencies come from configured provider latency distributions, so they measure the architecture's scheduling, not real providers.
Method, assumptions and limits are in `benchmarks/README.md`. Wall time per benchmark is
listed at the end.

## policy_gate (1000 adversarial conversations per mode)

| mode | violating proposed | violating executed | flagged after execution | false blocks on legit writes | check latency |
|---|---|---|---|---|---|
| post-hoc detector (baseline) | 1677 | 1437 | 1348 | 0 | 3000 ms modeled, after execution |
| pre-execution checker | 1732 | 0 | n/a | 0 | p50 8.0 us, p99 44.9 us (measured) |

| violation kind | baseline proposed | baseline executed | checker proposed | checker executed |
|---|---|---|---|---|
| early_mark_delivered | 391 | 376 | 415 | 0 |
| refund_over_limit | 746 | 745 | 836 | 0 |
| repeat_write:issue_refund | 121 | 42 | 89 | 0 |
| repeat_write:mark_delivered | 144 | 23 | 147 | 0 |
| second_refund | 275 | 251 | 245 | 0 |

Logged verdicts recomputed from the log: 7732, mismatches: 0.

## grounding_gate (20000 sentences, 520 corrupted)

| config | verifier calls | corrupted caught | corrupted released | false blocks on clean |
|---|---|---|---|---|
| tier 1 only (runtime default) | 0 | 100.0% | 0 | 3.1% |
| tier 1 + lexical scorer + tier 3 | 1702 | n/a | 0 | 31.5% |
| every sentence to tier 3 (baseline) | 20000 | 94.2% | 30 | 0.0% (assumed) |

False blocks by category (tier 1 only):

| category | clean sentences | false blocks | rate |
|---|---|---|---|
| claimless | 8330 | 0 | 0.0% |
| eta_clock | 1586 | 0 | 0.0% |
| eta_minutes | 1465 | 0 | 0.0% |
| order_id | 2307 | 0 | 0.0% |
| paraphrase_clock | 385 | 0 | 0.0% |
| paraphrase_money | 603 | 0 | 0.0% |
| pay | 2252 | 0 | 0.0% |
| refund | 1945 | 0 | 0.0% |
| relative_eta | 607 | 607 | 100.0% |

Tier 1 latency: p50 48.7 us, p99 196.4 us. Share of sentences needing tier 3 with the lexical scorer: 8.5%.

## speculation_latency (voice to voice as the caller hears it)

| eager threshold | persona | v2v p50 ms | v2v p95 ms | speculations | cancelled share | wasted output tokens |
|---|---|---|---|---|---|---|
| 0.3 | clean | 776 | 1.3e+03 | 200 | 0.0% | 0.0% |
| 0.3 | slow_speaker | 945 | 1.41e+03 | 243 | 16.5% | 0.1% |
| 0.4 | clean | 776 | 1.3e+03 | 200 | 0.0% | 0.0% |
| 0.4 | slow_speaker | 940 | 1.42e+03 | 228 | 11.0% | 0.1% |
| 0.5 | clean | 776 | 1.3e+03 | 200 | 0.0% | 0.0% |
| 0.5 | slow_speaker | 941 | 1.42e+03 | 214 | 5.1% | 0.0% |
| 0.6 | clean | 810 | 1.34e+03 | 200 | 0.0% | 0.0% |
| 0.6 | slow_speaker | 971 | 1.43e+03 | 211 | 3.8% | 0.0% |
| 0.7 | clean | 927 | 1.46e+03 | 200 | 0.0% | 0.0% |
| 0.7 | slow_speaker | 1.1e+03 | 1.64e+03 | 206 | 1.5% | 0.1% |
| off | clean | 1.05e+03 | 1.58e+03 | 0 | n/a | 0.0% |
| off | slow_speaker | 1.29e+03 | 1.9e+03 | 0 | n/a | 0.0% |

## endpointing (3000 turns, 1360 with a mid-sentence pause)

| controller | transcripts | premature cutoffs | on paused turns | mean end latency ms | p95 end latency ms |
|---|---|---|---|---|---|
| hybrid | bare partials | 2.4% | 5.3% | 755 | 996 |
| fixed_800ms | bare partials | 14.1% | 31.2% | 790 | 799 |
| text_only | bare partials | 4.3% | 9.6% | 86.7 | 130 |
| hybrid | noisy punctuation | 5.0% | 11.1% | 924 | 2.18e+03 |
| fixed_800ms | noisy punctuation | 14.1% | 31.2% | 790 | 799 |
| text_only | noisy punctuation | 11.6% | 25.7% | 763 | 3e+03 |
| hybrid | punctuated partials | 2.4% | 5.3% | 755 | 996 |
| fixed_800ms | punctuated partials | 14.1% | 31.2% | 790 | 799 |
| text_only | punctuated partials | 4.3% | 9.6% | 10 | 18.9 |

## idempotency (500 crash seeds per config)

| config | crashed runs | intended writes | applied | double applies | missed |
|---|---|---|---|---|---|
| switchboard | 500 | 745 | 745 | 0 | 0 |
| switchboard_ledger_lost | 500 | 745 | 745 | 0 | 0 |
| switchboard_downstream_ignores_key | 500 | 745 | 863 | 118 | 0 |
| naive_retry | 500 | 745 | 993 | 248 | 0 |

## multiparty (courier and customer flow, 32 seeds per task)

| persona | pass^1 | pass^4 | pass^8 |
|---|---|---|---|
| clean | 1 | 1 | 1 |
| wer_5 | 1 | 1 | 1 |
| wer_12 | 0.938 | 0.762 | 0.556 |
| slow_speaker | 1 | 1 | 1 |
| barge_in_heavy | 1 | 1 | 1 |
| overall | 0.988 | 0.952 | 0.911 |

## cost (per conversation minute, prices are assumptions)

| routing | cache | total $/min | LLM $/min | LLM share | turns to talker |
|---|---|---|---|---|---|
| talker_thinker | auto | 0.0688 | 0.0134 | 19.4% | 36.4% |
| talker_thinker | always_hit | 0.0667 | 0.0112 | 16.8% | 36.4% |
| talker_thinker | never | 0.0807 | 0.0252 | 31.3% | 36.4% |
| thinker_only | auto | 0.0737 | 0.0173 | 23.4% | 36.1% |
| thinker_only | always_hit | 0.0716 | 0.0151 | 21.1% | 36.1% |
| thinker_only | never | 0.0912 | 0.0347 | 38.1% | 36.1% |

## wall time

| benchmark | seconds |
|---|---|
| policy_gate | 27.4 |
| grounding_gate | 1.2 |
| speculation_latency | 5.7 |
| endpointing | 1.7 |
| idempotency | 25.5 |
| multiparty | 4.1 |
| cost | 3.0 |
| total | 68.7 |
