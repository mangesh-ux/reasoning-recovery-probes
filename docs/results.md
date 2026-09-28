# Pilot P0 results

**Status:** completed operational feasibility campaign; scientific block for a
later activation-probe study.

This is an aggregate-only record. It contains no benchmark rows, prompts,
completions, token IDs, or per-problem identities.

## Frozen run identity

| Field | Value |
| --- | --- |
| Run | run-0f4e9dfd145f5dcb5642 |
| Source commit for all 720 raw receipts | 48d5efa915e3a5a95561d6963d9149ecbe2dceea |
| Frozen configuration semantic canonical-JSON SHA-256 | 2ce61102dbb987394fb488bb95eeeb855f5ee2ff77011d9474d5d3536b0de432 |
| Manifest semantic canonical-JSON SHA-256 | 3efe3ce41a812f03bd5270381bc850d994a0026e82e590c588f8732802134c84 |
| Model | Qwen/Qwen3-1.7B at 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| Dataset | HuggingFaceH4/MATH-500 test split at 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be |
| Runtime | Native Windows CUDA, BF16, PyTorch 2.11.0+cu128, Transformers 4.52.4, math-verify 0.9.0 |

The frozen P0 contract and label semantics remain in
[pilot_p0_protocol.md](pilot_p0_protocol.md). The raw artifacts remain local
and ignored by Git.

The immutable aggregate analysis receipt is
summary-349a313ceafea6b5a779e718427e575b413f421f080b57daa290523b9ad6c2d5--b886e8bff9a05817.json
(record type P0_DERIVED_ANALYSIS). The matching full operational summary is
summary-708299d02478c5ba2d2160dbdece860e94e1b459465be1426d3f53258969b3df--3869ebebc573d1dd.json.
Both remain local immutable aggregate artifacts; this document records their
reviewed public-safe statistics.

## Completion, availability, and evaluation

| Measure | Result |
| --- | ---:|
| Problems completed | 30 / 30 |
| Base trajectories completed | 120 / 120 |
| Expected checkpoint receipts recorded | 600 / 600 |
| Available checkpoints | 464 / 600 |
| Base final answers: correct / incorrect / non-evaluable | 67 / 1 / 52 |
| Conditional base accuracy among evaluable answers | 67 / 68 = 98.53% |
| Base final-answer non-evaluable rate | 52 / 120 = 43.33% |
| Forced answers: correct / incorrect / non-evaluable | 175 / 5 / 284 |
| Forced-answer non-evaluable rate among available checkpoints | 284 / 464 = 61.21% |
| Evaluator errors | 0 |
| Failed receipts / CUDA OOM / interrupted-unknown | 0 / 0 / 0 |

Checkpoint availability is a property of the original saved trajectory and its
thinking boundary; unavailable does not mean a failed forced generation.

| Generated-token checkpoint | Available / 120 | Unavailable reason summary |
| --- | ---:| --- |
| 256 | 120 | None |
| 512 | 120 | None |
| 1024 | 110 | 9 after the think close; 1 too short |
| 2048 | 64 | 15 after the think close; 41 too short |
| 3072 | 50 | 3 after the think close; 67 too short |

The 120 base trajectories had a median length of 2,611 generated tokens
(mean 2,867.39; minimum 991; p25 1,780; p75 4,096; p90 4,096; maximum
4,096). Forty-seven base trajectories and 261 forced completions reached
their configured generation cap; those records remain present rather than
being repaired or replaced.

## Transitions

The primary analysis applies the frozen stricter eligibility rule. In
particular, it requires an available checkpoint, natural-EOS base and forced
completions, a base think-close marker, and evaluable answers. The
observed-before-censoring view requires only evaluable answers and is reported
as a sensitivity description, not as a replacement analysis.

| Label | Strict primary | Observed before analytic censoring |
| --- | ---:| ---:|
| C_TO_C | 126 | 158 |
| C_TO_W | 0 | 0 |
| W_TO_C | 2 | 3 |
| W_TO_W | 0 | 0 |
| Total labelled transitions | 128 | 161 |

| Checkpoint | Strict labelled transitions | C_TO_C | C_TO_W | W_TO_C | W_TO_W |
| --- | ---:| ---:| ---:| ---:| ---:|
| 256 | 36 | 35 | 0 | 1 | 0 |
| 512 | 36 | 35 | 0 | 1 | 0 |
| 1024 | 43 | 43 | 0 | 0 | 0 |
| 2048 | 11 | 11 | 0 | 0 | 0 |
| 3072 | 2 | 2 | 0 | 0 | 0 |

| Checkpoint | Strict W_TO_C rate among labelled | Strict W_TO_W rate among labelled | Recovery rate conditional on wrong checkpoint |
| --- | ---:| ---:| ---:|
| 256 | 1 / 36 = 2.78% | 0 / 36 = 0.00% | 1 / 1 = 100%; one event only |
| 512 | 1 / 36 = 2.78% | 0 / 36 = 0.00% | 1 / 1 = 100%; one event only |
| 1024 | 0 / 43 = 0.00% | 0 / 43 = 0.00% | Not estimable; no wrong checkpoint |
| 2048 | 0 / 11 = 0.00% | 0 / 11 = 0.00% | Not estimable; no wrong checkpoint |
| 3072 | 0 / 2 = 0.00% | 0 / 2 = 0.00% | Not estimable; no wrong checkpoint |

Of 600 checkpoint receipts, 128 satisfied the strict primary mapping and 472
were explicitly excluded by the frozen rule. Those exclusions are retained;
they must not be relabelled as incorrect outcomes or discarded silently.

## Same-problem contrasts and clustering

The strict W_TO_C result consists of two events at two checkpoint positions,
but both arise from one unique problem. Its problem-level Herfindahl index is
1.0 and that problem contributes 100% of strict W_TO_C events. There are zero
strict W_TO_W events.

There are zero same-problem/same-checkpoint groups containing both W_TO_C and
W_TO_W. The observed-before-censoring view is not materially different for
this decision: it has three W_TO_C events from two problems, zero W_TO_W
events, and zero mixed groups.

## Operational measurements

| Stage | Generated tokens | Recorded generation time | Aggregate tokens/s | Peak reserved VRAM |
| --- | ---:| ---:| ---:| ---:|
| Base trajectories | 344,087 | 24,835.38 s | 13.855 | 4,273,995,776 B |
| Forced checkpoint answers | 207,347 | 12,582.01 s | 16.480 | 4,066,377,728 B |
| Combined | 551,434 | 37,417.40 s (10.394 h) | 14.737 | 4,273,995,776 B peak |

The RTX 3070 Ti Laptop GPU reports 8,589,410,304 bytes of memory. The largest
measured reserved peak therefore left 4,315,414,528 bytes (4.019 GiB) of
headroom for P0 generation. The full local run directory measured 11,991,718
bytes (11.436 MiB) at final inspection; the immutable summary recorded
11,916,627 bytes before its own final write.

## Viability verdict

**Can this setup meaningfully answer the activation-recovery question as
proposed? No.**

P0 operationally succeeded, but the primary study population has no
wrong-non-recovery class: strict W_TO_C is 2, strict W_TO_W is 0, and both
strict recovery events are concentrated in one problem. There is consequently
no defensible grouped train/validation/test classification target and no
within-problem recovery-versus-non-recovery comparison. The sensitivity view
does not restore that missing class.

No activation data were collected, no A/B/C model was fitted, and no final
experiment was frozen or executed. Changing the model, dataset, selection,
checkpoints, decoding, or label semantics to manufacture events would be a
new scientific protocol and requires explicit future authorization.

P0 does not establish that hidden activations predict recovery, that continued
reasoning is valuable, that a stopping rule is safe, or that checkpoint
interventions are causal.

## Work-item status

| Work item | Status | Outcome |
| --- | --- | --- |
| RR-01 | DONE | Native Windows CUDA runtime was qualified and measured. |
| RR-02 | DONE | The immutable 30 x 4 P0 campaign completed. |
| RR-03 | DONE | Aggregate-only feasibility analysis and same-problem contrasts were recorded. |
| RR-04 | DONE | Viability review reached a scientific block. |
| RR-05 | BLOCKED | A final protocol cannot be frozen for an absent comparison class. |
| RR-06 | BLOCKED | A/B/C comparison is not meaningful with zero W_TO_W examples. |
| RR-07 | BLOCKED | No activation representation may be selected for the blocked study. |
| RR-08 | BLOCKED | No valid grouped two-class evaluation can be frozen. |
| RR-09 | DONE | The skeptical validity review identified the fatal event-yield issue. |
| RR-10 | DONE | P0 hardware measurements and conditional future guidance were documented. |
| RR-11 | BLOCKED | A final reusable pipeline is not authorized. |
| RR-12 | BLOCKED | No final activation experiment may execute. |
| RR-13 | BLOCKED | There are no activation-probe results to analyze. |
| RR-14 | DONE | Aggregate results, validity, hardware, final-design decision, and a machine-readable summary are present. |
