# V2-A1 Final-Answer Validation Amendment

**Protocol ID:** v2-recovery-activation-probe-20261002-a1  
**Decision date:** 2026-10-02  
**Status:** explicitly authorized before any V2 scientific model outputs; execution remains conditional on all operational and scientific gates.  
**Base protocol:** [V2 Recovery Representation Study Protocol](v2_protocol.md).

## 1. Reason for a separate amendment

The original V2 loader requires valid question, final-answer, difficulty, and
topic fields in every source row before selecting its cohort. It encountered
a global final-answer data-contract failure. That failure and its receipts
remain part of the original V2 record; A1 does not turn the failed run into a
successful original V2 run or overwrite its artifacts.

An explicitly authorized, local CPU-only source audit inspected the pinned
source's schema and constructed a diagnostic shadow cohort without model
inference, evaluator calls, or performance inspection. Its reviewed,
aggregate-safe observations were:

| Audit quantity | Recorded value |
|---|---:|
| Source rows | 103,022 |
| Empty final-answer fields | 2 |
| Difficulty of both affected rows | 5.0 |
| Affected rows inside the fixed difficulty-7/8 cohort | 0 |
| Selected shadow-cohort rows passing all required schema checks | 160 |
| Grouped train / validation / test counts | 96 / 32 / 32 |
| Replacements or answer-based substitutions | 0 |

These are source-contract diagnostics, not recovery events, probe results,
model capability evidence, or a scientific success. The shadow cohort is not
an official model-backed campaign. The audit's physical-file provenance,
ordered original-index proof, invalid-field ledger, intent, and aggregate
outputs remain private and preserved.

On 2026-10-02 the user explicitly authorized the narrow validation-scope
amendment below and one paid execution attempt within a USD 25 total Runpod
cap, including prior spending. This is neither permission for additional
attempts nor a guarantee that the campaign will finish or yield evaluable
results. Qualification, live affordability, receipt-integrity, budget, and
scientific stop gates still apply.

## 2. Exact configuration delta

The new frozen profile is `configs/recovery_v2a1_frozen.yaml`. Relative to
`configs/recovery_v2_frozen.yaml`, only these four semantic paths change:

| Path | Original V2 | V2-A1 |
|---|---|---|
| `study.protocol_id` | `v2-recovery-activation-probe-20260929` | `v2-recovery-activation-probe-20261002-a1` |
| `dataset.answer_validation_policy` | `all-source-final-answer-v1` (legacy default, omitted from serialization) | `selected-cohort-final-answer-v1` |
| `artifacts.root` | `artifacts/v2` | `artifacts/v2a1` |
| `artifacts.public_summary_path` | `reports/v2_summary.json` | `reports/v2a1_summary.json` |

The original YAML is unchanged. Its semantic configuration SHA-256 remains
`d4903cbaaab23d2fc195b9ecce30f4c5816c3e3027e8cc2e8a09f8eed367b88f`.
The separately serialized A1 profile has semantic SHA-256
`bbab807d3ead00739dcab7482eecb6df763e72fa0308c19b12f0a6e06f30a6b3`.
The parser accepts only the exact original profile or this exact A1 profile;
partial namespace/policy combinations and scientific overrides are rejected.

All other scientific and runtime settings remain identical: the pinned
DeepMath source revision, Qwen3-1.7B model and tokenizer revisions, BF16,
runtime versions, question identities, duplicate policy, selection and split
seeds, 80/80 difficulty quotas, 96/32/32 split, prompt, six rollouts per
problem, 4,096-token cap, fixed checkpoints, symmetric greedy readout,
evaluator, activation geometry, A/B/C features, probe family, weighting,
regularization and layer selection, held-out estimand, bootstrap, and event
floors. A1 does not authorize a prompt-regime extension or a new method.

## 3. Validation order and no-replacement boundary

The A1 data contract is:

1. Resolve the exact pinned source revision, preserve physical-file hashes
   and source order, and retain original source indices. Do not clean,
   reorder, delete, or rewrite source rows.
2. Validate **question, difficulty, and topic globally**, exactly as in V2.
   A missing, mistyped, empty, or non-finite structural field remains a
   data-contract stop, even outside the eventual cohort.
3. Construct identities and perform the original answer-independent hash
   ranking and duplicate-cluster control at difficulties 7.0 and 8.0. Fix
   the complete ordered set of 160 selected identities before checking any
   selected reference-answer value. Do not filter or rank candidates by
   answer existence, answer content, topic, or later outcomes.
4. Validate the final-answer field for **every selected identity**. Any
   selected missing, non-string, empty, or whitespace-only answer stops A1.
   Do not choose the next-ranked candidate, replace the row, or relabel it.
5. Only a fully valid selected cohort may become the official immutable A1
   master manifest and grouped split. The same 48/16/16 per-difficulty
   allocation, leakage controls, and test-unblinding restriction apply.

Nonselected invalid final answers are preserved in the source and diagnostic
ledger; they are neither silently repaired nor used to choose the cohort.
The original profile continues to apply global final-answer validation.
The amendment is a named validation-scope exception, not a general
"skip invalid rows" policy.

## 4. Identity clarification, not a selection change

The implementation's existing `source-index-question-sha256-v1` identity
uses the original source index and SHA-256 of the **raw, exact question
string**. The problem ID is `deepmath:<source_index>:<first 16 hash hex
characters>`; the complete question hash is retained in the manifest.
Whitespace is not normalized before computing this identity.

Only duplicate clustering uses the SHA-256 of the question after splitting
on whitespace and joining with a single space. It does not perform semantic
deduplication or Unicode normalization. Selection ranks the original
seed/difficulty/problem-ID/source-index tuple, with source index as the tie
breaker, and preserves the existing shared duplicate-cluster exclusion
across the two strata.

The original protocol's normalized-question identity wording was inaccurate
and is corrected in its dated erratum. This is a clarification of existing
code, not a changed identity policy, selector, cohort, or scientific setting.

## 5. Evidence separation and execution gates

Original V2 failure receipts, hardware qualifications, budget decisions, and
audit outputs remain immutable in their original namespaces. A1 writes its
own configuration-bound artifacts under `artifacts/v2a1` and an
aggregate-safe summary at `reports/v2a1_summary.json`; it must not relabel or
adopt original scientific receipts. P0 remains immutable historical
feasibility evidence and is not a data source for A1.

Before a model-backed A1 unit, review and commit the amended implementation
and capture the clean source commit/tree identity, exact configuration hash,
pinned assets and runtime, and immutable manifest/split identities. Use the
existing intent-first, append-only ledger and write-once receipts. Failed,
interrupted-unknown, malformed, capped, non-evaluable, and OOM units remain
visible and are not automatically retried or replaced.

The original aggregate-only pre-fit checks, class/event floors, train-only
preprocessing, immutable validation-selection record before test generation,
and once-only held-out analysis remain mandatory. A scientific or budget
block is a valid terminal outcome; it does not authorize changing the cohort,
precision, checkpoints, readout, evaluator, fitting method, or test boundary.

As of this amendment, there are **no V2-A1 scientific results**. Any future
claim must follow the base protocol's bounded positive, null, confounded,
underpowered/non-evaluable, or hardware-block interpretation. Public outputs
may include only reviewed aggregates and safe provenance/configuration
identities, never raw rows, answers, prompts, traces, tokens, activations,
private artifacts, secrets, or machine-specific paths.
