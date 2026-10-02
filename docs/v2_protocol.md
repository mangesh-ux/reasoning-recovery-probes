# V2 Recovery Representation Study Protocol

**Protocol ID:** v2-recovery-activation-probe-20260929  
**Status:** frozen before any V2 model-backed scientific execution  
**Scope:** a new study in a separate V2 namespace; it is not a continuation, reinterpretation, or extension of the completed P0 campaign.

## 1. Purpose and claim boundary

The primary question is:

> Given that Qwen3-1.7B is wrong at an intermediate reasoning checkpoint, do the checkpoint's residual-stream hidden activations contain incremental information about whether the same stochastic trajectory's later reasoning prefix will produce a correct answer under the same deterministic answer readout, beyond structural/problem features and current observable readout-confidence features?

The target is a predictive association under the exact model, dataset, prompting, readout, evaluator, and split specified here. It is not:

- a claim that continued reasoning causally fixes an answer;
- a claim that a recovery probe supports early stopping;
- a claim that a learned direction generalizes to other models, datasets, prompts, or numerical backends;
- a claim about KV-cache tensors;
- a reuse of P0 examples or a retroactive conversion of P0 into confirmatory evidence.

P0 remains immutable historical feasibility evidence. Its MATH-500 selection, stochastic trajectories, forced readouts, receipts, and reports are never read as V2 inputs.

## 2. Rationale

### 2.1 Model and numerical representation

V2 uses Qwen/Qwen3-1.7B at model and tokenizer revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e in BF16 thinking mode.

This model is an open-weight reasoning model with an explicit thinking-mode template. Its small size permits repeated stochastic trajectories and internal-state extraction while retaining BF16 numerical representation. BF16 is the scientific measurement backend because fixed-prefix work in the separate confidence audit showed candidate-identity instability under aggressive quantization. That audit does not establish V2 results; it motivates avoiding an unnecessary numerical confound.

Qwen's published thinking-mode sampling values are temperature 0.6, top-p 0.95, top-k 20, and min-p 0. The V2 configuration pins those values, its chat-template mode, the exact tokenizer revision, and runtime-version expectations.

### 2.2 Dataset and capability frontier

V2 uses the train split of zwhe99/DeepMath-103K at revision 5cf055d1fe3d7a2eb19719ac020211469736ae44. The source schema is fixed as question, final_answer, difficulty, and topic.

The primary population contains exactly 160 unique normalized-question clusters:

| Difficulty stratum | Selected problems |
|---|---:|
| 7.0 | 80 |
| 8.0 | 80 |

DeepMath levels 7 and 8 are a pre-specified mid-high difficulty regime. The rationale is capability-frontier control: a population that is trivial produces mostly correct-to-correct trajectories, while a population that is universally impossible produces mostly wrong-to-wrong trajectories. The protocol does not claim that levels 7 and 8 are universally optimal. It freezes them before V2 outputs and will not search levels after seeing probe performance.

Topic metadata is retained for reporting and the structural baseline only. It is never an outcome-based selection signal.

## 3. Dataset selection, duplicate control, and split

### 3.1 Stable identity and deterministic selection

The source split row index and SHA-256 of its raw, exact question string define each V2 problem identity. Only the duplicate-cluster identity uses the normalized-question SHA-256. This wording is clarified by the dated erratum below; the implemented identity and selection policies are unchanged. The selector:

1. validates the frozen source schema;
2. accepts only rows whose difficulty value is exactly 7.0 or 8.0 under the source numeric representation;
3. creates a stable problem identity from source index and question hash;
4. ranks eligible rows independently inside each difficulty stratum by SHA-256 of selection seed, difficulty, stable problem identity, and source index;
5. retains the first 80 unique duplicate clusters in each stratum;
6. records row identity, source index, difficulty, topic hash, question hash, answer hash, and duplicate-cluster hash in one immutable master manifest.

No reference answer, model output, correctness label, confidence value, evaluator result, or hidden activation participates in selection.

If the frozen dataset cannot provide 80 valid unique clusters in either stratum, V2 stops with a data-contract failure. It does not broaden a difficulty stratum or substitute another dataset.

### 3.2 Problem-grouped split

The master manifest assigns every selected problem to exactly one split before any V2 model-backed scientific request:

| Split | Difficulty 7 | Difficulty 8 | Total |
|---|---:|---:|---:|
| Train | 48 | 48 | 96 |
| Validation | 16 | 16 | 32 |
| Test | 16 | 16 | 32 |

Assignment is a deterministic SHA-256 rank inside each difficulty stratum using split seed 20260930. All rollouts and all checkpoints for one problem, and every member of a normalized-question duplicate cluster, remain in one split.

The test set is not used for feature preprocessing, regularization selection, layer selection, or model selection. Test trajectories may be generated only after an immutable validation-selection record exists. Test metrics are computed once by the final test command.

## 4. Base reasoning trajectories

For each selected problem, V2 derives six independent stochastic seeds using SHA-256 of root seed 20260929, stable problem ID, and rollout index in 0 through 5. The derived nonzero integer seed is stored in every receipt.

The model receives the frozen thinking-mode chat template around this exact user prompt:

~~~text
Solve the following mathematics problem. Work through the reasoning carefully.
After the thinking section, give only a concise final answer in exactly one
\boxed{...} expression, with no explanation after the box.

Problem:
{problem}
~~~

The base request samples up to 4,096 new tokens with:

| Parameter | Value |
|---|---|
| do_sample | true |
| temperature | 0.6 |
| top_p | 0.95 |
| top_k | 20 |
| min_p | 0.0 |
| num_beams | 1 |
| batch size | 1 |

The runner detects the first exact token-ID subsequence for the frozen close-thinking marker, </think>. It saves all emitted generated token IDs, including the marker if present, without decoding and re-tokenizing them.

The terminal reasoning prefix is:

- the generated token prefix immediately before the first close-marker token ID if the marker occurs; or
- all 4,096 generated token IDs if the cap is reached with no close marker.

An emitted opening thinking marker counts as a generated reasoning token if it is in the saved generated sequence. Template tokens do not count. The close marker never belongs to a terminal or checkpoint reasoning prefix.

An EOS or other terminal event before a close marker and before the cap is a typed early-no-close boundary failure. It remains in receipts but creates no terminal reasoning prefix, no checkpoint activation, and no primary label. It is never silently converted to cap behavior.

## 5. Fixed geometric checkpoints

The primary checkpoints are exact raw generated-token coordinates:

~~~text
128, 256, 512, 1024, 2048
~~~

Coordinate p means the first p saved generated token IDs before the natural close marker. A checkpoint is available only when p is inside the terminal reasoning prefix. Any coordinate after the close marker, after an early terminal event, or beyond a cap-shortened trajectory is explicit unavailable evidence.

These fixed geometrically spaced coordinates are the primary design because they:

1. are deterministic and independent of the model's self-reflection wording;
2. measure the same computational coordinate across trajectories;
3. enable same-problem, same-checkpoint comparisons across rollouts;
4. avoid conditioning the primary design on words such as wait or actually;
5. provide higher early resolution while keeping long-horizon measurement bounded.

Semantic markers may be recorded only as exploratory observable metadata after the primary study is terminal. They are not checkpoint selectors.

## 6. Symmetric deterministic answer readout

V2 never uses a naturally generated post-thinking answer as its primary terminal outcome. It applies one identical deterministic operation to every available checkpoint prefix and to the terminal reasoning prefix:

~~~text
exact chat-template prompt token IDs
+ exact saved reasoning-prefix token IDs
+ exact frozen readout cue token IDs for </think>\n\n
-> greedy answer readout with max_new_tokens = 64
~~~

The base prompt already instructs the model to emit one concise boxed answer after thinking. The cue ends thinking; it does not request a new long chain of thought.

The same tokenizer, cue token IDs, EOS policy, greedy decoding, maximum length, extraction policy, and evaluator apply to terminal and checkpoint readouts. The readout is successful for primary-label purposes only when it reaches EOS, produces one extractable last balanced boxed answer, and obtains an evaluator result of CORRECT or INCORRECT. Capped, interrupted, malformed, missing, empty, and evaluator-error outcomes remain explicit as NON_EVALUABLE or ERROR; they are not relabeled as wrong.

The evaluator is math-verify version 0.9.0 under the recorded runtime contract. All V2 result records retain:

~~~text
CORRECT
INCORRECT
NON_EVALUABLE
ERROR
~~~

## 7. Recovery labels

For a checkpoint with an evaluable short answer readout and an evaluable terminal-prefix short answer readout:

| Checkpoint readout | Terminal-prefix readout | Label |
|---|---|---|
| INCORRECT | CORRECT | W_TO_C |
| INCORRECT | INCORRECT | W_TO_W |
| CORRECT | CORRECT | C_TO_C |
| CORRECT | INCORRECT | C_TO_W |

The only primary probe rows are currently-wrong checkpoints:

~~~text
W_TO_C is the positive label.
W_TO_W is the negative label.
~~~

These labels describe a later deterministic readout from the same stochastic trajectory's saved terminal reasoning prefix. They do not identify natural-answer recovery and do not identify a causal effect of continuing reasoning.

## 8. Observable readout features

The runtime computes compact scalar statistics immediately from greedy readout scores and does not save full vocabulary-by-token logits. Feature construction occurs before evaluator labels are joined.

For a completed readout, the candidate-token region is the token sequence strictly between the outer opening and closing brace tokens of the selected last balanced boxed answer. The parser uses balanced braces and explicit character-to-token reconstruction. It does not invent character-level probability values when braces share a tokenizer token.

The frozen observable features are:

| Feature | Exact definition |
|---|---|
| mean_answer_token_logprob | Arithmetic mean of selected-token log probabilities over the candidate-token region, excluding EOS |
| min_answer_token_logprob | Minimum selected-token log probability over that same region |
| answer_token_count | Number of tokens in that candidate-token region |
| first_readout_token_entropy_nats | Entropy of the first generated readout-step probability distribution after deterministic processors |
| first_readout_token_probability_margin | Largest minus second-largest probability at that first generated readout step |
| previous_readout_agreement | Whether the immediate previous available checkpoint readout in the same rollout has the same normalized extracted answer |
| previous_readout_available | One if that comparison is available, zero otherwise |

For model fitting, previous_readout_agreement is encoded as zero when unavailable and its availability indicator is included. Numeric missing values use a train-fold median imputation plus a train-fold missingness indicator. No terminal answer, terminal length, later checkpoint, gold answer, evaluator status, or future feature is included.

## 9. Hidden-state extraction

For every available fixed checkpoint, V2 performs one separate BF16 forward pass over:

~~~text
exact chat-template prompt token IDs + exact saved checkpoint-prefix token IDs
~~~

The readout cue is never added to this activation input. The runner uses inference mode, no cache, no attention outputs, and post-transformer-block forward hooks. It captures only the final sequence position from every transformer block, immediately copies each vector to CPU in BF16, and releases GPU references.

The frozen representation is:

| Property | Definition |
|---|---|
| Layer indices | Zero-based transformer-block outputs |
| Expected layer count | 28 |
| Expected width | 2,048 |
| State location | Post-block residual output before final RMS normalization |
| Embedding state | Excluded |
| Final-normalized state | Excluded |
| Stored matrix | [28, 2048] BF16, one immutable private artifact per available checkpoint |

The implementation must not request or retain a full layer-by-sequence activation collection. It records matrix shape, dtype, tensor hash, input hash, runtime provenance, elapsed time, and peak CUDA allocator telemetry. A failed activation extraction is retained as a typed failure. The paired B-versus-C comparison uses only the same primary-label rows with a successfully stored activation; it reports the resulting attrition and does not impute activations.

## 10. A/B/C predictive comparison

All three models are classifiers, not language models. They use only rows with primary W_TO_C or W_TO_W labels.

### A: structural/problem baseline

A uses:

1. checkpoint token position;
2. normalized checkpoint position, defined only as checkpoint position divided by 4,096;
3. raw problem-token count under the frozen tokenizer without special tokens;
4. DeepMath difficulty;
5. DeepMath topic.

The normalized coordinate never divides by terminal reasoning length because that length is future information.

### B: observable baseline

B adds the frozen current-readout scalar features in Section 8. It contains no hidden state and no information from the terminal readout.

### C: activation probe

For each transformer layer separately, C concatenates the B feature representation and the one checkpoint activation vector from that layer.

Every A, B, and C model is an unweighted-class, L2-regularized logistic regression using the L-BFGS solver and maximum 10,000 iterations. No MLP, nonlinear model, resampling method, or post-hoc calibrator is allowed as a primary model.

Training rows receive weights so that each problem contributes equal total weight, regardless of its number of available checkpoint rows. The primary point estimates use the corresponding equal-total-problem row weights; ordinary unweighted-row estimates are reported only as a sensitivity view.

Numerical standardization, median imputation, missingness indicators, topic one-hot encoding, and zero-variance handling are fit on training data only. Unknown test topics are ignored by the fitted one-hot encoder rather than causing schema expansion.

The regularization grid is:

~~~text
0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0
~~~

For A, B, and each C layer, the selected regularization value minimizes validation log loss. Ties choose the smaller C value. The one primary activation layer is selected by the smallest validation log loss among the per-layer tuned C models; a tie chooses the lower layer index.

After the selection record is immutable, the selected A, B, and primary C models are refit once on the union of training and validation rows. Every layer also has a validation-selected C model for the complete descriptive test-layer curve. Test data never selects a hyperparameter or layer.

## 11. Data-validity and event gates

Before fitting, analysis may inspect only aggregate label and attrition counts by frozen split. It may not inspect activation-based performance or test metrics.

The protocol declares the following feasibility floors before V2 outputs:

| Gate | Requirement |
|---|---|
| Train fit | At least 50 W_TO_C and 50 W_TO_W rows, each arising from at least 15 training problems |
| Validation selection | At least 20 rows of each class, each arising from at least 8 validation problems |
| Test primary metrics | At least 20 rows of each class, each arising from at least 8 test problems |
| Within-problem control | At least 15 matched problem-by-checkpoint groups from at least 8 test problems |

These floors are feasibility gates, not power guarantees. They prevent an elastic post-output judgment of what counts as enough independent evidence. If a gate fails, V2 records a scientific block or underpowered result. It does not alter the population, difficulty strata, checkpoint grid, prompt, BF16 backend, readout, feature definitions, labels, probe class, or split.

## 12. Evaluation and uncertainty

The primary held-out metrics are PR-AUC, log loss, and Brier score. AUROC and fixed-decile calibration diagnostics are secondary. W_TO_C is the positive class.

The primary inferential estimand is the paired held-out reduction in log loss from B to C at the validation-selected primary layer, evaluated on the identical activation-available test cohort.

Uncertainty uses 2,000 deterministic problem-cluster bootstrap resamples with seed 20261001. Resampling occurs at the problem level, retaining all of a selected problem's test rows. The report gives percentile confidence intervals, the number of valid resamples, and the invalid-resample fraction. An interval is non-evaluable if fewer than 95 percent of resamples contain both classes for the relevant metric.

The within-problem control considers each held-out group with the same problem and checkpoint coordinate containing at least one W_TO_C rollout and one W_TO_W rollout. For each group, pairwise concordance is the mean over all positive-negative score pairs, with a tie contributing 0.5. Groups are averaged within problem, then problems are averaged equally. The same problem-cluster bootstrap supplies uncertainty. The report gives matched-group count, unique-problem count, B and C concordance, and the B-to-C difference.

The complete test-layer curve reports B-to-C-layer deltas in PR-AUC, log loss, and Brier score for all 28 layers. It is descriptive and multiplicity-limited; no test-best layer is selected.

## 13. Hardware and execution sequence

The local RTX 3070 Ti Laptop GPU has 8 GiB memory and is BF16-capable. P0 demonstrated base and forced generation on this device, but it did not qualify score capture or all-layer activation extraction.

Before any V2 scientific request, a synthetic-only hardware qualification must:

1. validate the pinned model, tokenizer, template, cue, BF16 capability, and expected layer geometry;
2. test synthetic thinking generation, 64-token readout generation, and hook-based activation extraction;
3. test activation forwards at 512, 1,024, 2,048, and 4,096 synthetic input tokens;
4. record tokens per second, forward passes per second, peak allocated and reserved memory, activation matrix size, and projected maximum workload;
5. require at least 1 GiB allocator headroom and no OOM at every mandatory shape;
6. calculate conservative full-campaign time and storage estimates.

The local campaign proceeds only if the synthetic qualification projects at most 48 wall-clock GPU hours and all hardware gates pass. Otherwise V2 is a hardware block locally and the project provides a one-command runner for a 24 GiB CUDA GPU. The runner preserves BF16, cohort, checkpoints, sampling, readout, and artifact contracts; it does not quantize, offload, or weaken the protocol to fit the laptop.

The scientific execution order is:

1. commit the V2 implementation and freeze its config hash;
2. build immutable DeepMath manifest and split;
3. complete synthetic runtime qualification;
4. collect train and validation trajectories, readouts, and available activations;
5. apply the pre-fit event and integrity gates using aggregate counts only;
6. fit and freeze validation-selected preprocessing, regularization, and layer decision;
7. collect test trajectories and available activations once;
8. run one held-out analysis and write aggregate-safe reports.

Every model-backed stage requires a clean committed source tree and an exact source commit/tree identity in its V2 run identity. Each logical request uses an intent-first append-only ledger and write-once receipt. Failed, capped, malformed, non-evaluable, OOM, and interrupted-unknown units remain visible and are never automatically retried or replaced.

## 14. Deferred prompt-regime extension

The prompt-regime question is explicitly deferred until the primary V2 study has a terminal analysis and decision. It requires a separately frozen V2-E1 protocol and a deterministic manifest that does not overlap any of the 160 primary V2 problems.

If authorized then, it will compare neutral/default, deep/deliberate, and concise-but-accurate prompting. The neutral-trained primary probe will be evaluated without retraining under the other regimes. The extension first reports behavioral differences in reasoning length, terminal-readout accuracy, and W_TO_C/W_TO_W rates; only then does it describe cross-regime recovery transfer. It does not claim novelty merely because prompts are linearly separable, and it does not call KV-cache tensors residual-stream activations.

## 15. Allowed conclusions

| Terminal outcome | Allowed conclusion |
|---|---|
| Positive | Under this frozen V2 setting, the validation-selected checkpoint activation probe added held-out predictive information beyond the B features; scope remains model-, data-, readout-, and split-specific. |
| Null | Under this frozen V2 setting, the tested linear activation addition did not improve the stated held-out estimand; this does not prove that no recovery representation exists. |
| Confounded | A pooled gain without the matched within-problem control is reported as pooled-only and problem-difficulty confounded. |
| Underpowered or non-evaluable | Frozen event, split, or matched-group requirements were not met. No alternative population or protocol is searched. |
| Hardware block | The required BF16 workload could not be qualified on the tested hardware; this is not a scientific no-effect conclusion. |

## 16. Deliverables

The V2 record consists of:

- docs/v2_protocol.md;
- docs/v2_validity_review.md;
- docs/v2_results.md;
- docs/v2_hardware.md;
- docs/interview_brief.md;
- reports/v2_summary.json;
- private, ignored raw artifacts under artifacts/v2.

Public-safe documents and reports may contain only reviewed aggregate counts, safe provenance hashes, configuration identities, protocol decisions, and bounded conclusions. They must not include benchmark rows, prompts, completions, token IDs, hidden-state tensors, model weights, caches, secrets, or local machine paths.

## Dated erratum: 2026-10-02

Section 3.1 formerly described the problem-identity question hash as
normalized. The frozen implementation actually computes it from the raw,
exact question string and original source index under
`source-index-question-sha256-v1`. Whitespace normalization (split on
whitespace, join with one space) applies only to
`normalized-question-sha256-v1` duplicate clusters. The corrected wording
documents existing behavior; it changes neither code-level identities nor
cohort membership, ranking, seeds, or any scientific setting.

The original V2 profile and its global final-answer data-contract failure
remain preserved. The explicitly approved, separately namespaced
[V2-A1 amendment](v2a1_protocol_amendment.md) changes final-answer validation
scope only and does not retroactively amend the original scientific profile
or erase its failure.
