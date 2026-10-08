# From non-evaluable recovery to bounded completion forecasting

Date: 2026-10-08. Completed scientific artifact v1. Original V2-A1 is unchanged.

## Decision and result

The original answer-recovery question is **non-evaluable**: collection finished,
but the symmetric greedy-64 answer instrument produced no eligible wrong→correct
or wrong→wrong rows. It did not yield a negative activation finding. A fixed
TRAIN-only CPU reference gate then failed: only 4/12 references produced nonempty
original-parser output. Eight produced none; even successful extraction did not
establish whole-expression semantic validity. The proposed GPU readout comparison
was not run.

Instead, a separately frozen **zero-GPU** study reused the existing activations
to ask whether a reasoning state predicts natural thinking completion within
the original token budget. This exploratory comparison completed successfully.

At checkpoint 512, adding the TRAIN-selected layer-15 residual vector to the
confidence-plus-token-prefix baseline reduced held-out log loss from **0.6687 to
0.6134 nats**. The equal-problem paired improvement was **0.0553 nats**, with a
2,000-resample problem-cluster 95% interval of **[0.0212, 0.0888]**. The held-out
partition comprised 192 rollouts from the original 32 VALIDATION problems.
No original TEST data or new language-model inference was used.

Supported conclusion: the tested linear residual-state addition improved
**bounded natural-thinking-completion prediction relative to these specified
observable baselines** on a small exploratory held-out cohort. This does not
establish answer correctness, recovery, latent belief, safe early stopping or
inference savings.

## What was preserved and reused

The original BF16 Qwen3-1.7B campaign preserved 768 trajectories, 768 terminal
readouts, 3,840 checkpoint/activation receipt slots and 3,793 all-layer matrices.
All 9,122 collection operations completed; the selector subsequently failed
because primary recovery rows were absent. No original probe or test collection
followed. The model/tokenizer revision, dataset revision, 160-problem manifest,
96/32/32 grouped split, fixed coordinates and original labels remain unchanged.

The new study reused all 768 coordinate-512 matrices plus saved exact prefix IDs,
structural metadata and original observable readout scores. It did not regenerate
or reinterpret the original answer labels. Other checkpoint matrices and every
unavailable/failure receipt remain preserved but are not enrolled in this
single-landmark analysis. See the [original measurement diagnosis](v2a1_readout_repair_decision_v1_20261008.md)
and [aggregate measurement record](../reports/v2a1_measurement_diagnosis_v1.json).

## New question, labels and population

For an original trajectory still thinking after its first 512 generated tokens,
predict whether its first close-thinking marker appears within its saved
4,096-token maximum. Positives require a genuine natural close marker; negatives
require a genuine full-cap trajectory without that marker. Already-closed,
early-terminated, failed or inconsistent trajectories are not negative labels.
At this coordinate all original train/validation trajectories were eligible.

| Partition | Problems | Rows | Close / cap rows | Close / cap problems | Mixed problems |
|---|---:|---:|---:|---:|---:|
| TRAIN | 96 | 576 | 151 / 425 | 52 / 90 | 46 |
| VALIDATION held out | 32 | 192 | 67 / 125 | 20 / 29 | 17 |

Every consumed original payload was checked against its inventory by full size
and SHA256 before parsing. Each activation input was exactly prompt plus saved
first-512 IDs, without a close cue or future token. Every enrolled tensor passed
serialized/content SHA, CPU device, BF16 `[28,2048]`, and finiteness checks; there
was no activation attrition. BF16-to-float32 analysis conversion was lossless.

## Baselines, selection and leakage controls

- A: checkpoint geometry, problem token count, difficulty and topic.
- B: A plus the seven original current-readout confidence/agreement scalars.
- Blex: B plus fixed hashed counts of the final 128 prefix tokens in 512 buckets.
- C and Clex: B and Blex respectively plus the same selected layer vector.
- Prevalence: a constant estimated from TRAIN alone.

A, B, Blex, C and Clex use L2 linear logistic regression with identical rows,
no class balancing and equal total problem weights. Missing confidence values retain explicit
missingness; no failed answer is made correct or incorrect. Imputation, scaling,
topic vocabulary and variance filtering were fitted only on the training
partition of each fold. No terminal length, later token, gold answer, evaluator
outcome, problem ID, seed or rollout index entered the feature matrix.

Five difficulty-stratified, fixed problem-grouped TRAIN folds evaluated four
regularization values and 28 layers. All 128 candidate configurations converged
(640 CV fits); selected models were fitted on TRAIN only and hash-locked before
held-out evaluation. Clex selected zero-based layer 15 (transformer block 16).
Secondary C used that same layer. Every selected regularization value was 0.001.
Held-out evaluation restored frozen weights, without fitting or tuning.

The original runner physically recorded terminal readouts before checkpoint
reconstruction. Therefore this study establishes **prefix-information-only
provenance**, not chronological pre-outcome feature capture. Activations are
deterministic transformations of existing input tokens/model parameters; the
claim is incremental predictive utility of a representation, not information
unavailable in the full tokens.

## Held-out results

Metrics average problems equally; PR-AUC is weighted average precision.

| Model | Log loss, nats ↓ | Brier ↓ | PR-AUC ↑ | ROC-AUC ↑ |
|---|---:|---:|---:|---:|
| TRAIN prevalence | 0.6651 | 0.2347 | 0.3490 | 0.5000 |
| A | 0.6606 | 0.2329 | 0.4591 | 0.5906 |
| B | 0.6609 | 0.2331 | 0.4498 | 0.6054 |
| Blex | 0.6687 | 0.2364 | 0.3826 | 0.5667 |
| C | 0.5940 | 0.2057 | 0.6398 | 0.7379 |
| Clex | 0.6134 | 0.2141 | 0.5415 | 0.7053 |

| Paired comparison | Log-loss reduction, nats | Problem-bootstrap 95% interval |
|---|---:|---:|
| Primary: Blex − Clex | 0.0553 | [0.0212, 0.0888] |
| Secondary: B − C | 0.0669 | [0.0249, 0.1075] |
| Lexical control: B − Blex | −0.0078 | [−0.0364, 0.0202] |

All 2,000 primary bootstrap draws retained both classes. Resamples retained
duplicate problem occurrences; intervals condition on the trained/selected
models and omit model-selection/training-set uncertainty. Other metrics are
descriptive point estimates, not additional inferential endpoints.
Secondary and matched intervals are not multiplicity-adjusted and do not
independently confirm the primary exploratory association.

Among the 17 held-out problems with both outcomes, Blex concordance was 0.4047
and Clex concordance 0.5358. Their paired difference was 0.1310, with matched
problem-bootstrap interval [0.0044, 0.2683]. This supplies a limited within-problem
control, but the interval is wide and its lower edge is close to zero. Clex's
absolute within-problem concordance is modest; do not describe it as strong
rollout-level ranking accuracy.

The token-count augmentation itself did not improve B. It is a limited lexical
control, not evidence that activations beat an exhaustive semantic/token-based
predictor or a stronger observable baseline. The confidence-only comparison also
improved, using the same TRAIN-selected layer and without a new held-out search.

## Interpretation, limitations and stopping decision

This is a retrospective pivot chosen after TRAIN measurement-failure aggregates,
then locked before new-target validation enrollment and performance inspection.
Original validation served once as an exploratory holdout. It is **not** the
original untouched test, a preregistered recovery result or independent prospective
confirmation. Only 32 held-out and 17 matched problems limit precision and
generalization. Difficulty/cohort, prompting, sampling, checkpoint, model size
and fixed token horizon all bound the claim. Closing thinking is not solving.

The completed artifact is useful for an interview discussion of reproducible
representation probing, problem-grouped evaluation, leakage control and honest
measurement failure. Stop this namespace here: do not tune after the held-out
result, search another coordinate, relabel cap failures, or launch a paid retry.
Original correctness recovery remains untested. A future prospective study would
require a valid separately audited answer instrument and new explicit execution
authority; it is not necessary to report this completed exploratory result.

## Reproducibility and interview wording

The [frozen protocol](completion_forecasting_protocol_v1.md),
[configuration](../configs/completion_forecasting_v1.json),
[CPU reproduction guide](completion_forecasting_reproduction_v1.md), and
[machine-readable results](../reports/completion_forecasting_v1.json) define the
complete analysis. Private receipts record exact source/input/model/prediction
hashes. An independent saved-prediction calculation reproduced primary,
secondary and matched effects/intervals within `1e-12`, and verified the frozen
source/model chain and pre-evaluation lock. The portable original-measurement
audit independently reproduced zero primary recovery rows from 4,613
authenticated inputs. All 35 synthetic tests passed. Original tracked research
files and frozen artifacts were not modified.

The [final interview brief](interview_brief_completion_v1.md) provides a short
explanation, resume bullets and answers to common methodological questions.

Defensible resume bullets:

- Built an auditable BF16 reasoning-state dataset for Qwen3-1.7B, preserving 768
  stochastic trajectories and 3,793 all-layer checkpoint matrices with exact
  token-prefix and artifact-hash provenance.
- Executed a zero-GPU, problem-grouped activation-probe study after diagnosing
  a failed recovery instrument; reduced exploratory held-out thinking-completion
  log loss by 0.055 nats versus confidence-plus-prefix features (32 problems;
  problem-bootstrap 95% interval 0.021–0.089), without claiming answer recovery.

Do not say “proved reasoning recovery,” “completed the original test experiment,”
“reduced inference cost,” “decoded latent correctness,” or “activations contain
information absent from tokens.” The honest result is narrower and defensible.
