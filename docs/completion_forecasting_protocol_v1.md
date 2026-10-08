# Completion forecasting from frozen reasoning states — protocol v1

Protocol ID: `completion-forecasting-v1-20261008`.

Date: 2026-10-08. **Status: CPU-only protocol reviewed; frozen together with the implementation in an immutable protocol/source lock before validation risk-set inspection.** Execution is limited to the user's autonomous research goal and the support gates below. This document does not authorize GPU work, test collection, a restart, or changes to V2-A1.

## 1. Question and evidence boundary

At the original fixed 512-token checkpoint, does a saved Qwen3-1.7B BF16 residual-stream state improve prediction of whether that same stochastic trajectory will emit its first close-thinking marker within its existing 4,096-token budget, beyond structural features, saved current-readout confidence, and an observable token-prefix baseline?

This is a newly selected, retrospective **bounded reasoning-completion forecasting** question. It is not correctness, reasoning recovery, solution quality, latent belief, early-stopping utility, or a causal effect of hidden activations. Emitting a close-thinking marker does not establish that a problem was solved. Reaching the cap does not establish that the trajectory would never close with more tokens.

The endpoint was chosen after observing TRAIN measurement-failure aggregates. This plan can lock the subsequent analysis, but cannot turn the pivot into a preregistered original V2 result. The original recovery endpoint remains non-evaluable, its operational failure remains unchanged, and all original artifacts and conclusions remain immutable.

The verified TRAIN diagnosis reports 151 natural-close and 425 cap-prefix trajectories, 2,848 available checkpoint activation inputs, and 32 unavailable coordinates after closure. At coordinate 512, this implies at least 119 positive and 425 negative potential rows, from at least 20 and 71 problems respectively because each problem has six rollouts. These are feasibility bounds before new enrollment or tensor-value checks, not fitted findings. Validation completion support has not been established.

## 2. Frozen inputs and scope

Reuse only the V2-A1 cohort and recovered artifacts: original Qwen3-1.7B/model-tokenizer revision, BF16 backend, six stochastic rollouts per problem, 4,096-token base cap, saved prompt/generated token IDs, original readout measurements, and saved post-block activation matrices. Do not regenerate trajectories, readouts, activations, confidence values, reference answers, or split assignments.

- Original 96 TRAIN problems: development, preprocessing, grouped cross-validation and final training.
- Original 32 VALIDATION problems: one locked, held-out exploratory evaluation; no tuning or train-plus-validation refit.
- Original 32 TEST identities: remain unused, ungenerated and uninspected. Do not rename validation as test.

Safe identity anchors are original source commit `926d076f3baf7b5d31328f8865bd70cefe8396c1`, original configuration hash `bbab807d3ead00739dcab7482eecb6df763e72fa0308c19b12f0a6e06f30a6b3`, manifest hash `8eb97c603f9731a8f482cfc855ba048d03613de420c846eba5a11f28fa80dc22`, and split hash `bad1088f2857f421326c2ce4d7ff94d0f913fef2df4ddc3d8ab5b1edd2052685`.

New code, configuration, derived labels, features, fits, selections and aggregate reports use the new protocol ID. Original logical identities and full payload hashes are read-only provenance joins, never rewritten labels. Public reports contain reviewed aggregates and safe hashes, not benchmark identities, questions, answers, token sequences, tensors, or private machine paths.

## 3. One primary landmark and target

There is exactly one primary checkpoint: the first **512 saved generated token IDs** after the original chat-template prompt. Do not select another coordinate after viewing performance.

A row is eligible only if the original trajectory has an available coordinate-512 reasoning prefix. That prefix contains no close-thinking marker, including when closure starts immediately after its final token. This is an at-risk landmark population, not all original problem-rollout slots.

Derive labels independently from exact saved base token IDs and the frozen boundary contract:

- `CLOSE_WITHIN_4096` (`y=1`): the first valid close-thinking marker starts at generated index 512 or later and completes inside the existing 4,096-token budget. Its original boundary must be `CLOSE_MARKER_FOUND`.
- `NO_CLOSE_WITHIN_4096` (`y=0`): no valid close marker occurs and the original boundary is a genuine `CAP_PREFIX` with exactly 4,096 saved generated IDs and `MAX_NEW_TOKENS` termination.
- Already-closed/unavailable coordinates, early-no-close termination, failed/interrupted requests and invalid boundaries are reported exclusions, not negatives. No replacements, continuation generations or synthetic labels.

Do not require a boxed answer, EOS readout, evaluator success or current wrongness for this endpoint. Do not read reference-answer content or join original correctness/evaluation fields. Record the original availability denominator, exclusions, resulting class counts, distinct problems and difficulty breakdowns.

## 4. Identical paired comparison cohort and integrity

Verify each consumed payload's full SHA256 and size against the recovered inventory before parsing. Verify prompt/prefix hashes and the coordinate-512 activation input as exactly `saved prompt IDs + saved first-512 prefix IDs`, without a readout cue or future tokens. Verify serialized tensor hash, matrix-content hash, CPU storage, BF16 dtype, `[28, 2048]` shape and finite values before using a matrix. A hash inconsistency stops the audit; do not guess a replacement file.

Use the same activation-valid enrolled rows for A, B, Blex, C and Clex. Preserve unavailable/missing/invalid activation attrition, never impute hidden vectors, and reapply support gates after attrition. Numeric readout-feature missingness does not exclude a row.

Separate feature construction from the future-label join. A feature builder must not accept terminal readouts, eventual marker coordinates, terminal reasoning length, later checkpoints, gold answers, evaluator fields, outcome labels, seed/index identifiers or problem identity as predictors. Problem identity is only a grouping, fold and weighting key.

The original runner recorded terminal readouts before checkpoint measurements. The valid leakage claim is **unchanged prefix-only inputs**, not chronological recording before the outcome. Saved current-readout confidence reflects a counterfactual close-cue-plus-64-token measurement, not an observed natural continuation.

## 5. Features and models

All models are unweighted-class L2 logistic regression, L-BFGS, `max_iter=10000`, `tol=1e-6`, deterministic `random_state=0`, with sample weights specified below. No nonlinear probe, training resampling, class balancing, post-hoc calibration or new language-model call.

| Model | Predictor representation |
|---|---|
| A | Original checkpoint position (512), position/4,096, problem-token count, difficulty and topic. Constant coordinates are removed by train-only variance filtering. |
| B | A plus the seven original current-readout scalar features below. |
| Blex | B plus the fixed trailing-prefix hashed token-count vector below. |
| C | B plus one selected layer's saved 2,048-dimensional residual vector. |
| Clex | Blex plus that same layer's saved residual vector. |

The seven B features are original `mean_answer_token_logprob`, `min_answer_token_logprob`, `answer_token_count`, `first_readout_token_entropy_nats`, `first_readout_token_probability_margin`, `previous_readout_agreement`, and `previous_readout_available`. Copy the scalar definitions from the original instrument; no evaluator-derived correction. Previous agreement uses only the immediately previous available fixed checkpoint at or before 256 and narrow whitespace normalization of extracted readout answers. Missing comparisons encode agreement as zero plus availability zero. No later checkpoint supplies an agreement feature.

**Blex:** take exactly the last 128 IDs of the saved first-512-token reasoning prefix, excluding prompt and readout-cue IDs. For every nonnegative token ID `t`, form UTF-8 text `completion-forecasting-v1|trailing-token-hash|` followed by its canonical base-10 integer string. Its bucket is `int(SHA256(text).hexdigest()[:16], 16) % 512`. Count occurrences in each of 512 buckets, with no sign hashing, IDF, vocabulary fitting, label-dependent selection, or future text. Raw integer counts are subsequently train-standardized. Hash collisions and loss of word order limit this control; it is not an exhaustive baseline for observable semantics.

Fit medians, missingness indicators, numeric means/scales, topic one-hot vocabulary and zero-variance filtering on the training partition only. Medians use observed training values; means and variances use the equal-problem weights. Remove columns with training variance at most `1e-12`. Unknown held-out topics map to an all-zero topic vector. Missing or nonfinite optional original confidence scalars map to missing; record nonfinite conversion counts explicitly. For a numeric column missing in every training row, predeclare zero imputation and its missingness indicator, then apply train-only variance filtering. This differs from the original V2 preprocessor's all-missing error behavior and belongs only to the new namespace. Never discard rows to obtain observed boxed-answer features.

A train-only, equal-problem prevalence predictor is also reported as a trivial reference. No held-out prevalence enters its fit.

## 6. TRAIN-only selection and immutable lock

Assign every original TRAIN problem to one of five folds before reading new labels. Within each frozen difficulty stratum, rank problems by SHA256 of UTF-8 `completion-forecasting-v1|fold|` plus problem ID; break hash ties by problem ID. Assign rank `i` to fold `i % 5`. All six rollouts and all duplicate-cluster members stay together. Keep these assignments even if class support is uneven.

Use the fixed inverse-regularization grid **`[0.001, 0.01, 0.1, 1.0]`**. Refit preprocessing for every training fold. For each candidate, pool its held-out-fold predictions and calculate equal-total-problem log loss; do not average fold losses equally when fold sizes differ. Every CV training partition must contain both classes. A one-class held-out fold does not invalidate log loss; pooled metrics require both classes.

Tune A, B and Blex separately. For Clex, tune every original layer 0–27 across the grid and select the smallest pooled TRAIN-CV log loss; ties choose smaller C, then lower layer index. This single Clex-selected layer is also used for secondary C. Tune secondary C's C value using TRAIN CV at that layer only. Do not pick a different secondary layer or use validation to resolve ties.

Mandatory fits must converge and produce finite probabilities; failures are recorded and block selection, without changing solver, grid, folds or cohort. Do not select a surviving subset of failed layer candidates or expand a search after seeing performance.

The immutable selection record contains protocol/configuration/source and library identities, consumed-input/cohort hashes, fixed folds, all TRAIN-CV candidate losses, attrition, chosen regularization values, the one primary layer, feature schemas, preprocessing parameters and final TRAIN-only model hashes. Complete and verify that record before any held-out activation performance inspection. TRAIN-CV performance selected from this search is development evidence, not unbiased final evaluation.

## 7. Support gates and one held-out evaluation

Freeze this protocol and the audited analysis source before deriving VALIDATION risk-set labels. After the lock, inspect only aggregate support/attrition before fitting; no validation activation performance may influence selection.

| Gate | Predeclared requirement on the identical activation-valid cohort |
|---|---|
| TRAIN feasibility | At least 50 rows per class from at least 15 independent problems per class. |
| CV feasibility | Every CV training partition contains both classes; all mandatory fits meet the fixed convergence/integrity contract. |
| Held-out feasibility | At least 20 VALIDATION rows per class from at least eight independent problems per class. |
| Within-problem interpretation | At least eight held-out problems each containing a positive and a negative rollout at coordinate 512. |

A class-support failure ends predictive progression in this namespace; report descriptive support and uncertainty limitations without switching coordinates, labels, cohorts or folds. A matched-group failure does not suppress the paired pooled result, but prohibits a within-problem interpretation: label any pooled gain as unresolved problem-level confounding. The original V2 multi-coordinate matched-group gate is not silently reused for this one-coordinate endpoint.

Once the immutable selection and gates pass, fit selected models on all eligible TRAIN rows only, then perform **one** held-out exploratory evaluation on original VALIDATION. Do not tune or refit after this evaluation. No original TEST collection or analysis is part of this protocol.

## 8. Estimand, weighting and uncertainty

The primary estimand is the equal-problem paired reduction in held-out log loss:

`Delta_LL = log_loss(Blex) - log_loss(Clex)`.

Positive values favor the activation addition. Both predictions must refer to the same rows. Secondary comparisons are B versus C at the already selected layer, B versus Blex, and descriptive A/B/Blex reference metrics. Report log loss, positive-class PR-AUC (weighted average precision), Brier score and class prevalence. Do not select an endpoint from whichever metric is favorable.

Within any fit or metric cohort, a problem with `n_g` rows receives row weight `(N/G) / n_g`, so each problem contributes equal total weight and mean weight is one. Do not treat six rollouts as six independent problems. Preprocessing remains train-partition-only; no terminal length normalizes weights or coordinates.

Use 2,000 deterministic problem-cluster bootstrap resamples, seed `20261008`, retaining all eligible rollouts for each sampled problem occurrence. Repeated sampled problems remain repeated draws; do not deduplicate them. Recompute equal-problem metric weights for occurrences and report percentile 95% intervals, valid-resample count and invalid fraction. Apply these intervals to paired log-loss improvements and matched concordance; PR-AUC, ROC-AUC and Brier scores are descriptive point estimates, not additional inferential endpoints. If fewer than 95% of resamples contain both classes, report intervals as non-evaluable under this protocol. These intervals condition on the selected trained model and do not quantify model-selection or training-set uncertainty.

For matched problems, calculate each model's concordance over every positive–negative rollout pair, assigning ties 0.5; average pair concordance within problem and then equally across matched problems. Report Blex, Clex and their paired difference with matched-problem bootstrap uncertainty. A null or unsupported matched contrast limits any pooled association.

## 9. Completion, reporting and resource boundary

Produce a reproducible CPU-only command, frozen new configuration, integrity/support receipts, TRAIN selection, final model provenance, one aggregate held-out report, limitations, and an interview brief. Unit/synthetic tests must cover prefix-only features, strict target boundaries, grouped folds, weighting, train-only preprocessing, unchanged original artifacts, bootstrap duplicate draws and matched scoring.

Allowed positive wording is limited to incremental predictive association for **bounded natural thinking closure under the recorded model, cohort, prompting, sampling, checkpoint and token budget**, with held-out exploratory and confounding qualifications. A null says the tested linear activation addition did not improve the stated comparison; it is not evidence that no completion representation exists. Do not describe this as a recovery-probe success or a completed original held-out test experiment.

Additional GPU hours and model-generated tokens: **zero**. Use local CPU resources only; record actual runtime and versions. Do not create/resume paid resources, alter storage or budgets, collect new readouts/test data, inspect the original test split, or modify frozen artifacts. Any later GPU or prospective original-test study requires a separate explicit execution decision.
