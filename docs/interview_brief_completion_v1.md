# Interview brief: an honest activation-study result

Date: 2026-10-08. Final v1; supersedes the interim interview brief for the new
completion endpoint, without changing the original V2-A1 conclusions.

## Thirty-second explanation

“I built a provenance-controlled BF16 reasoning-state dataset for Qwen3-1.7B:
768 stochastic trajectories and 3,793 all-layer checkpoint matrices. The
original answer-recovery instrument failed its eligibility gate, so I preserved
that non-evaluable outcome. I then froze a separate CPU-only question: can the
state at token 512 predict whether thinking naturally closes by token 4,096?
With problem-grouped training and a locked exploratory holdout, activations
improved log loss by 0.055 nats beyond confidence plus fixed prefix-token features,
with a problem-bootstrap 95% interval of 0.021–0.089 across 32 held-out problems.
That is a thinking-completion result, not correctness or recovery.”

## Resume bullets

- Built an auditable BF16 Qwen3-1.7B reasoning-state dataset with 768 stochastic
  trajectories and 3,793 all-layer checkpoint matrices; preserved exact token
  prefixes, model/runtime identities, split boundaries and artifact hashes.
- Completed a zero-GPU activation-probe study with problem-grouped evaluation;
  improved exploratory held-out thinking-completion log loss by 0.055 nats over
  confidence-plus-prefix features (32 problems; problem-bootstrap 95% interval
  0.021–0.089), while retaining the original non-evaluable recovery finding.

## Questions to be ready for

**Why did the original endpoint fail?** Collection completed, but almost all
64-token counterfactual answer readouts hit their cap; parsing/evaluation also
failed frequently. There were zero eligible wrong→correct or wrong→wrong rows.
A fixed 12-reference TRAIN CPU diagnostic yielded only four nonempty parser
outputs. Increasing generation length alone would not solve the reference gate.

**What exactly was predicted?** First natural close-thinking marker within the
original 4,096-token budget, given an available token-512 prefix. A full-cap
trajectory without the marker is negative for this bounded event—not an
incorrect mathematical answer. There were 151/576 TRAIN and 67/192 held-out
positive rollouts, with no tensor attrition at this coordinate.

**What prevents leakage?** Exact prefix-only activations; no terminal text,
future length, evaluator labels or gold answers in predictors. All preprocessing
and regularization/layer selection used five problem-grouped TRAIN folds.
Layer 15 and all final model weights were locked before held-out prediction.
Features were reconstructed after outcomes physically, so the defensible claim
is prefix-information-only provenance, not chronological pre-outcome recording.

**What are the baselines?** Structural features; those plus seven observable
readout confidence/agreement scalars; and a fixed hashed token-count augmentation.
The primary comparison adds one 2,048-dimensional residual vector to the last
baseline. Activations are deterministic representations of tokens and model
parameters, not information absent from the tokens. The lexical control is not
an exhaustive semantic predictor and did not improve the confidence baseline.

**How strong is the evidence?** A small exploratory validation-heldout result,
not the original confirmatory test. The paired interval resamples entire
problems and conditions on the selected trained models; it omits model-selection
uncertainty. Within-problem concordance improved, but absolute Clex concordance
was only 0.536 across 17 mixed-outcome problems, with a wide interval.
Secondary and matched intervals are not multiplicity-adjusted or independent
confirmation of the primary exploratory result.

**What remains untested?** Correct-answer recovery, safe early stopping, token
savings, causal mechanisms and prospective generalization. The original TEST
split remains untouched. No new language-model inference or GPU hours were used
for the pivot. Do not tune this namespace after its held-out result.

Full methods/results: [final report](recovery_activation_study_final_v1.md).
Reproduction: [CPU guide](completion_forecasting_reproduction_v1.md).
