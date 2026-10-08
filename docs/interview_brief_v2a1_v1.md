# V2-A1 interview brief — v1

Date: 2026-10-08. **Verified collection and measurement diagnosis; predictive comparison not yet tested.** This is separate from the historical interview brief and does not revise its original study records.

## Thirty-second explanation

I built an auditable study asking whether Qwen3-1.7B checkpoint activations add information about future answer recovery beyond observable confidence. I completed 768 BF16 stochastic trajectories and preserved 3,793 all-layer checkpoint matrices with exact token-prefix and artifact-hash provenance. The answer instrument failed its scientific eligibility gate: short readouts mostly hit their cap, and reference parsing blocked many extracted answers. I preserved the failure instead of treating missing answers as wrong. The A/B/C prediction question remains untested. I am now developing a separately named, zero-GPU forecast of natural thinking completion using the collected states; that is a different target, not a disguised recovery result.

## What is actually complete?

- A frozen, problem-grouped 96/32/32 cohort, with six rollouts per selected problem; TRAIN and VALIDATION collection completed on 128 problems. The original 32-problem test split remains uncollected.
- 768 trajectory and terminal-readout receipts, 3,840 fixed-checkpoint receipt slots, and 3,793 available BF16 `[28, 2048]` matrices, captured before the answer-readout cue. The 47 unavailable coordinates are preserved.
- An intent-first ledger with 9,122 completed collection operations, zero failed/interrupted collection operations and no pending intents; an exact archive/inventory and fresh extraction verification.
- A read-only eligibility audit establishing zero W_TO_C and W_TO_W rows in each collected split. The selector therefore stopped before model fitting or test collection.
- A training-only diagnosis covering token/prefix/EOS/box semantics and a separately frozen CPU reference gate. Only 4/12 references returned nonempty parser outputs, below the required 10/12 and five-per-difficulty floors. No replacement or GPU repair followed.

These are measurement and research-engineering results, not activation-prediction performance.

## Why did the experiment not yield labels?

In TRAIN, 574/576 terminal readouts and 2,774/2,848 available checkpoint readouts hit the 64-token cap. Category-covering examples included unfinished derivation, truncated boxes, and boxes followed by more explanation without EOS. Among 85 extracted checkpoint answers, 63 had recorded reference-parse failures. That latter fraction is conditioned on extraction and is not a whole-corpus corruption estimate.

The independent audit did not demonstrate a chat-template, exact-prefix, EOS-classification or boxed-extraction bug. It established that the implemented instrument was not producing the endpoint required by the frozen protocol. A longer cap might introduce more reasoning; an answer prefill changes the observation operator. Neither can be called an interchangeable readout without a new versioned measurement study.

## What does this say about hidden activations?

Nothing yet about incremental recovery prediction. Zero eligible labels makes the estimand non-evaluable; it is not a measured zero effect or a negative result. No A/B/C model, validation-selected layer, held-out log loss, paired improvement, confidence interval, or matched-control result is available for the original recovery question.

## Why keep the data?

The saved token IDs define exact checkpoint inputs, and the matrices were captured from those inputs without the answer cue. They can be reused for a scientifically distinct, preregistered target without regenerating trajectories. Existing structural and readout-confidence features are immutable measurements; a new answer instrument would require new confidence features. Full vocabulary logits and ungenerated continuations were not archived.

The next chosen direction forecasts whether an at-risk original trajectory closes its thinking before the 4,096-token cap from checkpoint 512. It avoids mathematical-answer parsing and new GPU inference, but measures termination rather than correctness. Its target, censoring policy, baselines and held-out evaluation must be frozen before inspecting predictive performance. No such predictive result is claimed in this brief.

## Leakage and uncertainty questions to expect

**Why split by problem?** Rollouts and checkpoints from the same problem are correlated. Keeping them in one split prevents a row-level split from leaking problem identity, and problem-cluster uncertainty avoids treating repeated rows as independent evidence.

**Were features recorded before outcomes?** Not chronologically: the original runner issued terminal readouts before checkpoint reconstruction. The valid statement is that activation/current-feature inputs contain only checkpoint-available information, excluding future tokens, terminal length, terminal answers, gold answers and evaluator status. Input lineage must be checked independently of wall-clock order.

**Why simple probes?** The planned A/B/C comparison uses L2 logistic regression with training-only preprocessing and validation-only regularization/layer selection, making incremental activation value interpretable. This describes the implemented design, not a fitted result.

**How would you control problem confounding?** Use identical paired B/C cohorts, equal-total-problem weights, problem-grouped uncertainty, and same-problem/same-coordinate rollout controls. A pooled gain alone cannot establish trajectory-specific recovery information.

**What would you do differently?** Treat answer readout and reference parsing as scientific instruments requiring independent feasibility validation before expensive representation collection. Preserve exact IDs and typed failures so an instrument failure does not destroy otherwise valid data.

## Defensible resume bullets now

- Engineered an auditable BF16 reasoning-state collection pipeline for Qwen3-1.7B; preserved 768 stochastic trajectories and 3,793 all-layer checkpoint matrices with exact token-prefix, grouped-split and artifact-hash provenance.
- Diagnosed a non-evaluable recovery endpoint by separating generation completion, EOS/extraction eligibility and reference parsing; preserved failures without label substitution or post-outcome cohort tuning.

Do not write “activations predicted recovery,” “beat observable confidence,” “completed held-out A/B/C evaluation,” “proved no recovery representation,” or “enabled safe early stopping.” Update these bullets only after the separately versioned predictive evaluation produces reviewed results.

## Evidence to show an interviewer

Use [the draft research report](recovery_study_final_report_v1_draft.md), [the aggregate diagnosis](../reports/v2a1_measurement_diagnosis_v1.json), [the readout repair decision](v2a1_readout_repair_decision_v1_20261008.md), [the CPU diagnostic contract](reference_parsability_diagnostic_v1.md), and the frozen [protocol](v2_protocol.md). Show aggregate provenance and code, not benchmark rows, reference answers, private traces, token IDs or activation tensors.
