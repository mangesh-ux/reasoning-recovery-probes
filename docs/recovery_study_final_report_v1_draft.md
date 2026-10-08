# Recovery-representation study — final-report draft v1

Date: 2026-10-08. **DRAFT / INCOMPLETE.** This is a new reporting document, not an amendment of P0, V2, or V2-A1. It records verified completed work and explicitly identifies the remaining untested question. No proposed repair or predictive pivot is represented as an executed result.

## Executive finding

The V2-A1 train/validation collection completed successfully, preserving 768 stochastic Qwen3-1.7B trajectories and 3,793 all-layer checkpoint activation matrices. The symmetric 64-token answer-readout instrument supplied **zero eligible wrong-to-correct and wrong-to-wrong rows** in both training and validation. Model selection consequently terminated before fitting or test collection. The original operational outcome remains `OPERATIONAL_FAILURE`; the scientific endpoint is **non-evaluable**, not a null activation-prediction result.

The principal observed obstacles are readout truncation/continued explanation and reference-answer parsing failures. A completed predictive comparison has not yet been established. The separately frozen CPU reference-parsability gate **FAILED**: four of 12 TRAIN references returned nonempty parser output, versus the required ten; neither difficulty stratum met its five-of-six floor. Eight parser outputs were empty; this does not mean that the stored reference strings were empty. No GPU readout comparison followed.

The selected next research direction is a separately named **natural thinking-completion forecast at checkpoint 512**, using existing data without new GPU work. Its target is whether the original stochastic reasoning trajectory emits its close-thinking marker before the 4,096-token cap. This is reasoning-termination prognosis, not correctness or recovery. Its protocol and predictive results are still pending in this draft.

## 1. Research question and immutable design

The original question is whether a checkpoint residual-state vector adds predictive information about later same-trajectory answer recovery beyond structural and observable answer-confidence features. It concerns a deterministic, induced readout endpoint, not natural-answer recovery, a causal intervention, latent belief, or safe early stopping.

| Design component | Frozen V2-A1 contract |
|---|---|
| Model / representation | Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, BF16 |
| Dataset | DeepMath-103K, revision `5cf055d1fe3d7a2eb19719ac020211469736ae44` |
| Selected population | 160 answer-independently selected unique normalized-question clusters; 80 each at difficulty 7 and 8 |
| Grouped split | 96 training / 32 validation / 32 untouched test problems; all rollouts/checkpoints from a problem remain together |
| Reasoning | Six stochastic rollouts per problem; 4,096-token maximum; exact saved token IDs are prefix authority |
| Checkpoints | Generated-token positions 128, 256, 512, 1,024 and 2,048, before the first close-thinking marker |
| Answer instrument | Same close-thinking cue and greedy 64-token readout at checkpoint and terminal prefix; EOS, balanced box and evaluable answer required |
| Representation | One pre-cue final-position post-block residual matrix per available checkpoint; `[28, 2048]` BF16 |

The complete scientific definitions and the selected-reference validation amendment remain in [the frozen protocol](v2_protocol.md) and [the A1 amendment](v2a1_protocol_amendment.md). The historical [V2 results](v2_results.md) document an earlier local hardware block; they are not the V2-A1 cloud collection result and remain unchanged.

## 2. Completed collection and endpoint attrition

| Quantity | TRAIN | VALIDATION | Total |
|---|---:|---:|---:|
| Problems collected | 96 | 32 | 128 |
| Base trajectories / terminal readout receipts | 576 | 192 | 768 |
| Intended fixed-checkpoint receipt slots | 2,880 | 960 | 3,840 |
| Available checkpoint readouts / activation matrices | 2,848 | 945 | 3,793 |
| Unavailable coordinates after close-thinking boundary | 32 | 15 | 47 |
| Terminal readouts ending at 64-token cap | 574 | 190 | 764 |
| Checkpoint readouts ending at 64-token cap | 2,774 | 926 | 3,700 |
| Terminal EOS readouts | 2 | 2 | 4 |
| Checkpoint EOS readouts | 74 | 19 | 93 |
| Eligible W_TO_C rows / independent problems | 0 / 0 | 0 / 0 | 0 / 0 |
| Eligible W_TO_W rows / independent problems | 0 / 0 | 0 / 0 | 0 / 0 |

All 9,122 issued collection operations completed; the retained ledger has zero failed/interrupted collection operations and zero pending intents. Collection exited successfully on 2026-10-07 at 06:27:37 UTC. The subsequent selector failed at 06:31:44 UTC because at least one primary row was required. No validation-selection artifact, test authorization, or test request/completion followed.

Training-only diagnosis found 85 extracted checkpoint boxes, of which 63 had recorded reference-parse failures. This is a count conditioned on extracted answers, not an estimate of corpus-wide reference validity. Of 2,848 available training checkpoint readouts, 2,758 lacked a complete box and five were malformed. The sole eligible EOS/EOS transition was C_TO_C, which is not a primary recovery row. Missing, capped, unavailable and non-evaluable cases were not relabeled as incorrect.

Fifteen category-covering training texts were examined descriptively. They support truncation, continued derivation, boxes followed by explanation, and occasional repeated close-thinking markers; they do not establish those patterns' population prevalence or systematic looping. Independent input/box/EOS audits did not demonstrate a saved-prefix, chat-boundary, EOS-classification or boxed-extractor defect. See [the versioned readout diagnosis](v2a1_readout_repair_decision_v1_20261008.md).

### New CPU-only reference gate

The twelve-problem TRAIN panel was frozen by the predeclared hash ranking before reference values were parsed; all failures remained in place, with no replacement. The unchanged math-verify 0.9.0 backend parsed each reference once, without model inference or candidate equivalence evaluation.

| CPU diagnostic | Observed result | Required gate |
|---|---:|---:|
| Nonempty parser outputs, all fixed panel references | 4 / 12 | At least 10 / 12 |
| Difficulty 7 nonempty parser outputs | 3 / 6 | At least 5 / 6 |
| Difficulty 8 nonempty parser outputs | 1 / 6 | At least 5 / 6 |
| Empty parser outputs | 8 / 12 | Preserved, not replaced |
| References yielding a symbolic object and fallback string | 4 / 12 | Diagnostic reporting only |
| Candidate equivalence evaluations / GPU calls | 0 / 0 | None permitted by this gate |
| Validation/test reference values consumed | 0 | None permitted by this gate |

The result is `REFERENCE_EVALUATOR_REPRESENTATION_BLOCK`. Nonempty parsing is not semantic validation or proof of whole-expression coverage; every reference's coverage remained unverified. The fixed gate ends here. It does not authorize wrappers, evaluator changes, new panels, or relabeling of the original study. The aggregate-safe counterpart is [the measurement diagnosis summary](../reports/v2a1_measurement_diagnosis_v1.json).

## 3. Predictive comparison and uncertainty: NOT RUN

The planned comparison uses A (structural/problem features), B (A plus current-readout confidence), and C (B plus one layer's activation vector), with L2 logistic regression, training-only preprocessing, equal-total-problem weights, and validation-only regularization/layer selection. The primary estimand is paired held-out B-minus-C log loss on the same activation-available cohort. A 2,000-resample problem-cluster bootstrap and same-problem/same-coordinate ranking control address correlated rows and problem confounding.

| Required result | Verified present state |
|---|---|
| Train/validation event gates | FAILED: both primary classes absent |
| Fitted A/B/C models and selected layer | NOT RUN |
| Test log loss, PR-AUC, Brier score and calibration | UNTESTED; no numeric estimate |
| Paired B-minus-C effect / confidence interval | UNTESTED; zero is not an estimate |
| Matched within-problem concordance | UNTESTED; no eligible primary pairs |
| Full descriptive test-layer curve | NOT RUN |

The original floors are 50 rows from 15 independent training problems per class, and 20 rows from eight validation/test problems per class; the matched test control additionally requires 15 groups from eight problems. These are feasibility floors, not formal power guarantees. A 12-problem repair pilot cannot meet the full training diversity floor.

## 4. Completed provenance and reuse boundary

The recovered archive has a verified exact inventory, every payload size/SHA256 check, gzip completion, and successful fresh local extraction. This supports artifact completeness and byte identity, not mathematical label validity or a new tensor-value analysis.

| Identity | SHA256 / value |
|---|---|
| Scientific run | `v2-run-498f9d84f5b45d26d732` |
| Source commit | `926d076f3baf7b5d31328f8865bd70cefe8396c1` |
| Source tree | `d44acd46935a0a51911f0cc57c07d776f98f242e` |
| Semantic configuration | `bbab807d3ead00739dcab7482eecb6df763e72fa0308c19b12f0a6e06f30a6b3` |
| Canonical manifest | `8eb97c603f9731a8f482cfc855ba048d03613de420c846eba5a11f28fa80dc22` |
| Canonical grouped split | `bad1088f2857f421326c2ce4d7ff94d0f913fef2df4ddc3d8ab5b1edd2052685` |
| Full recovered archive | `213cb4055e3381a6cdff043d305b63ad0f36e735b32b043a8a7b61926065032d` |
| Scientific inventory | `b7782d22c207e8463fd4b41ecf772282a69f305ddc517c44239805df86198775` |
| Fresh extraction verification report | `85f24f02eb3e07dc8aaddd013aba65445701d1f25fd29b46c45c8979206fe6fa` |
| Aggregate train/validation eligibility audit | `1be1bdb6eaa1d86a502e7facd92bb42f7039cfe97bceb077ff62cf5ca9188bf4` |
| Train-only input/readout diagnosis | `3bc9b02246ba47a61b4987b13e9e7b9cf380106388b0ceef83e8cfaf80d500c0` |
| Train-only evaluator-reason/timing diagnosis | `78fd34e0ab4f9f011916a0ca622d64b755a03bf82f8bb45a1191bd6301ea58de` |
| Fixed TRAIN reference panel | `835e8baed59856ad6e6146fcf33b1f16526a539a7827bc669753c0bff8a36585` |
| CPU reference-gate result | `d622a40ede276304165f233f55578697fe2420df838d774d34f699db8cfc553f` |

Saved trajectories and their exact prefixes, structural features, and pre-cue activation matrices are reusable under the same input/model/runtime contract. Original answer outputs/confidence remain reference measurements. A new answer instrument requires newly versioned readout labels and B features; full vocabulary logits and ungenerated answer continuations cannot be recovered offline. Base trajectories were stopped at close-thinking or cap, so there are no naturally emitted post-thinking final answers to adopt.

The original runner scheduled terminal readouts before checkpoint reconstruction. The defensible leakage statement is **checkpoint-information-only inputs**, not chronological feature recording before the eventual outcome. Any new feature builder must exclude terminal answers, later tokens/checkpoints, terminal length, gold answers and evaluator status, and construct features before joining labels.

## 5. Remaining execution decisions

- **CPU reference gate — FAILED:** four nonempty outputs of twelve, with three of six at difficulty 7 and one of six at difficulty 8. The original fixed gate is closed; no retry, replacement or answer modification occurred.
- **Bounded answer-instrument comparison — BLOCKED / NOT RUN:** the proposed cap-only and boxed-prefill GPU diagnostic depended on the failed reference gate. Its planning envelope of 3 paid GPU-hours / USD1.60 incremental spend is not an instruction to spend. No new trajectory or activation collection followed.
- **Train-scale revised measurement, validation confirmation, probe fitting and untouched test — UNTESTED:** each depends on the locked instrument and full event/independent-problem gates. A pilot pass does not authorize them or retroactively complete V2-A1.
- **Alternative predictive endpoint — SELECTED FOR PROTOCOL DEVELOPMENT, NOT YET TESTED:** natural close-thinking before the reasoning cap, forecast from checkpoint 512, can reuse saved data without new answer labels or GPU inference. It tests reasoning termination, not mathematical recovery. Its new protocol must freeze target definition, at-risk censoring, structural/observable baselines, held-out boundary, layer selection and uncertainty before any predictive results. It does not redefine or complete the original V2-A1 endpoint.

## 6. Reproduction and finalization checklist

This draft is not yet the complete reproducible artifact. Before final release:

1. Review the new [V2-A1 measurement diagnosis record](../reports/v2a1_measurement_diagnosis_v1.json) against its aggregate evidence; it preserves the original terminal classification, denominators, exclusions, evidence hashes and untested metrics without replacing the historical V2 summary.
2. Provide a portable CPU-only read-only audit entry point that accepts an authenticated private inventory/extraction root, verifies consumed file size/hash before parsing, refuses test bodies, reproduces the aggregate tables, and emits no benchmark text or private identifiers publicly. Existing private audits establish these counts; a public reproduction interface is still missing.
3. Preserve the recorded CPU gate failure and freeze the separate checkpoint-512 completion-forecast protocol. Add its actual analysis and terminal decision in a new namespace. Any future reference-representation adapter remains a distinct proposal, not an executed recovery repair.
4. Run offline tests for prefix boundaries, EOS/box policies, intent accounting, grouped splits, feature lineage and report serialization. Passing synthetic tests cannot substitute for real event-yield or held-out evidence.
5. If a predictive comparison executes, append its fixed selection record, paired cohort, all metrics, grouped uncertainty, matched controls, all-layer descriptive curve, failure accounting and reproducible analysis command. Otherwise retain NOT RUN and explain the genuine feasibility/budget barrier.
6. Audit public output for private data, unsupported causal/generalization claims and proposed results presented as measured results. The separate [V2-A1 interview brief](interview_brief_v2a1_v1.md) currently supports completed engineering/measurement statements only; update its terminal interpretation only after any new predictive result is verified.

## 7. Interview-ready interpretation now

Supported engineering bullets, subject to final evidence review:

- Engineered an auditable BF16 reasoning-state collection pipeline for Qwen3-1.7B; preserved 768 stochastic trajectories and 3,793 all-layer checkpoint matrices with exact token-prefix and artifact-hash provenance.
- Diagnosed a non-evaluable recovery endpoint by separating generation completion, answer extraction, EOS eligibility and reference parsing; preserved failed measurements instead of manufacturing wrong-answer labels or tuning the cohort after outcomes.

Do not claim that activations predicted recovery, that probes beat observable confidence, that the result was a negative predictive finding, that the full experiment completed, or that the project established inference savings. A rigorous measurement failure is a defensible result; an unperformed held-out comparison is not.
