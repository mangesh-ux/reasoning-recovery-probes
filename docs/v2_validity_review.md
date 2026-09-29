# V2 Validity Review

**Protocol ID:** v2-recovery-activation-probe-20260929  
**Review state:** pre-inference requirements frozen; no V2 scientific model outputs reviewed  
**Purpose:** identify threats that the implementation must test, record, and report before a V2 claim can be considered.

## Decision

V2 may proceed only as a separate study with its own configuration, manifest, split, runtime contract, receipt namespace, and analysis code. P0 remains historical feasibility evidence and is not an input population.

The central claim is deliberately narrow: a checkpoint activation may add predictive information for the future symmetric-readout label beyond pre-outcome structural and observable features. A result cannot establish causality, a safe stopping policy, natural-answer recovery, general reasoning-state detection, or cross-prompt invariance.

## Pre-inference gate matrix

| Gate | Required evidence before the next phase | Failure classification | Prohibited response |
|---|---|---|---|
| V2 namespace | Distinct V2 config, code path, artifact root, and run identity; no P0 artifact read | Protocol-integrity failure | Reusing P0 receipts or treating P0 as confirmation |
| Dataset schema | Frozen revision, required fields, exactly 80 unique clusters per difficulty stratum | Data-contract failure | Changing strata or replacing rows |
| Manifest and split | Immutable master manifest and grouped 96/32/32 split; duplicate clusters do not cross splits | Selection-integrity failure | Row-level split or outcome-informed reselection |
| Runtime identity | Model/tokenizer commits, package/runtime versions, template hash, cue IDs, source commit/tree identity | Runtime-provenance failure | Continuing in the same run namespace |
| Symmetric readout | Terminal and checkpoint inputs use the identical readout builder, cue, decoder, cap, parser, and evaluator | Measurement-integrity failure | Comparing a forced checkpoint answer with a natural final answer |
| Activation input | Prefix-only forward input, no readout cue, post-block hook geometry [28, 2048], CPU BF16 transfer | Representation-integrity failure | Retaining full sequence-by-layer tensors or using a cue-contaminated state |
| Synthetic hardware | Required BF16 synthetic shapes complete with headroom, provenance, timing, and storage estimate | Hardware block | Quantizing, changing cap/checkpoints, or silently offloading |
| Train/validation event yield | Both classes meet the frozen row and problem floors | Scientific block | Fitting another model or searching a new cohort |
| Validation selection | Train-only preprocessing plus immutable validation model-selection record | Leakage or selection failure | Inspecting test metrics or retuning from test data |
| Test evaluation | One test invocation, paired B/C cohort, grouped bootstrap, full layer curve, matched control | Non-evaluable or confounded result | Selecting a test-best layer or rerunning favorable variants |

## Threat assessment

| Threat | Prevention in V2 | Required reporting |
|---|---|---|
| P0 reinterpretation | Separate protocol and no P0 raw artifact reuse | State that P0 motivated V2 but contributes no V2 rows |
| Outcome-based population selection | Hash-ranked, stratified manifest before model requests | Manifest hash, stratum counts, attrition reasons |
| Topic or difficulty leakage | Topic/difficulty used only as stated baseline metadata; no output enters selection | Feature lineage table |
| Duplicate-question leakage | Normalized-question clusters stay in one split | Cluster count and cross-split audit |
| Prompt/template drift | Pinned tokenizer/model, template hash, prompt hash, cue token IDs | Runtime contract and mismatch refusal |
| Sampling drift | Frozen sampling parameters and deterministic derived seed policy | Per-rollout seed and decoding receipt |
| Natural-final versus forced mismatch | Same short deterministic readout at every checkpoint and terminal prefix | Readout input/hash equality checks |
| Future-information feature leakage | Normalized position is p/4096; B uses current readout only | Automated prohibited-field audit |
| Evaluator misclassification | CORRECT, INCORRECT, NON_EVALUABLE, and ERROR remain distinct | Full outcome and exclusion taxonomy |
| Activation prompt contamination | Cue excluded from every activation forward input | Input-hash and cue-exclusion test |
| Activation memory distortion | Post-block hooks keep only final-token BF16 vectors | Matrix shape/dtype/storage proof and peak VRAM |
| Row-level split leakage | Problem-level assignment before inference | Problem-to-split uniqueness assertion |
| Hyperparameter/layer overfit | Validation-only C and layer selection; complete test layer curve | Immutable selection record plus all test layers |
| Repeated-row pseudo-replication | Equal-total-problem weighting and problem-cluster bootstrap | Cluster count, weighted estimate, row sensitivity |
| Difficulty confounding | Matched same-problem/same-checkpoint ranking control | Matched groups, unique problems, concordance CI |
| Missing event class | Numeric event gates set before outputs | Underpowered/non-evaluable conclusion |
| Retry selection bias | Intent-first ledger and interrupted-unknown preservation | Request accounting and retry count |
| Source-code drift on resume | Git commit/tree identity bound to V2 run identity | Resume mismatch rejection |
| Unsafe release | Raw data, token IDs, completions, and tensors stay ignored | Aggregate-only public summary audit |

## Feature-lineage review

### Permitted A features

| Feature | Available at checkpoint? | Reason |
|---|---|---|
| Checkpoint token coordinate | Yes | Frozen coordinate |
| Coordinate divided by 4,096 | Yes | Fixed known budget, not future length |
| Problem token count | Yes | Frozen prompt/problem |
| Difficulty | Yes | Dataset metadata |
| Topic | Yes | Dataset metadata |

### Permitted B additions

| Feature | Available at checkpoint? | Reason |
|---|---|---|
| Candidate-token mean/minimum log probability | Yes | Current deterministic readout only |
| Candidate-token count | Yes | Current deterministic readout only |
| First-step entropy/margin | Yes | Current deterministic readout only |
| Immediate prior checkpoint agreement/availability | Yes, if prior checkpoint exists | Earlier readout in same rollout |

### Forbidden features

| Field | Why forbidden |
|---|---|
| Terminal readout result or text | Future outcome |
| Terminal reasoning length | Future trajectory information |
| Gold answer or evaluator status | Label leakage |
| Later checkpoint output | Future information |
| Natural post-thinking final answer | Breaks symmetric readout semantics |
| Readout cue activation | Lets the probe learn the measurement operation |
| P0 metadata or outcome | Cross-study contamination |

## Event and interpretation gates

The frozen numeric floors are intentionally conservative feasibility conditions, not formal power guarantees:

| Stage | Per-class row floor | Per-class unique-problem floor |
|---|---:|---:|
| Training | 50 | 15 |
| Validation | 20 | 8 |
| Test | 20 | 8 |

The matched within-problem analysis additionally requires 15 problem-by-checkpoint groups from at least 8 held-out problems. Every group must have one or more W_TO_C rollouts and one or more W_TO_W rollouts.

Interpret results as follows:

| Observation | Classification |
|---|---|
| Test B-to-C gain and matched-control support | Positive, scoped association |
| Test B-to-C gain but no matched-control support | Pooled-only, confounded |
| Both metrics defined but no incremental gain | Null for the frozen linear probe |
| A required class or matched group is absent/too scarce | Underpowered or non-evaluable |
| Runtime/preflight fails before study collection | Hardware or operational block |

No classification permits outcome-informed changes to the primary V2 protocol.

## Test-unblinding rule

The V2 implementation must produce one immutable validation-selection artifact containing:

1. train/validation manifest and split hashes;
2. preprocessing fit identity;
3. selected A and B regularization values;
4. every layer's selected C value;
5. the validation-selected primary layer;
6. validation metrics and all event-gate counts;
7. source/config/runtime identity.

Only after this artifact exists may the test runner generate or analyze test cases. The test report is one terminal output for the selected source/config/manifest identity. A source, configuration, or runtime mismatch starts a new V2 protocol namespace; it is not a resume.

## Non-claims to preserve

The final documents must not say:

- “the model has a recovery neuron”;
- “hidden activations cause recovery”;
- “this enables safe early exit”;
- “the signal is prompt-invariant”;
- “Qwen3-1.7B findings generalize to reasoning models”;
- “a non-significant or non-evaluable result proves no recovery signal exists.”

## Validity-review conclusion before execution

The V2 design is valid to implement because it explicitly addresses P0's event-yield limitation with a new, independently frozen population and a grouped evaluation design. Its validity now depends on implementation conformance, synthetic qualification, hardware qualification, event gates, and test-blind model selection. Any failure remains a result to record, not a reason to revise the frozen protocol.
