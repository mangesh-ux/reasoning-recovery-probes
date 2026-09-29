# Research Interview Brief: Measurement, Fidelity, and Recovery

## The project arc

This project is a three-stage research-engineering story:

1. **Measurement audit.** I audited whether an intermediate-answer confidence instrument stopped at the candidate answer it claimed to score. The frozen Q6 holdout found that candidate-boundary termination would have reduced trial-probe generation by 79.37 percent. That is an instrumentation and trial-probe-cost observation, not a claim about end-to-end inference savings.
2. **Numerical fidelity.** I held a reasoning prefix fixed and asked whether candidate identity survived a numerical backend change. Q8 agreed with BF16 on 11 of 12 fixed-prefix candidates, while Q4 agreed on 5 of 12. The point was measurement fidelity, not a generic statement that quantization lowers accuracy.
3. **Recovery representation study.** V2 tests whether a checkpoint hidden state adds predictive information about recovery under a symmetric deterministic answer readout beyond structural and observable-confidence features. The protocol is frozen; the local machine completed synthetic qualification but hit the predeclared 48-hour hardware gate before any scientific data collection.

## Why Qwen3-1.7B?

Qwen3-1.7B is an open-weight reasoning model with explicit thinking-mode behavior. It is small enough to generate repeated stochastic trajectories and extract internal representations in BF16 under a controlled runtime contract. The objective is repeated, auditable measurement rather than using the largest possible model.

## Why DeepMath difficulty 7 and 8?

The P0 MATH-500 pilot was operationally feasible but scientifically saturated for the recovery target: it did not provide enough within-problem W_TO_C versus W_TO_W contrast. V2 therefore uses a new pre-specified DeepMath population at levels 7 and 8, chosen as a capability-frontier regime intended to avoid both trivial success and universal failure. The protocol does not claim these levels are optimal, and it never searches difficulty after probe performance is observed.

## Why BF16?

The scientific object includes hidden activations, logits, token probabilities, candidate identity, and recovery labels. If a cheap numerical backend changes a candidate under the same saved reasoning prefix, numerical representation can confound the intended reasoning-state measurement. BF16 is the V2 anchor to avoid that unnecessary confound.

## Why fixed geometric checkpoints?

The primary coordinates are 128, 256, 512, 1,024, and 2,048 generated reasoning tokens. They are deterministic, model-format independent, and allow same-problem/same-checkpoint comparisons across stochastic rollouts. They do not condition the primary design on self-reflection wording such as wait, actually, paragraphs, or sentences. Geometric spacing provides more early resolution while bounding readout and activation cost.

## Why symmetric answer readouts?

The intermediate and terminal answers are both produced by the same short deterministic operation:

~~~text
exact prompt IDs + exact saved reasoning prefix + frozen close-thinking cue
-> greedy 64-token boxed-answer readout
~~~

That avoids comparing a forced checkpoint answer with a naturally generated final answer under different generation semantics. The label asks whether a wrong current readout becomes correct after more saved reasoning, under the same measurement operation. It is not a causal early-stopping claim.

## Why logistic regression?

The primary probe is intentionally simple: L2-regularized logistic regression, one model per layer. It makes incremental value interpretable, restricts capacity, supports a clean B-versus-C comparison, and reduces the risk that a flexible model memorizes problem characteristics. Standardization and regularization selection happen only on training and validation problems.

## Why group by problem?

Multiple rollouts and checkpoints from one problem are correlated. A row-level split would leak problem characteristics from training into test. V2 assigns every rollout and checkpoint of one DeepMath problem to exactly one split, uses equal-total-problem weighting, and bootstraps held-out metrics at the problem level.

## What does problem-difficulty confounding mean?

A pooled probe may appear strong because it recognizes that some problems are easy or hard, not because it detects whether one currently-wrong trajectory will recover. V2 addresses this with the matched control: on held-out groups with the same problem and same checkpoint coordinate, compare W_TO_C and W_TO_W rollout scores. That is harder and more informative than pooled discrimination alone.

## What would a positive result mean?

It would mean that, in the frozen V2 setting, a validation-selected checkpoint residual-state vector added held-out predictive information beyond the structural and observable baselines. It would remain scoped to Qwen3-1.7B, DeepMath levels 7 and 8, BF16, the exact prompt/readout, the split, and the linear probe.

## What would a null or blocked result mean?

A null result means the tested linear activation addition did not improve the stated held-out comparison. An underpowered result means the frozen population did not contain enough independent W_TO_C and W_TO_W evidence. A hardware block means the required BF16 workload was not qualified on the tested GPU. None of these conclusions proves that recovery representations do not exist in every setting.

## Limitations

- The label is tied to a deterministic readout, not a natural final answer.
- Hidden states are model- and layer-specific.
- The study tests association, not intervention or causality.
- The dataset is a deliberately selected difficulty regime, not all mathematical reasoning.
- A complete layerwise curve is descriptive; no test-best layer is selected.
- The prompt-style transfer idea is deferred to a separately frozen, nonoverlapping exploratory protocol.

## Candidate resume wording

### Measured statements already supported

- Audited a reasoning-LLM early-stop implementation at source and token boundaries; independently quantified a 79.37 percent candidate-boundary reduction in frozen trial-probe generation, scoped explicitly as instrumentation overhead rather than end-to-end savings.
- Validated fixed-prefix numerical fidelity for reasoning measurements: Q8 matched BF16 candidate identity on 11 of 12 anchors, versus 5 of 12 for Q4.

### Measured V2 local hardware outcome

- Built and synthetically qualified a BF16, all-layer activation-probe pipeline on an 8 GiB RTX 3070 Ti Laptop GPU. All required synthetic shapes passed without OOM, but the conservative frozen workload projected to 54.37 GPU hours against a 48-hour gate; preserved the local hardware block without weakening the protocol.

### V2 wording only after a future 24 GiB scientific run

Use exactly one of the following only after V2 results are measured and reviewed:

- Positive: Designed and executed a leakage-safe, problem-grouped activation-probe study testing incremental recovery prediction beyond observable confidence, with all-layer linear probes and matched within-problem controls.
- Null: Designed and executed a leakage-safe activation-probe study; the frozen linear activation signal did not improve the held-out matched recovery comparison beyond observable confidence.
- Underpowered: Designed and executed a frozen recovery-data collection protocol and preserved an underpowered event-yield result rather than tuning the cohort after observing labels.
- Hardware block: Built and qualified a BF16 activation-probe pipeline, then documented a runtime feasibility boundary without weakening the scientific protocol.

Do not use a proposed bullet as a measured result. Keep the measurement-audit, fidelity, and recovery-study claims visibly distinct.
