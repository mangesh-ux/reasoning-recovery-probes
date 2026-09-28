# Final activation experiment design

**Status:** not frozen and not run.

The required final design cannot be responsibly frozen from this P0 outcome.
The strict primary data contain two W_TO_C observations from one problem and
zero W_TO_W observations, with zero same-problem/same-checkpoint contrasts.
There is no two-class target on which the requested A/B/C comparison or a
grouped held-out evaluation could answer the research question.

## Decision

| Required final-study element | Decision |
| --- | --- |
| Model and revision | Not selected for a final study |
| Dataset and revision | Not selected for a final study |
| Number of problems, rollouts, and checkpoints | Not frozen |
| W_TO_C / W_TO_W study population | Not viable under completed P0 |
| Forced-answer semantics and evaluator | Not carried forward as a final protocol |
| A structural baseline | Not designed |
| B observable baseline | Not designed |
| C activation probe | Not designed |
| Activation representation and layer set | Not collected or selected |
| Grouped splits, metrics, and uncertainty method | Not frozen |
| Exploratory analyses | None run |

The P0 model, dataset, checkpoint coordinates, forced-answer semantics, and
evaluator are preserved only as frozen feasibility evidence; they are not a
final experimental specification.

## Required boundary for any future work

Proceeding would require an explicit future authorization for a new scientific
protocol before new model-backed data are collected. That authorization would
need to state a new population and final design rather than silently changing
P0 after its result is known. Until then, do not collect activations, train
probes, or claim that hidden activations do or do not add incremental recovery
information.
