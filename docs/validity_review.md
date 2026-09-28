# P0 validity review

**Status:** completed. A fatal event-yield limitation blocks the proposed final
activation study before any activation or probe result was inspected.

The review applies the frozen primary definition in
[pilot_p0_protocol.md](pilot_p0_protocol.md): a W_TO_C or W_TO_W label is a
property of a forced answer at an exact saved prefix and the final answer of
that same saved stochastic trajectory. It is not a causal effect of continuing
to reason.

## Review

| Threat | Status | Evidence and consequence |
| --- | --- | --- |
| Insufficient recovery and non-recovery events | Unresolved — fatal | Strict primary analysis has 2 W_TO_C and 0 W_TO_W events. There is no negative class for the proposed recovery classifier. |
| Problem-difficulty confounding | Unresolved — fatal | Both strict W_TO_C events come from one problem. No within-problem recovery-versus-non-recovery comparison exists. |
| Repeated rollouts and checkpoints | Partially addressed | P0 records the clustered structure and reports problem and problem-by-checkpoint concentration. It does not treat repeated checkpoint rows as independent evidence. No valid final sample remains. |
| Leakage across train, validation, and test problems | Partially addressed | A future study would require problem-grouped splitting, but a final split was not frozen because its target class is absent. |
| Checkpoint-position effects | Partially addressed | Coordinates were fixed before generation and availability is reported by position. Wrong checkpoint outcomes occur only at 256 and 512 in the strict analysis. |
| Forced-answer artifact | Unresolved | The forced cue and greedy decoding are explicit and frozen, but the event scarcity prevents assessing how much the target depends on that operational readout. |
| Malformed answers and evaluator behavior | Partially addressed | Non-evaluable outcomes and caps are retained; evaluator errors are zero. Base non-evaluable answers are 52/120 and forced non-evaluable answers are 284/464 available checkpoints, which materially reduce usable labels. |
| Class imbalance | Unresolved — fatal | The strict and observed views both have zero W_TO_W events. Metrics such as PR-AUC, log loss, Brier score, calibration, and AUROC cannot provide a meaningful held-out two-class result. |
| Probe capacity and hyperparameter overfitting | Addressed by stopping | No probe, classifier, tuning loop, or activation representation was selected or run. |
| Layer-selection multiplicity | Addressed by stopping | No layerwise extraction or comparison was performed. |
| Model-specific and dataset-specific scope | Unresolved | Even a future positive result would be limited to a separately specified model, dataset, and forced-answer operationalization. P0 supplies no broader claim. |
| Stochastic continuation versus causal value | Addressed | The claim boundary is maintained: the saved original trajectory defines the outcome, and no causal or early-stopping conclusion is made. |

## Gate decision

The final activation experiment is **not scientifically viable under the
frozen P0 evidence**. This conclusion does not invent a post hoc numeric
threshold: it follows from the complete absence of the required W_TO_W
comparison class and of any matched within-problem contrast.

Do not collect activations, implement a final probe pipeline, fit A/B/C
models, or alter P0 to generate more favorable events. A future attempt would
require a newly authorized scientific protocol, independently frozen before
new model-backed data are inspected.
