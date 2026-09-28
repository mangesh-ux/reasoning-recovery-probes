# P0 research operating contract

## Scope

This repository begins with Pilot P0 for the Reasoning Recovery Probes
feasibility study. It is separate from `reasoning-confidence-audit` and must
not modify, reinterpret, or reuse its frozen artifacts. A later activation
study may be implemented only after P0 evidence, a viability decision, and a
separate frozen protocol are recorded.

## Non-negotiable rules

- P0 is feasibility-only data. It must not become confirmatory data without a
  separately approved protocol.
- Do not train probes, classifiers, logistic regressions, or activation-based
  predictors in P0.
- Do not enable hidden-state extraction in normal trajectory generation.
- Original stochastic trajectories are generated once. Exact saved token IDs
  are the source of truth for checkpoint prefixes and final-answer labels.
- Preserve failures, non-evaluable cases, malformed answers, interrupted
  requests, and CUDA OOMs. Never replace or silently relabel them.
- Fixed checkpoints are generated-token positions only. Do not substitute
  semantic reflection markers.
- The authorized P0 campaign is one immutable 30-problem x 4-seed manifest.
  First execute its first 10 problems x 4 seeds. Extend that same campaign to
  all 30 problems only after the predeclared operational gate in
  `docs/p0_execution_authorization.md` passes. Do not replace, duplicate, or
  silently broaden its first-stage trajectories.
- Do not collect activations or train predictive models before a P0 viability
  decision is recorded. Later activation code must use a separately frozen
  protocol and must not reinterpret P0 as confirmatory data.
- Capture model, dataset, tokenizer, runtime, decoding, configuration, and
  source provenance with every runnable artifact set.

## Decision boundary

Before the first model-backed run, review `docs/decision_log.md`. A change to
the dataset/model revision, selection manifest, checkpoint coordinate,
generation/forced-answer semantics, answer extraction, evaluator, or pilot
scope creates a new configuration and must be recorded rather than silently
folded into an existing result set.
