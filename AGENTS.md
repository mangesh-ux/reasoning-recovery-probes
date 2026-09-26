# P0 research operating contract

## Scope

This repository implements only Pilot P0 for the Reasoning Recovery Probes
feasibility study. It is separate from `reasoning-confidence-audit` and must
not modify, reinterpret, or reuse its frozen artifacts.

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
- Do not run the full P0 pilot automatically. Build and review the offline
  tests and 1-problem x 1-rollout smoke path first.
- Capture model, dataset, tokenizer, runtime, decoding, configuration, and
  source provenance with every runnable artifact set.

## Decision boundary

Before the first model-backed run, review `docs/decision_log.md`. A change to
the dataset/model revision, selection manifest, checkpoint coordinate,
generation/forced-answer semantics, answer extraction, evaluator, or pilot
scope creates a new configuration and must be recorded rather than silently
folded into an existing result set.
