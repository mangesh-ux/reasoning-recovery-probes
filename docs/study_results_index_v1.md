# Study results: start here

Date: 2026-10-08. The original V2-A1 recovery endpoint remains non-evaluable.
A separate, zero-GPU completion-forecasting comparison is now complete.

## Current completed deliverables

- [Final research report](recovery_activation_study_final_v1.md): methods,
  exploratory held-out results, uncertainty and limitations.
- [Machine-readable results](../reports/completion_forecasting_v1.json): safe
  aggregates and exact provenance hashes.
- [Interview brief](interview_brief_completion_v1.md): current explanation and
  defensible resume bullets.
- [CPU reproduction guide](completion_forecasting_reproduction_v1.md),
  [frozen new protocol](completion_forecasting_protocol_v1.md) and
  [configuration](../configs/completion_forecasting_v1.json).

Activations improved bounded natural-thinking-completion prediction relative
to the specified baselines. This is not a result about correct-answer recovery,
latent correctness or inference savings. The original test split was not used.

## Historical measurement and interim records

The following remain intact as earlier snapshots, not current completion-study
status. Their statements that the new predictive comparison was pending refer
to the time those documents were written.

- [Readout repair decision](v2a1_readout_repair_decision_v1_20261008.md): the
  original diagnosis and conditional prospective proposal; its GPU pilot was
  not executed.
- [Reference-parsability contract](reference_parsability_diagnostic_v1.md): the
  TRAIN-only CPU gate, which subsequently failed with 4/12 nonempty outputs.
- [Aggregate measurement diagnosis](../reports/v2a1_measurement_diagnosis_v1.json):
  original endpoint failure and pre-pivot status, not the new predictive result.
- [Interim report draft](recovery_study_final_report_v1_draft.md) and
  [interim V2-A1 interview brief](interview_brief_v2a1_v1.md): superseded for the
  completion endpoint by the final report and brief above.

## Publication boundary

The repository contains code, synthetic tests, aggregate results and documentation.
Raw trajectories, reference answers, per-problem records, activation tensors,
fitted-model payloads, predictions and private verification receipts remain local
under ignored artifact directories. The original executed reference-gate helper
also remains local because it contains a machine-specific cache location; its
frozen bytes were not changed for publication. The portable original-measurement
auditor and completion-study implementation are included.

New study files have byte-preserving Git attributes so Windows line-ending
conversion does not invalidate their recorded SHA256 identities. This changes
no original scientific configuration, label, split or conclusion.
