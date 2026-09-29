"""Intent-first V2 collection and private receipt-to-feature reconstruction.

This is intentionally separate from the P0 runner.  V2 has different
terminal semantics, symmetric readouts, all-layer activations, grouped splits,
and a test-unblinding boundary.  Raw receipts remain inside the ignored V2
artifact root; public callers receive only aggregate-safe summaries.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .artifacts import read_json
from .evaluation import EvaluationBackend, classify_transition, evaluate_extraction
from .extraction import extract_last_balanced_boxed
from .provenance import sha256_json, sha256_token_ids, utc_now_iso
from .v2_activations import (
    build_activation_input,
    serialize_activation_matrix,
)
from .v2_artifacts import (
    V2IntentState,
    V2RunLayout,
    v2_checkpoint_id,
    v2_rollout_id,
)
from .v2_boundary import construct_fixed_checkpoints, construct_reasoning_boundary, normalise_extracted_answer_for_agreement
from .v2_config import V2Config
from .v2_dataset import V2ProblemRecord, derive_v2_rollout_seed
from .v2_features import (
    CheckpointReadoutFeatureInput,
    V2PrimaryRow,
    build_v2_feature_rows,
    enroll_v2_primary_rows,
)
from .v2_runtime import V2GenerationRuntimeError, V2Runtime


class V2CollectionError(RuntimeError):
    """A V2 raw receipt cannot be safely constructed or recovered."""


class V2RuntimeSurface(Protocol):
    """The narrow model interface used by the collection runner."""

    @property
    def eos_token_ids(self) -> tuple[int, ...]: ...

    @property
    def provenance(self) -> Mapping[str, object]: ...

    @property
    def torch(self) -> Any: ...

    def tokenize_prompt(self, problem: str) -> Any: ...

    def generate_reasoning(
        self, input_token_ids: Sequence[int], *, seed: int, close_think_token_ids: Sequence[int]
    ) -> Any: ...

    def generate_readout(self, input_token_ids: Sequence[int]) -> Any: ...

    def extract_checkpoint_activation(self, activation_input: Any) -> Any: ...


@dataclass(frozen=True)
class V2CollectionExecution:
    """Aggregate operational accounting for one non-test or test split run."""

    split: str
    planned_problems: int
    planned_rollouts: int
    completed_base_rollouts: int
    failed_base_rollouts: int
    completed_terminal_readouts: int
    completed_checkpoint_readouts: int
    completed_activations: int
    unavailable_checkpoints: int
    failures_by_kind: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "planned_problems": self.planned_problems,
            "planned_rollouts": self.planned_rollouts,
            "completed_base_rollouts": self.completed_base_rollouts,
            "failed_base_rollouts": self.failed_base_rollouts,
            "completed_terminal_readouts": self.completed_terminal_readouts,
            "completed_checkpoint_readouts": self.completed_checkpoint_readouts,
            "completed_activations": self.completed_activations,
            "unavailable_checkpoints": self.unavailable_checkpoints,
            "failures_by_kind": dict(sorted(self.failures_by_kind.items())),
        }


@dataclass(frozen=True)
class V2AnalysisInputs:
    """Private row/tensor inputs whose aggregate consumers must not serialize IDs."""

    rows: tuple[V2PrimaryRow, ...]
    activation_matrices: Mapping[str, Any]
    all_label_counts: Mapping[str, int]
    feature_exclusion_counts: Mapping[str, int]
    raw_checkpoint_receipt_count: int

    def safe_counts(self) -> dict[str, object]:
        return {
            "primary_row_count": len(self.rows),
            "primary_label_counts": dict(sorted(Counter(row.transition_label for row in self.rows).items())),
            "all_transition_label_counts": dict(sorted(self.all_label_counts.items())),
            "feature_exclusion_counts": dict(sorted(self.feature_exclusion_counts.items())),
            "activation_matrix_row_count": len(self.activation_matrices),
            "raw_checkpoint_receipt_count": self.raw_checkpoint_receipt_count,
        }


def execute_v2_collection(
    *,
    config: V2Config,
    records: Sequence[V2ProblemRecord],
    split: str,
    layout: V2RunLayout,
    runtime: V2RuntimeSurface,
    evaluator: EvaluationBackend | None,
    source_identity: Mapping[str, object],
) -> V2CollectionExecution:
    """Collect exactly one frozen split; never regenerate an attempted request."""

    if split not in {"train", "validation", "test"}:
        raise V2CollectionError("V2 collection split must be train, validation, or test")
    if not records:
        raise V2CollectionError("V2 collection needs at least one selected record")
    commit = source_identity.get("commit")
    if not isinstance(commit, str) or not commit:
        raise V2CollectionError("V2 collection needs committed source identity")
    marker_ids, cue_ids = _validated_controls(runtime, config)
    execution = _MutableExecution(split=split, planned_problems=len(records))
    for record in sorted(records, key=lambda item: item.problem_id):
        for rollout_index in config.generation.rollout_indices:
            execution.planned_rollouts += 1
            rollout_identity = v2_rollout_id(
                layout.manifest_hash, record.problem_id, rollout_index
            )
            rollout_seed = derive_v2_rollout_seed(
                root_seed=config.generation.rollout_seed_root,
                problem_id=record.problem_id,
                rollout_index=rollout_index,
            )
            base = _ensure_base_receipt(
                config=config,
                layout=layout,
                runtime=runtime,
                record=record,
                split=split,
                rollout_identity=rollout_identity,
                rollout_index=rollout_index,
                rollout_seed=rollout_seed,
                marker_ids=marker_ids,
                source_commit=commit,
            )
            if base.get("status") != "COMPLETED":
                execution.failed_base_rollouts += 1
                for position in config.generation.checkpoint_token_positions:
                    checkpoint_identity = v2_checkpoint_id(rollout_identity, position)
                    _write_unavailable_checkpoint_pair(
                        config=config,
                        layout=layout,
                        record=record,
                        split=split,
                        rollout_identity=rollout_identity,
                        checkpoint_identity=checkpoint_identity,
                        checkpoint_token=position,
                        availability_status="PARENT_BASE_NOT_COMPLETED",
                        reason="no saved base reasoning trajectory is available",
                        source_commit=commit,
                    )
                    execution.unavailable_checkpoints += 1
                continue
            execution.completed_base_rollouts += 1
            terminal = _ensure_terminal_readout(
                config=config,
                layout=layout,
                runtime=runtime,
                evaluator=evaluator,
                record=record,
                split=split,
                rollout_identity=rollout_identity,
                base_receipt=base,
                cue_ids=cue_ids,
                marker_ids=marker_ids,
                source_commit=commit,
            )
            if terminal.get("status") == "COMPLETED":
                execution.completed_terminal_readouts += 1
            else:
                execution.failures["terminal_readout"] += 1
            boundary = _boundary_from_base(base, runtime=runtime, marker_ids=marker_ids, config=config)
            checkpoints = construct_fixed_checkpoints(
                boundary, config.generation.checkpoint_token_positions
            )
            prompt_ids = _token_ids_from_generation(base, "input_token_ids")
            for checkpoint in checkpoints:
                checkpoint_identity = v2_checkpoint_id(
                    rollout_identity, checkpoint.token_position
                )
                if not checkpoint.available:
                    _write_unavailable_checkpoint_pair(
                        config=config,
                        layout=layout,
                        record=record,
                        split=split,
                        rollout_identity=rollout_identity,
                        checkpoint_identity=checkpoint_identity,
                        checkpoint_token=checkpoint.token_position,
                        availability_status=checkpoint.status.value,
                        reason=checkpoint.reason,
                        source_commit=commit,
                    )
                    execution.unavailable_checkpoints += 1
                    continue
                assert checkpoint.prefix_generated_token_ids is not None
                readout = _ensure_checkpoint_readout(
                    config=config,
                    layout=layout,
                    runtime=runtime,
                    evaluator=evaluator,
                    record=record,
                    split=split,
                    rollout_identity=rollout_identity,
                    checkpoint_identity=checkpoint_identity,
                    checkpoint_token=checkpoint.token_position,
                    prompt_token_ids=prompt_ids,
                    prefix_token_ids=checkpoint.prefix_generated_token_ids,
                    prefix_sha256=checkpoint.prefix_sha256,
                    cue_ids=cue_ids,
                    terminal_receipt=terminal,
                    source_commit=commit,
                )
                if readout.get("status") == "COMPLETED":
                    execution.completed_checkpoint_readouts += 1
                else:
                    execution.failures["checkpoint_readout"] += 1
                activation = _ensure_checkpoint_activation(
                    config=config,
                    layout=layout,
                    runtime=runtime,
                    record=record,
                    split=split,
                    rollout_identity=rollout_identity,
                    checkpoint_identity=checkpoint_identity,
                    checkpoint_token=checkpoint.token_position,
                    prompt_token_ids=prompt_ids,
                    prefix_token_ids=checkpoint.prefix_generated_token_ids,
                    prefix_sha256=checkpoint.prefix_sha256,
                    source_commit=commit,
                )
                if activation.get("status") == "COMPLETED":
                    execution.completed_activations += 1
                else:
                    execution.failures["activation"] += 1
    return execution.freeze()


def load_v2_analysis_inputs(
    *,
    config: V2Config,
    records: Sequence[V2ProblemRecord],
    layout: V2RunLayout,
) -> V2AnalysisInputs:
    """Build V2 rows after collection without reading terminal values as features."""

    observations: list[CheckpointReadoutFeatureInput] = []
    labels: dict[str, str | None] = {}
    matrices: dict[str, Any] = {}
    labels_all: Counter[str] = Counter()
    receipt_count = 0
    for record in records:
        for rollout_index in config.generation.rollout_indices:
            rollout_identity = v2_rollout_id(layout.manifest_hash, record.problem_id, rollout_index)
            for position in config.generation.checkpoint_token_positions:
                checkpoint_identity = v2_checkpoint_id(rollout_identity, position)
                path = layout.checkpoint_readout_path(checkpoint_identity)
                if not path.exists():
                    continue
                receipt_count += 1
                receipt = read_json(path)
                if receipt.get("status") != "COMPLETED":
                    continue
                readout = receipt.get("readout")
                if not isinstance(readout, Mapping):
                    raise V2CollectionError("completed checkpoint receipt lacks a readout")
                observable = readout.get("observable_scores")
                if not isinstance(observable, Mapping):
                    raise V2CollectionError("completed checkpoint readout lacks observable scores")
                extraction = receipt.get("extraction")
                extracted = (
                    extraction.get("extracted_text")
                    if isinstance(extraction, Mapping) and extraction.get("status") == "EXTRACTED"
                    else None
                )
                observations.append(
                    CheckpointReadoutFeatureInput(
                        row_id=checkpoint_identity,
                        problem_id=record.problem_id,
                        rollout_id=rollout_identity,
                        checkpoint_token=position,
                        problem_token_count=_problem_token_count_from_base(layout, rollout_identity),
                        difficulty=record.difficulty,
                        topic=record.topic,
                        mean_answer_token_logprob=_optional_float(
                            observable.get("mean_answer_token_logprob")
                        ),
                        min_answer_token_logprob=_optional_float(
                            observable.get("min_answer_token_logprob")
                        ),
                        answer_token_count=_optional_int(observable.get("answer_token_count")),
                        first_readout_token_entropy_nats=_optional_float(
                            observable.get("first_readout_token_entropy_nats")
                        ),
                        first_readout_token_probability_margin=_optional_float(
                            observable.get("first_readout_token_probability_margin")
                        ),
                        normalized_extracted_answer=normalise_extracted_answer_for_agreement(
                            extracted if isinstance(extracted, str) else None
                        ),
                        readout_available=True,
                    )
                )
                label = receipt.get("transition_label")
                if label is not None and not isinstance(label, str):
                    raise V2CollectionError("checkpoint transition label has invalid type")
                labels[checkpoint_identity] = label
                if isinstance(label, str):
                    labels_all[label] += 1
                activation_path = layout.activation_tensor_path(checkpoint_identity)
                activation_receipt_path = layout.activation_receipt_path(checkpoint_identity)
                if activation_path.exists() and activation_receipt_path.exists():
                    activation_receipt = read_json(activation_receipt_path)
                    if activation_receipt.get("status") == "COMPLETED":
                        matrices[checkpoint_identity] = _load_activation_tensor(
                            activation_path, config=config, expected_sha256=activation_receipt.get("activation", {}).get("matrix_sha256") if isinstance(activation_receipt.get("activation"), Mapping) else None
                        )
    feature_rows = build_v2_feature_rows(
        observations,
        normalized_checkpoint_denominator=config.features.normalized_checkpoint_denominator,
    )
    enrollment = enroll_v2_primary_rows(feature_rows, labels)
    return V2AnalysisInputs(
        rows=enrollment.rows,
        activation_matrices=matrices,
        all_label_counts=dict(labels_all),
        feature_exclusion_counts=enrollment.exclusion_counts,
        raw_checkpoint_receipt_count=receipt_count,
    )


def _ensure_base_receipt(
    *,
    config: V2Config,
    layout: V2RunLayout,
    runtime: V2RuntimeSurface,
    record: V2ProblemRecord,
    split: str,
    rollout_identity: str,
    rollout_index: int,
    rollout_seed: int,
    marker_ids: Sequence[int],
    source_commit: str,
) -> dict[str, Any]:
    existing = _recover_or_existing(
        layout=layout,
        path=layout.rollout_path(rollout_identity),
        intent_id=f"base:{rollout_identity}",
        record_type="V2_BASE_ROLLOUT",
        identity=_base_identity(
            config, layout, record, split, rollout_identity, rollout_index, rollout_seed, source_commit
        ),
    )
    if existing is not None:
        return existing
    ledger = layout.ledger
    ledger.append("INTENT_STARTED", intent_id=f"base:{rollout_identity}", payload={"kind": "base_reasoning", "rollout_id": rollout_identity})
    prompt = None
    try:
        prompt = runtime.tokenize_prompt(record.question)
        generation = runtime.generate_reasoning(
            prompt.token_ids, seed=rollout_seed, close_think_token_ids=marker_ids
        )
        boundary = construct_reasoning_boundary(
            generation.generated_token_ids,
            close_think_token_ids=marker_ids,
            eos_token_ids=runtime.eos_token_ids,
            max_new_tokens=config.generation.reasoning_max_new_tokens,
            termination_status=generation.termination_status,
        )
        receipt = {
            **_base_identity(config, layout, record, split, rollout_identity, rollout_index, rollout_seed, source_commit),
            "status": "COMPLETED",
            "prompt": {
                "user_prompt": prompt.user_prompt,
                "chat_prompt_text": prompt.chat_prompt_text,
                "problem_token_count": prompt.problem_token_count,
                "prompt_token_ids": list(prompt.token_ids),
                "prompt_token_ids_sha256": prompt.token_ids_sha256,
            },
            "generation": generation.to_dict(include_input_token_ids=True),
            "reasoning_boundary": boundary.safe_summary(),
            "runtime_provenance_hash": sha256_json(runtime.provenance),
            "timestamp_utc": utc_now_iso(),
        }
        layout.write_rollout(rollout_identity, receipt)
        ledger.append("INTENT_COMPLETED", intent_id=f"base:{rollout_identity}", payload={"kind": "base_reasoning"})
        return receipt
    except Exception as error:
        receipt = {
            **_base_identity(config, layout, record, split, rollout_identity, rollout_index, rollout_seed, source_commit),
            "status": "FAILED",
            "prompt": _prompt_payload(prompt),
            "failure": _failure_payload(error),
            "timestamp_utc": utc_now_iso(),
        }
        layout.write_rollout(rollout_identity, receipt)
        ledger.append("INTENT_FAILED", intent_id=f"base:{rollout_identity}", payload={"kind": "base_reasoning", "failure": receipt["failure"]})
        return receipt


def _ensure_terminal_readout(
    *,
    config: V2Config,
    layout: V2RunLayout,
    runtime: V2RuntimeSurface,
    evaluator: EvaluationBackend | None,
    record: V2ProblemRecord,
    split: str,
    rollout_identity: str,
    base_receipt: Mapping[str, Any],
    cue_ids: Sequence[int],
    marker_ids: Sequence[int],
    source_commit: str,
) -> dict[str, Any]:
    identity = _readout_identity(config, layout, record, split, rollout_identity, None, source_commit)
    existing = _recover_or_existing(
        layout=layout,
        path=layout.terminal_readout_path(rollout_identity),
        intent_id=f"terminal-readout:{rollout_identity}",
        record_type="V2_TERMINAL_READOUT",
        identity=identity,
    )
    if existing is not None:
        return existing
    boundary = _boundary_from_base(base_receipt, runtime=runtime, marker_ids=marker_ids, config=config)
    if not boundary.has_terminal_reasoning_prefix:
        receipt = {
            **identity,
            "status": "UNAVAILABLE",
            "reason": boundary.reason,
            "terminal_reasoning_boundary": boundary.safe_summary(),
            "timestamp_utc": utc_now_iso(),
        }
        layout.write_terminal_readout(rollout_identity, receipt)
        return receipt
    assert boundary.terminal_reasoning_prefix_token_ids is not None
    prompt_ids = _token_ids_from_generation(base_receipt, "input_token_ids")
    input_ids = prompt_ids + boundary.terminal_reasoning_prefix_token_ids + tuple(cue_ids)
    ledger = layout.ledger
    ledger.append("INTENT_STARTED", intent_id=f"terminal-readout:{rollout_identity}", payload={"kind": "terminal_readout", "rollout_id": rollout_identity})
    try:
        readout = runtime.generate_readout(input_ids)
        extraction = extract_last_balanced_boxed(readout.generation.generated_text)
        evaluation = evaluate_extraction(record.reference_answer, extraction, backend=evaluator)
        receipt = {
            **identity,
            "status": "COMPLETED",
            "terminal_reasoning_boundary": boundary.safe_summary(),
            "readout_input": {
                "prompt_token_count": len(prompt_ids),
                "terminal_prefix_token_count": len(boundary.terminal_reasoning_prefix_token_ids),
                "terminal_prefix_sha256": boundary.terminal_prefix_sha256,
                "cue_token_ids": list(cue_ids),
                "cue_sha256": sha256_token_ids(cue_ids),
                "input_token_ids": list(input_ids),
                "input_sha256": sha256_token_ids(input_ids),
            },
            "readout": readout.to_dict(include_input_token_ids=False),
            "extraction": extraction.to_dict(),
            "evaluation": evaluation.to_dict(),
            "timestamp_utc": utc_now_iso(),
        }
        layout.write_terminal_readout(rollout_identity, receipt)
        ledger.append("INTENT_COMPLETED", intent_id=f"terminal-readout:{rollout_identity}", payload={"kind": "terminal_readout"})
        return receipt
    except Exception as error:
        receipt = {**identity, "status": "FAILED", "failure": _failure_payload(error), "timestamp_utc": utc_now_iso()}
        layout.write_terminal_readout(rollout_identity, receipt)
        ledger.append("INTENT_FAILED", intent_id=f"terminal-readout:{rollout_identity}", payload={"kind": "terminal_readout", "failure": receipt["failure"]})
        return receipt


def _ensure_checkpoint_readout(
    *,
    config: V2Config,
    layout: V2RunLayout,
    runtime: V2RuntimeSurface,
    evaluator: EvaluationBackend | None,
    record: V2ProblemRecord,
    split: str,
    rollout_identity: str,
    checkpoint_identity: str,
    checkpoint_token: int,
    prompt_token_ids: Sequence[int],
    prefix_token_ids: Sequence[int],
    prefix_sha256: str | None,
    cue_ids: Sequence[int],
    terminal_receipt: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    identity = _readout_identity(config, layout, record, split, rollout_identity, checkpoint_token, source_commit, checkpoint_identity)
    existing = _recover_or_existing(layout=layout, path=layout.checkpoint_readout_path(checkpoint_identity), intent_id=f"checkpoint-readout:{checkpoint_identity}", record_type="V2_CHECKPOINT_READOUT", identity=identity)
    if existing is not None:
        return existing
    input_ids = tuple(prompt_token_ids) + tuple(prefix_token_ids) + tuple(cue_ids)
    ledger = layout.ledger
    ledger.append("INTENT_STARTED", intent_id=f"checkpoint-readout:{checkpoint_identity}", payload={"kind": "checkpoint_readout", "checkpoint_id": checkpoint_identity})
    try:
        readout = runtime.generate_readout(input_ids)
        extraction = extract_last_balanced_boxed(readout.generation.generated_text)
        evaluation = evaluate_extraction(record.reference_answer, extraction, backend=evaluator)
        transition_label, eligibility = _transition_label(
            checkpoint_readout=readout.to_dict(include_input_token_ids=False),
            checkpoint_evaluation=evaluation.to_dict(),
            terminal_receipt=terminal_receipt,
        )
        receipt = {
            **identity,
            "status": "COMPLETED",
            "availability_status": "AVAILABLE",
            "saved_prefix_token_count": checkpoint_token,
            "saved_prefix_sha256": prefix_sha256,
            "readout_input": {"prompt_token_count": len(prompt_token_ids), "prefix_token_count": len(prefix_token_ids), "prefix_sha256": prefix_sha256, "cue_token_ids": list(cue_ids), "cue_sha256": sha256_token_ids(cue_ids), "input_token_ids": list(input_ids), "input_sha256": sha256_token_ids(input_ids)},
            "readout": readout.to_dict(include_input_token_ids=False),
            "extraction": extraction.to_dict(),
            "evaluation": evaluation.to_dict(),
            "terminal_evaluation_reference": _terminal_evaluation_reference(terminal_receipt),
            "transition_label": transition_label,
            "analytic_eligibility": eligibility,
            "timestamp_utc": utc_now_iso(),
        }
        layout.write_checkpoint_readout(checkpoint_identity, receipt)
        ledger.append("INTENT_COMPLETED", intent_id=f"checkpoint-readout:{checkpoint_identity}", payload={"kind": "checkpoint_readout"})
        return receipt
    except Exception as error:
        receipt = {**identity, "status": "FAILED", "availability_status": "AVAILABLE", "saved_prefix_token_count": checkpoint_token, "saved_prefix_sha256": prefix_sha256, "terminal_evaluation_reference": _terminal_evaluation_reference(terminal_receipt), "transition_label": None, "analytic_eligibility": {"eligible": False, "reason_codes": ["CHECKPOINT_READOUT_FAILED"]}, "failure": _failure_payload(error), "timestamp_utc": utc_now_iso()}
        layout.write_checkpoint_readout(checkpoint_identity, receipt)
        ledger.append("INTENT_FAILED", intent_id=f"checkpoint-readout:{checkpoint_identity}", payload={"kind": "checkpoint_readout", "failure": receipt["failure"]})
        return receipt


def _ensure_checkpoint_activation(
    *,
    config: V2Config,
    layout: V2RunLayout,
    runtime: V2RuntimeSurface,
    record: V2ProblemRecord,
    split: str,
    rollout_identity: str,
    checkpoint_identity: str,
    checkpoint_token: int,
    prompt_token_ids: Sequence[int],
    prefix_token_ids: Sequence[int],
    prefix_sha256: str | None,
    source_commit: str,
) -> dict[str, Any]:
    identity = _activation_identity(config, layout, record, split, rollout_identity, checkpoint_identity, checkpoint_token, source_commit)
    existing = _recover_or_existing(layout=layout, path=layout.activation_receipt_path(checkpoint_identity), intent_id=f"activation:{checkpoint_identity}", record_type="V2_CHECKPOINT_ACTIVATION", identity=identity)
    if existing is not None:
        return existing
    activation_input = build_activation_input(prompt_token_ids, prefix_token_ids)
    ledger = layout.ledger
    ledger.append("INTENT_STARTED", intent_id=f"activation:{checkpoint_identity}", payload={"kind": "checkpoint_activation", "checkpoint_id": checkpoint_identity})
    try:
        extraction = runtime.extract_checkpoint_activation(activation_input)
        content = serialize_activation_matrix(extraction.matrix, torch=runtime.torch)
        tensor_path = layout.write_activation_tensor(checkpoint_identity, content)
        receipt = {
            **identity,
            "status": "COMPLETED",
            "availability_status": "AVAILABLE",
            "saved_prefix_token_count": checkpoint_token,
            "saved_prefix_sha256": prefix_sha256,
            "activation_input": activation_input.safe_summary(),
            "activation": extraction.safe_summary(),
            "tensor_filename": tensor_path.name,
            "tensor_serialized_sha256": _file_sha256(tensor_path),
            "timestamp_utc": utc_now_iso(),
        }
        layout.write_activation_receipt(checkpoint_identity, receipt)
        ledger.append("INTENT_COMPLETED", intent_id=f"activation:{checkpoint_identity}", payload={"kind": "checkpoint_activation"})
        return receipt
    except Exception as error:
        receipt = {**identity, "status": "FAILED", "availability_status": "AVAILABLE", "saved_prefix_token_count": checkpoint_token, "saved_prefix_sha256": prefix_sha256, "failure": _failure_payload(error), "timestamp_utc": utc_now_iso()}
        layout.write_activation_receipt(checkpoint_identity, receipt)
        ledger.append("INTENT_FAILED", intent_id=f"activation:{checkpoint_identity}", payload={"kind": "checkpoint_activation", "failure": receipt["failure"]})
        return receipt


def _write_unavailable_checkpoint_pair(
    *, config: V2Config, layout: V2RunLayout, record: V2ProblemRecord, split: str,
    rollout_identity: str, checkpoint_identity: str, checkpoint_token: int,
    availability_status: str, reason: str | None, source_commit: str,
) -> None:
    readout_path = layout.checkpoint_readout_path(checkpoint_identity)
    if not readout_path.exists():
        layout.write_checkpoint_readout(checkpoint_identity, {
            **_readout_identity(config, layout, record, split, rollout_identity, checkpoint_token, source_commit, checkpoint_identity),
            "status": "UNAVAILABLE", "availability_status": availability_status,
            "reason": reason, "transition_label": None,
            "analytic_eligibility": {"eligible": False, "reason_codes": [availability_status]},
            "timestamp_utc": utc_now_iso(),
        })
    activation_path = layout.activation_receipt_path(checkpoint_identity)
    if not activation_path.exists():
        layout.write_activation_receipt(checkpoint_identity, {
            **_activation_identity(config, layout, record, split, rollout_identity, checkpoint_identity, checkpoint_token, source_commit),
            "status": "UNAVAILABLE", "availability_status": availability_status,
            "reason": reason, "timestamp_utc": utc_now_iso(),
        })


def _recover_or_existing(*, layout: V2RunLayout, path: Path, intent_id: str, record_type: str, identity: Mapping[str, object]) -> dict[str, Any] | None:
    if path.exists():
        receipt = read_json(path)
        if receipt.get("record_type") != record_type:
            raise V2CollectionError(f"receipt type mismatch for {path}")
        for key in ("config_hash", "run_id"):
            if receipt.get(key) != identity.get(key):
                raise V2CollectionError(f"immutable receipt identity mismatch for {path}")
        return receipt
    state = layout.ledger.preserve_interrupted_unknown(intent_id)
    if state is V2IntentState.NOT_STARTED:
        return None
    # A request may have been sent before a process crash.  Preserve uncertainty
    # instead of issuing it again, and make the evidence visible as a receipt.
    receipt = {
        **identity,
        "status": "INTERRUPTED_UNKNOWN",
        "reason": "prior intent lacked an immutable terminal receipt",
        "timestamp_utc": utc_now_iso(),
    }
    _write_by_record_type(layout, record_type, intent_id, receipt)
    return receipt


def _write_by_record_type(layout: V2RunLayout, record_type: str, intent_id: str, receipt: Mapping[str, Any]) -> None:
    if record_type == "V2_BASE_ROLLOUT":
        layout.write_rollout(str(receipt["rollout_id"]), receipt)
    elif record_type == "V2_TERMINAL_READOUT":
        layout.write_terminal_readout(str(receipt["rollout_id"]), receipt)
    elif record_type == "V2_CHECKPOINT_READOUT":
        layout.write_checkpoint_readout(str(receipt["checkpoint_id"]), receipt)
    elif record_type == "V2_CHECKPOINT_ACTIVATION":
        layout.write_activation_receipt(str(receipt["checkpoint_id"]), receipt)
    else:  # pragma: no cover - internal fixed call sites
        raise V2CollectionError(f"unknown V2 receipt type for {intent_id}")


def _boundary_from_base(base: Mapping[str, Any], *, runtime: V2RuntimeSurface, marker_ids: Sequence[int], config: V2Config) -> Any:
    generation = base.get("generation")
    if not isinstance(generation, Mapping):
        raise V2CollectionError("completed V2 base receipt lacks generation")
    return construct_reasoning_boundary(
        _token_ids_from_mapping(generation, "generated_token_ids"),
        close_think_token_ids=marker_ids,
        eos_token_ids=runtime.eos_token_ids,
        max_new_tokens=config.generation.reasoning_max_new_tokens,
        termination_status=_required_string(generation, "termination_status"),
    )


def _transition_label(*, checkpoint_readout: Mapping[str, Any], checkpoint_evaluation: Mapping[str, Any], terminal_receipt: Mapping[str, Any]) -> tuple[str | None, dict[str, object]]:
    reasons: list[str] = []
    if checkpoint_readout.get("termination_status") != "EOS":
        reasons.append("CHECKPOINT_READOUT_NOT_EOS")
    if terminal_receipt.get("status") != "COMPLETED":
        reasons.append("TERMINAL_READOUT_NOT_COMPLETED")
    terminal_readout = terminal_receipt.get("readout")
    terminal_evaluation = terminal_receipt.get("evaluation")
    if not isinstance(terminal_readout, Mapping) or terminal_readout.get("termination_status") != "EOS":
        reasons.append("TERMINAL_READOUT_NOT_EOS")
    if not isinstance(terminal_evaluation, Mapping):
        reasons.append("TERMINAL_EVALUATION_UNAVAILABLE")
    if reasons:
        return None, {"eligible": False, "reason_codes": reasons}
    from .evaluation import EvaluationResult, EvaluationStatus
    try:
        checkpoint = EvaluationResult(
            status=EvaluationStatus(str(checkpoint_evaluation.get("status"))),
            correct=checkpoint_evaluation.get("correct"),
            backend_name=checkpoint_evaluation.get("backend_name"),
            backend_version=checkpoint_evaluation.get("backend_version"),
            reason=checkpoint_evaluation.get("reason"),
            error_type=checkpoint_evaluation.get("error_type"),
            parsed_reference_count=checkpoint_evaluation.get("parsed_reference_count"),
            parsed_candidate_count=checkpoint_evaluation.get("parsed_candidate_count"),
        )
        terminal = EvaluationResult(
            status=EvaluationStatus(str(terminal_evaluation.get("status"))),
            correct=terminal_evaluation.get("correct"),
            backend_name=terminal_evaluation.get("backend_name"),
            backend_version=terminal_evaluation.get("backend_version"),
            reason=terminal_evaluation.get("reason"),
            error_type=terminal_evaluation.get("error_type"),
            parsed_reference_count=terminal_evaluation.get("parsed_reference_count"),
            parsed_candidate_count=terminal_evaluation.get("parsed_candidate_count"),
        )
    except Exception:
        return None, {"eligible": False, "reason_codes": ["EVALUATION_RECORD_MALFORMED"]}
    transition = classify_transition(checkpoint, terminal)
    if transition is None:
        return None, {"eligible": False, "reason_codes": ["NON_EVALUABLE_OR_ERROR"]}
    return transition.value, {"eligible": True, "reason_codes": []}


def _terminal_evaluation_reference(terminal_receipt: Mapping[str, Any]) -> dict[str, object]:
    evaluation = terminal_receipt.get("evaluation")
    return {"status": evaluation.get("status"), "correct": evaluation.get("correct")} if isinstance(evaluation, Mapping) else {"status": None, "correct": None}


def _base_identity(config: V2Config, layout: V2RunLayout, record: V2ProblemRecord, split: str, rollout_identity: str, rollout_index: int, rollout_seed: int, source_commit: str) -> dict[str, object]:
    return {"schema_version": 1, "record_type": "V2_BASE_ROLLOUT", "study_protocol_id": config.study.protocol_id, "config_hash": config.config_hash, "run_id": layout.run_id, "manifest_hash": layout.manifest_hash, "split_hash": layout.split_hash, "source_git_commit": source_commit, "split": split, "problem_id": record.problem_id, "source_index": record.source_index, "difficulty": record.difficulty, "topic": record.topic, "question_sha256": record.question_sha256, "reference_answer_sha256": record.reference_answer_sha256, "rollout_id": rollout_identity, "rollout_index": rollout_index, "rollout_seed": rollout_seed}


def _readout_identity(config: V2Config, layout: V2RunLayout, record: V2ProblemRecord, split: str, rollout_identity: str, checkpoint_token: int | None, source_commit: str, checkpoint_identity: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"schema_version": 1, "record_type": "V2_TERMINAL_READOUT" if checkpoint_identity is None else "V2_CHECKPOINT_READOUT", "study_protocol_id": config.study.protocol_id, "config_hash": config.config_hash, "run_id": layout.run_id, "manifest_hash": layout.manifest_hash, "split_hash": layout.split_hash, "source_git_commit": source_commit, "split": split, "problem_id": record.problem_id, "rollout_id": rollout_identity}
    if checkpoint_identity is not None:
        payload.update({"checkpoint_id": checkpoint_identity, "checkpoint_token": checkpoint_token})
    return payload


def _activation_identity(config: V2Config, layout: V2RunLayout, record: V2ProblemRecord, split: str, rollout_identity: str, checkpoint_identity: str, checkpoint_token: int, source_commit: str) -> dict[str, object]:
    return {"schema_version": 1, "record_type": "V2_CHECKPOINT_ACTIVATION", "study_protocol_id": config.study.protocol_id, "config_hash": config.config_hash, "run_id": layout.run_id, "manifest_hash": layout.manifest_hash, "split_hash": layout.split_hash, "source_git_commit": source_commit, "split": split, "problem_id": record.problem_id, "rollout_id": rollout_identity, "checkpoint_id": checkpoint_identity, "checkpoint_token": checkpoint_token}


def _validated_controls(runtime: V2RuntimeSurface, config: V2Config) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(runtime, V2Runtime):
        # Tests may provide a small surface. Production runtime has an exact
        # validation method; the fallback still enforces the token relationship.
        marker_ids = tuple(int(value) for value in getattr(runtime, "control_token_ids")(config.generation.close_think_marker_text))
        cue_ids = tuple(int(value) for value in getattr(runtime, "control_token_ids")(config.readout.cue_text))
    else:
        marker_ids, cue_ids = runtime.validate_readout_contract()
    if not marker_ids or tuple(cue_ids[: len(marker_ids)]) != marker_ids:
        raise V2CollectionError("V2 readout cue does not start with exact close-marker IDs")
    return marker_ids, cue_ids


def _token_ids_from_generation(receipt: Mapping[str, Any], key: str) -> tuple[int, ...]:
    generation = receipt.get("generation")
    if not isinstance(generation, Mapping):
        raise V2CollectionError("base receipt lacks generation mapping")
    return _token_ids_from_mapping(generation, key)


def _token_ids_from_mapping(payload: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise V2CollectionError(f"receipt field {key!r} must contain non-empty token IDs")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in value):
        raise V2CollectionError(f"receipt field {key!r} has invalid token IDs")
    return tuple(int(token) for token in value)


def _problem_token_count_from_base(layout: V2RunLayout, rollout_identity: str) -> int:
    receipt = read_json(layout.rollout_path(rollout_identity))
    prompt = receipt.get("prompt")
    if not isinstance(prompt, Mapping):
        raise V2CollectionError("base receipt lacks prompt metadata")
    value = prompt.get("problem_token_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V2CollectionError("base receipt has invalid problem token count")
    return value


def _load_activation_tensor(path: Path, *, config: V2Config, expected_sha256: object) -> Any:
    try:
        import torch
        from .v2_activations import activation_matrix_sha256
    except ModuleNotFoundError as error:  # pragma: no cover - runtime environment
        raise V2CollectionError("V2 analysis needs torch to load private activation tensors") from error
    try:
        matrix = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch fallback, still CPU-only
        matrix = torch.load(path, map_location="cpu")
    shape = tuple(int(item) for item in getattr(matrix, "shape", ()))
    expected_shape = (config.model.expected_num_hidden_layers, config.model.expected_hidden_size)
    if shape != expected_shape or matrix.dtype != torch.bfloat16 or matrix.device.type != "cpu":
        raise V2CollectionError("private activation tensor violates frozen V2 shape/dtype/device")
    actual_hash = activation_matrix_sha256(matrix, torch=torch)
    if isinstance(expected_sha256, str) and actual_hash != expected_sha256:
        raise V2CollectionError("private activation tensor hash differs from immutable receipt")
    # NumPy does not expose a BF16 dtype on every supported runtime.  The
    # private artifact remains BF16; this transient analysis copy is float32.
    return matrix.float().numpy()


def _prompt_payload(prompt: Any) -> Mapping[str, object] | None:
    if prompt is None:
        return None
    return {"user_prompt": getattr(prompt, "user_prompt", None), "chat_prompt_text": getattr(prompt, "chat_prompt_text", None), "problem_token_count": getattr(prompt, "problem_token_count", None), "prompt_token_ids": list(getattr(prompt, "token_ids", ())), "prompt_token_ids_sha256": getattr(prompt, "token_ids_sha256", None)}


def _failure_payload(error: Exception) -> dict[str, object]:
    return {"error_type": type(error).__name__, "message": str(error)[:1000], "cuda_oom": bool(getattr(error, "cuda_oom", False)), "elapsed_seconds": _optional_float(getattr(error, "elapsed_seconds", None))}


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise V2CollectionError(f"receipt field {key!r} must be a non-empty string")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2CollectionError("expected an optional numeric receipt value")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V2CollectionError("expected an optional non-negative integer receipt value")
    return value


def _file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _MutableExecution:
    split: str
    planned_problems: int
    planned_rollouts: int = 0
    completed_base_rollouts: int = 0
    failed_base_rollouts: int = 0
    completed_terminal_readouts: int = 0
    completed_checkpoint_readouts: int = 0
    completed_activations: int = 0
    unavailable_checkpoints: int = 0
    failures: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = Counter()

    def freeze(self) -> V2CollectionExecution:
        assert self.failures is not None
        return V2CollectionExecution(
            split=self.split, planned_problems=self.planned_problems,
            planned_rollouts=self.planned_rollouts,
            completed_base_rollouts=self.completed_base_rollouts,
            failed_base_rollouts=self.failed_base_rollouts,
            completed_terminal_readouts=self.completed_terminal_readouts,
            completed_checkpoint_readouts=self.completed_checkpoint_readouts,
            completed_activations=self.completed_activations,
            unavailable_checkpoints=self.unavailable_checkpoints,
            failures_by_kind=dict(self.failures),
        )
