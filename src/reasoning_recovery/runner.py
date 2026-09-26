"""Fail-closed orchestration for the feasibility-only P0 execution plan.

This module intentionally does not load a model or dataset at import time.  It
only coordinates an already-qualified runtime, exact saved token IDs, and
immutable local receipts.  There is no hidden-state collection, probe fitting,
or inference from activations here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .artifacts import (
    AppendOnlyLedger,
    IntentState,
    RunLayout,
    checkpoint_id,
    rollout_id,
)
from .checkpointing import CheckpointStatus, build_fixed_checkpoints, find_subsequence
from .config import PilotConfig
from .dataset import ProblemRecord
from .evaluation import EvaluationBackend, EvaluationStatus, evaluate_extraction
from .extraction import extract_last_balanced_boxed
from .forced_answer import build_forced_answer_input
from .model_runtime import (
    DecodingParameters,
    GeneratedSequence,
    TokenizedPrompt,
)
from .provenance import sha256_token_ids, utc_now_iso


class P0Runtime(Protocol):
    """The narrow runtime surface needed for offline-testable orchestration."""

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        """The effective model-generation EOS IDs."""

    def tokenize_prompt(self, user_prompt: str) -> TokenizedPrompt:
        """Return the native chat-template IDs for one initial prompt."""

    def generate(
        self, input_token_ids: Sequence[int], decoding: DecodingParameters
    ) -> GeneratedSequence:
        """Generate one batch-one continuation."""


@dataclass(frozen=True)
class RunExecution:
    """Stable identities used to construct the operational summary."""

    planned_problem_ids: tuple[str, ...]
    planned_rollout_ids: tuple[str, ...]


def execute_p0_run(
    *,
    config: PilotConfig,
    records: Sequence[ProblemRecord],
    rollout_seeds: Sequence[int],
    layout: RunLayout,
    runtime: P0Runtime,
    evaluator: EvaluationBackend,
    close_marker_token_ids: Sequence[int],
    close_think_cue_token_ids: Sequence[int],
    runtime_provenance_hash: str,
    source_git_commit: str | None,
) -> RunExecution:
    """Execute only the declared base/forced requests, preserving all outcomes.

    Existing completed, failed, and interrupted requests are never regenerated.
    Each newly issued model request is bracketed by a durable intent event and
    an immutable receipt.  Child requests use raw saved token IDs; they never
    decode/re-tokenize an original trajectory prefix.
    """

    if not records:
        raise ValueError("at least one selected problem record is required")
    if not rollout_seeds:
        raise ValueError("at least one rollout seed is required")
    marker_ids = _token_ids(close_marker_token_ids, label="close_marker_token_ids")
    cue_ids = _token_ids(close_think_cue_token_ids, label="close_think_cue_token_ids")
    if tuple(cue_ids[: len(marker_ids)]) != marker_ids:
        raise ValueError("forced cue IDs must begin with close-marker IDs")

    if not isinstance(runtime_provenance_hash, str) or len(runtime_provenance_hash) != 64:
        raise ValueError("runtime_provenance_hash must be a SHA-256 digest")
    context = _RunContext(
        config=config,
        layout=layout,
        evaluator=evaluator,
        marker_ids=marker_ids,
        cue_ids=cue_ids,
        runtime_provenance_hash=runtime_provenance_hash,
        source_git_commit=source_git_commit,
    )
    planned_rollouts: list[str] = []
    for record in records:
        for seed in rollout_seeds:
            identity = rollout_id(layout.manifest_hash, record.problem_id, int(seed))
            planned_rollouts.append(identity)
            base_receipt = _run_base_rollout(
                context=context,
                runtime=runtime,
                record=record,
                rollout_seed=int(seed),
                rollout_identity=identity,
            )
            _run_checkpoint_plan(
                context=context,
                runtime=runtime,
                record=record,
                rollout_seed=int(seed),
                rollout_identity=identity,
                base_receipt=base_receipt,
            )
    return RunExecution(
        planned_problem_ids=tuple(record.problem_id for record in records),
        planned_rollout_ids=tuple(planned_rollouts),
    )


@dataclass(frozen=True)
class _RunContext:
    config: PilotConfig
    layout: RunLayout
    evaluator: EvaluationBackend
    marker_ids: tuple[int, ...]
    cue_ids: tuple[int, ...]
    runtime_provenance_hash: str
    source_git_commit: str | None


def _run_base_rollout(
    *,
    context: _RunContext,
    runtime: P0Runtime,
    record: ProblemRecord,
    rollout_seed: int,
    rollout_identity: str,
) -> dict[str, Any]:
    ledger = context.layout.ledger
    intent_id = f"intent:{rollout_identity}"
    state, receipt = _recover_request(
        ledger=ledger,
        intent_id=intent_id,
        reader=lambda: context.layout.read_rollout(rollout_identity),
        completed_event="BASE_COMPLETED",
        failed_event="BASE_FAILED",
    )
    if state is not IntentState.NOT_STARTED:
        if receipt is None:
            receipt = _interrupted_base_receipt(
                context=context,
                record=record,
                rollout_seed=rollout_seed,
                rollout_identity=rollout_identity,
            )
            context.layout.write_rollout(rollout_identity, receipt)
        return receipt

    decoding = DecodingParameters.base(context.config.generation, rollout_seed)
    ledger.append(
        "INTENT_STARTED",
        intent_id=intent_id,
        payload={
            "request_kind": decoding.request_kind,
            "rollout_id": rollout_identity,
            "problem_id": record.problem_id,
            "rollout_seed": rollout_seed,
            "manifest_hash": context.layout.manifest_hash,
            "config_hash": context.layout.config_hash,
        },
    )
    rendered_prompt = context.config.prompt.render(record.problem)
    try:
        prompt = runtime.tokenize_prompt(rendered_prompt)
        generation = runtime.generate(prompt.token_ids, decoding)
        extraction_payload, evaluation_payload = _answer_records(
            record.reference_answer,
            generation.generated_text,
            context.evaluator,
        )
        receipt = {
            **_base_receipt_identity(
                context=context,
                record=record,
                rollout_seed=rollout_seed,
                rollout_identity=rollout_identity,
            ),
            "status": "COMPLETED",
            "prompt": _prompt_payload(prompt),
            "generation": generation.to_dict(),
            "thinking_boundary": _thinking_boundary(
                generation.generated_token_ids, context.marker_ids
            ),
            "final_extraction": extraction_payload,
            "final_evaluation": evaluation_payload,
            "timestamp_utc": utc_now_iso(),
        }
    except Exception as error:
        receipt = {
            **_base_receipt_identity(
                context=context,
                record=record,
                rollout_seed=rollout_seed,
                rollout_identity=rollout_identity,
            ),
            "status": "FAILED",
            "prompt_rendered": rendered_prompt,
            "requested_decoding": decoding.to_dict(),
            "failure": _failure_payload(error),
            "timestamp_utc": utc_now_iso(),
        }
        context.layout.write_rollout(rollout_identity, receipt)
        ledger.append(
            "BASE_FAILED",
            intent_id=intent_id,
            payload={
                "rollout_id": rollout_identity,
                "receipt_path": context.layout.rollout_path(rollout_identity).name,
            },
        )
        return receipt
    context.layout.write_rollout(rollout_identity, receipt)
    ledger.append(
        "BASE_COMPLETED",
        intent_id=intent_id,
        payload={
            "rollout_id": rollout_identity,
            "receipt_path": context.layout.rollout_path(rollout_identity).name,
        },
    )
    return receipt


def _run_checkpoint_plan(
    *,
    context: _RunContext,
    runtime: P0Runtime,
    record: ProblemRecord,
    rollout_seed: int,
    rollout_identity: str,
    base_receipt: Mapping[str, Any],
) -> None:
    """Write every declared child coordinate, including unavailable states."""

    status = base_receipt.get("status")
    positions = context.config.generation.checkpoint_token_positions
    if status != "COMPLETED":
        parent_status = (
            "PARENT_BASE_FAILED"
            if status == "FAILED"
            else "PARENT_BASE_INTERRUPTED_UNKNOWN"
        )
        for position in positions:
            _write_unavailable_checkpoint(
                context=context,
                record=record,
                rollout_seed=rollout_seed,
                rollout_identity=rollout_identity,
                checkpoint_token=position,
                availability_status=parent_status,
                reason="no saved base trajectory is available for this child request",
            )
        return

    generation = base_receipt.get("generation")
    if not isinstance(generation, Mapping):
        raise RuntimeError(f"completed base receipt has no generation: {rollout_identity}")
    generated_ids = _receipt_token_ids(
        generation.get("generated_token_ids"), label="base generated token IDs"
    )
    prompt_ids = _receipt_token_ids(
        generation.get("input_token_ids"), label="base prompt token IDs"
    )
    final_evaluation = base_receipt.get("final_evaluation")
    if not isinstance(final_evaluation, Mapping):
        raise RuntimeError(f"completed base receipt has no final evaluation: {rollout_identity}")

    checkpoints = build_fixed_checkpoints(
        generated_ids,
        positions,
        close_think_token_ids=context.marker_ids,
        eos_token_ids=runtime.eos_token_ids,
    )
    for checkpoint in checkpoints:
        child_identity = checkpoint_id(rollout_identity, checkpoint.token_position)
        if not checkpoint.available:
            _write_unavailable_checkpoint(
                context=context,
                record=record,
                rollout_seed=rollout_seed,
                rollout_identity=rollout_identity,
                checkpoint_token=checkpoint.token_position,
                availability_status=checkpoint.status.value,
                reason=checkpoint.reason,
                base_generated_token_count=len(generated_ids),
            )
            continue
        _run_forced_checkpoint(
            context=context,
            runtime=runtime,
            record=record,
            rollout_seed=rollout_seed,
            rollout_identity=rollout_identity,
            child_identity=child_identity,
            prompt_token_ids=prompt_ids,
            checkpoint=checkpoint,
            final_evaluation=final_evaluation,
        )


def _run_forced_checkpoint(
    *,
    context: _RunContext,
    runtime: P0Runtime,
    record: ProblemRecord,
    rollout_seed: int,
    rollout_identity: str,
    child_identity: str,
    prompt_token_ids: Sequence[int],
    checkpoint: Any,
    final_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    ledger = context.layout.ledger
    intent_id = f"intent:{child_identity}"
    state, receipt = _recover_request(
        ledger=ledger,
        intent_id=intent_id,
        reader=lambda: context.layout.read_checkpoint(child_identity),
        completed_event="CHECKPOINT_COMPLETED",
        failed_event="CHECKPOINT_FAILED",
    )
    if state is not IntentState.NOT_STARTED:
        if receipt is None:
            receipt = _interrupted_checkpoint_receipt(
                context=context,
                record=record,
                rollout_seed=rollout_seed,
                rollout_identity=rollout_identity,
                child_identity=child_identity,
                checkpoint_token=checkpoint.token_position,
                prefix_sha256=checkpoint.prefix_sha256,
            )
            context.layout.write_checkpoint(child_identity, receipt)
        return receipt

    decoding = DecodingParameters.forced(context.config.forced_answer.max_new_tokens)
    ledger.append(
        "INTENT_STARTED",
        intent_id=intent_id,
        payload={
            "request_kind": decoding.request_kind,
            "checkpoint_id": child_identity,
            "parent_rollout_id": rollout_identity,
            "checkpoint_token": checkpoint.token_position,
            "saved_prefix_sha256": checkpoint.prefix_sha256,
            "manifest_hash": context.layout.manifest_hash,
            "config_hash": context.layout.config_hash,
        },
    )
    try:
        forced_input = build_forced_answer_input(
            prompt_token_ids, checkpoint, context.cue_ids
        )
        generation = runtime.generate(forced_input.input_token_ids, decoding)
        extraction_payload, evaluation_payload = _answer_records(
            record.reference_answer,
            generation.generated_text,
            context.evaluator,
        )
        transition = _transition_from_payloads(evaluation_payload, final_evaluation)
        receipt = {
            **_checkpoint_receipt_identity(
                context=context,
                record=record,
                rollout_seed=rollout_seed,
                rollout_identity=rollout_identity,
                child_identity=child_identity,
                checkpoint_token=checkpoint.token_position,
            ),
            "status": "COMPLETED",
            "availability_status": CheckpointStatus.AVAILABLE.value,
            "saved_prefix_token_count": checkpoint.token_position,
            "saved_prefix_sha256": forced_input.saved_prefix_sha256,
            "forced_answer_input": {
                "prompt_token_count": len(forced_input.prompt_token_ids),
                "prompt_sha256": forced_input.prompt_sha256,
                "close_think_cue_token_ids": list(forced_input.close_think_cue_token_ids),
                "close_think_cue_sha256": forced_input.cue_sha256,
                "input_token_count": len(forced_input.input_token_ids),
                "input_sha256": forced_input.input_sha256,
            },
            "forced_generation": generation.to_dict(include_input_token_ids=False),
            "forced_extraction": extraction_payload,
            "forced_evaluation": evaluation_payload,
            "final_evaluation_reference": {
                "status": final_evaluation.get("status"),
                "correct": final_evaluation.get("correct"),
            },
            "transition_label": transition,
            "timestamp_utc": utc_now_iso(),
        }
    except Exception as error:
        receipt = {
            **_checkpoint_receipt_identity(
                context=context,
                record=record,
                rollout_seed=rollout_seed,
                rollout_identity=rollout_identity,
                child_identity=child_identity,
                checkpoint_token=checkpoint.token_position,
            ),
            "status": "FAILED",
            "availability_status": CheckpointStatus.AVAILABLE.value,
            "saved_prefix_token_count": checkpoint.token_position,
            "saved_prefix_sha256": checkpoint.prefix_sha256,
            "requested_decoding": decoding.to_dict(),
            "failure": _failure_payload(error),
            "timestamp_utc": utc_now_iso(),
        }
        context.layout.write_checkpoint(child_identity, receipt)
        ledger.append(
            "CHECKPOINT_FAILED",
            intent_id=intent_id,
            payload={
                "checkpoint_id": child_identity,
                "receipt_path": context.layout.checkpoint_path(child_identity).name,
            },
        )
        return receipt
    context.layout.write_checkpoint(child_identity, receipt)
    ledger.append(
        "CHECKPOINT_COMPLETED",
        intent_id=intent_id,
        payload={
            "checkpoint_id": child_identity,
            "receipt_path": context.layout.checkpoint_path(child_identity).name,
        },
    )
    return receipt


def _write_unavailable_checkpoint(
    *,
    context: _RunContext,
    record: ProblemRecord,
    rollout_seed: int,
    rollout_identity: str,
    checkpoint_token: int,
    availability_status: str,
    reason: str | None,
    base_generated_token_count: int | None = None,
) -> dict[str, Any]:
    child_identity = checkpoint_id(rollout_identity, checkpoint_token)
    existing = context.layout.read_checkpoint(child_identity)
    if existing is not None:
        return existing
    receipt = {
        **_checkpoint_receipt_identity(
            context=context,
            record=record,
            rollout_seed=rollout_seed,
            rollout_identity=rollout_identity,
            child_identity=child_identity,
            checkpoint_token=checkpoint_token,
        ),
        "status": "UNAVAILABLE",
        "availability_status": availability_status,
        "reason": reason,
        "base_generated_token_count": base_generated_token_count,
        "timestamp_utc": utc_now_iso(),
    }
    context.layout.write_checkpoint(child_identity, receipt)
    return receipt


def _recover_request(
    *,
    ledger: AppendOnlyLedger,
    intent_id: str,
    reader: Any,
    completed_event: str,
    failed_event: str,
) -> tuple[IntentState, dict[str, Any] | None]:
    """Recover a crash boundary without silently issuing the request again."""

    state = ledger.state(intent_id)
    receipt = reader()
    if state is IntentState.NOT_STARTED:
        if receipt is not None:
            raise RuntimeError(
                f"receipt exists without a recorded request intent: {intent_id}"
            )
        return state, None
    if state is IntentState.STARTED_UNKNOWN:
        if receipt is None:
            return ledger.preserve_interrupted_unknown(intent_id), None
        status = receipt.get("status")
        if status == "COMPLETED":
            ledger.append(completed_event, intent_id=intent_id, payload={"recovered": True})
            return IntentState.COMPLETED, receipt
        if status == "FAILED":
            ledger.append(failed_event, intent_id=intent_id, payload={"recovered": True})
            return IntentState.FAILED, receipt
        if status == "INTERRUPTED_UNKNOWN":
            ledger.append("INTERRUPTED_UNKNOWN", intent_id=intent_id, payload={"recovered": True})
            return IntentState.INTERRUPTED_UNKNOWN, receipt
        raise RuntimeError(
            f"started request has an unsupported receipt status {status!r}: {intent_id}"
        )
    if state in {IntentState.COMPLETED, IntentState.FAILED} and receipt is None:
        raise RuntimeError(f"terminal request lacks immutable receipt: {intent_id}")
    return state, receipt


def _base_receipt_identity(
    *,
    context: _RunContext,
    record: ProblemRecord,
    rollout_seed: int,
    rollout_identity: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "BASE_ROLLOUT",
        "run_id": context.layout.run_id,
        "manifest_hash": context.layout.manifest_hash,
        "config_hash": context.layout.config_hash,
        "runtime_provenance_sha256": context.runtime_provenance_hash,
        "source_git_commit": context.source_git_commit,
        "rollout_id": rollout_identity,
        "problem_id": record.problem_id,
        "source_index": record.source_index,
        "problem": record.problem,
        "problem_sha256": record.problem_sha256,
        "reference_answer": record.reference_answer,
        "reference_answer_sha256": record.reference_answer_sha256,
        "metadata": dict(record.metadata),
        "metadata_sha256": record.metadata_sha256,
        "rollout_seed": rollout_seed,
    }


def _checkpoint_receipt_identity(
    *,
    context: _RunContext,
    record: ProblemRecord,
    rollout_seed: int,
    rollout_identity: str,
    child_identity: str,
    checkpoint_token: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "FORCED_CHECKPOINT",
        "run_id": context.layout.run_id,
        "manifest_hash": context.layout.manifest_hash,
        "config_hash": context.layout.config_hash,
        "runtime_provenance_sha256": context.runtime_provenance_hash,
        "source_git_commit": context.source_git_commit,
        "checkpoint_id": child_identity,
        "parent_rollout_id": rollout_identity,
        "problem_id": record.problem_id,
        "source_index": record.source_index,
        "rollout_seed": rollout_seed,
        "checkpoint_token": checkpoint_token,
    }


def _interrupted_base_receipt(
    *,
    context: _RunContext,
    record: ProblemRecord,
    rollout_seed: int,
    rollout_identity: str,
) -> dict[str, object]:
    return {
        **_base_receipt_identity(
            context=context,
            record=record,
            rollout_seed=rollout_seed,
            rollout_identity=rollout_identity,
        ),
        "status": "INTERRUPTED_UNKNOWN",
        "failure": {
            "error_type": "INTERRUPTED_UNKNOWN",
            "message": "prior invocation ended after intent start without a receipt",
            "cuda_oom": False,
            "elapsed_seconds": None,
            "memory_before": None,
            "memory_after": None,
        },
        "timestamp_utc": utc_now_iso(),
    }


def _interrupted_checkpoint_receipt(
    *,
    context: _RunContext,
    record: ProblemRecord,
    rollout_seed: int,
    rollout_identity: str,
    child_identity: str,
    checkpoint_token: int,
    prefix_sha256: str | None,
) -> dict[str, object]:
    return {
        **_checkpoint_receipt_identity(
            context=context,
            record=record,
            rollout_seed=rollout_seed,
            rollout_identity=rollout_identity,
            child_identity=child_identity,
            checkpoint_token=checkpoint_token,
        ),
        "status": "INTERRUPTED_UNKNOWN",
        "availability_status": CheckpointStatus.AVAILABLE.value,
        "saved_prefix_token_count": checkpoint_token,
        "saved_prefix_sha256": prefix_sha256,
        "failure": {
            "error_type": "INTERRUPTED_UNKNOWN",
            "message": "prior invocation ended after intent start without a receipt",
            "cuda_oom": False,
            "elapsed_seconds": None,
            "memory_before": None,
            "memory_after": None,
        },
        "timestamp_utc": utc_now_iso(),
    }


def _prompt_payload(prompt: TokenizedPrompt) -> dict[str, object]:
    return {
        "user_prompt": prompt.user_prompt,
        "chat_prompt_text": prompt.chat_prompt_text,
        "token_ids": list(prompt.token_ids),
        "token_ids_sha256": prompt.token_ids_sha256,
    }


def _thinking_boundary(
    generated_ids: Sequence[int], marker_ids: Sequence[int]
) -> dict[str, object]:
    starts = _all_subsequence_starts(generated_ids, marker_ids)
    return {
        "coordinate": "raw generated token IDs after the native chat-template prompt; includes emitted <think> opening token when present",
        "close_think_marker_token_ids": list(marker_ids),
        "close_think_marker_sha256": sha256_token_ids(tuple(marker_ids)),
        "first_close_marker_start_zero_based": starts[0] if starts else None,
        "close_marker_count": len(starts),
        "status": (
            "NO_CLOSE_MARKER"
            if not starts
            else "ONE_CLOSE_MARKER"
            if len(starts) == 1
            else "MULTIPLE_CLOSE_MARKERS"
        ),
    }


def _all_subsequence_starts(
    tokens: Sequence[int], target: Sequence[int]
) -> tuple[int, ...]:
    start = 0
    starts: list[int] = []
    values = tuple(int(token) for token in tokens)
    needle = tuple(int(token) for token in target)
    while start <= len(values) - len(needle):
        relative = find_subsequence(values[start:], needle)
        if relative is None:
            break
        found = start + relative
        starts.append(found)
        start = found + 1
    return tuple(starts)


def _answer_records(
    reference_answer: str, generated_text: str, evaluator: EvaluationBackend
) -> tuple[dict[str, object], dict[str, object]]:
    """Keep extraction/evaluation faults separate from a completed generation."""

    try:
        extraction = extract_last_balanced_boxed(generated_text)
    except Exception as error:
        return (
            {
                "status": "ERROR",
                "extracted_text": None,
                "start_offset": None,
                "end_offset": None,
                "reason": "answer extraction raised an unexpected error",
                "error_type": type(error).__name__,
            },
            {
                "status": EvaluationStatus.ERROR.value,
                "correct": None,
                "backend_name": getattr(evaluator, "name", None),
                "backend_version": getattr(evaluator, "version", None),
                "reason": "answer extraction failed",
                "error_type": type(error).__name__,
                "parsed_reference_count": None,
                "parsed_candidate_count": None,
            },
        )
    try:
        evaluation = evaluate_extraction(reference_answer, extraction, backend=evaluator)
    except Exception as error:
        return (
            extraction.to_dict(),
            {
                "status": EvaluationStatus.ERROR.value,
                "correct": None,
                "backend_name": getattr(evaluator, "name", None),
                "backend_version": getattr(evaluator, "version", None),
                "reason": "answer evaluation raised an unexpected error",
                "error_type": type(error).__name__,
                "parsed_reference_count": None,
                "parsed_candidate_count": None,
            },
        )
    return extraction.to_dict(), evaluation.to_dict()


def _transition_from_payloads(
    checkpoint: Mapping[str, Any], final: Mapping[str, Any]
) -> str | None:
    valid = {EvaluationStatus.CORRECT.value, EvaluationStatus.INCORRECT.value}
    checkpoint_status = checkpoint.get("status")
    final_status = final.get("status")
    if checkpoint_status not in valid or final_status not in valid:
        return None
    checkpoint_correct = checkpoint.get("correct")
    final_correct = final.get("correct")
    if type(checkpoint_correct) is not bool or type(final_correct) is not bool:
        raise RuntimeError("evaluated receipt has a non-Boolean correctness value")
    return {
        (False, True): "W_TO_C",
        (False, False): "W_TO_W",
        (True, True): "C_TO_C",
        (True, False): "C_TO_W",
    }[(checkpoint_correct, final_correct)]


def _failure_payload(error: Exception) -> dict[str, object]:
    memory_before = getattr(error, "memory_before", None)
    memory_after = getattr(error, "memory_after", None)
    return {
        "error_type": getattr(error, "error_type", type(error).__name__),
        "message": str(error)[:1000],
        "cuda_oom": bool(getattr(error, "cuda_oom", False)),
        "elapsed_seconds": getattr(error, "elapsed_seconds", None),
        "memory_before": memory_before.to_dict() if memory_before is not None else None,
        "memory_after": memory_after.to_dict() if memory_after is not None else None,
    }


def _token_ids(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    try:
        converted = tuple(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a sequence of integer IDs") from error
    if not converted:
        raise ValueError(f"{label} cannot be empty")
    return converted


def _receipt_token_ids(values: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(values, list):
        raise RuntimeError(f"{label} are absent or malformed in saved receipt")
    return _token_ids(values, label=label)
