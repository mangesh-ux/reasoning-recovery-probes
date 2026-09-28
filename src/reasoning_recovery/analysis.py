"""Aggregate-only, immutable-derived analysis for feasibility-only P0.

The functions here read local receipts but never issue a model, dataset, or
evaluator request.  They deliberately report descriptive accounting rather
than fitting a predictor or treating repeated rollouts as independent samples.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import RunLayout, read_json
from .evaluation import EvaluationStatus, TransitionLabel
from .provenance import sha256_json


ANALYSIS_SCHEMA_VERSION = 1
_VALID_TRANSITIONS = frozenset(label.value for label in TransitionLabel)


@dataclass(frozen=True)
class _Eligibility:
    """One receipt's explicit analytic-inclusion state.

    Older receipts have no eligibility field and remain eligible by default.
    A malformed field fails closed for analytic denominators while remaining
    visible in the receipt and eligibility counts.
    """

    eligible: bool
    status: str
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _TransitionObservation:
    """A transition that passed receipt, eligibility, and integrity checks."""

    label: str
    problem_id: str
    checkpoint_token: int


def analyze_p0_run(
    layout: RunLayout,
    *,
    planned_problem_ids: Iterable[str],
    planned_rollout_ids: Iterable[str],
    checkpoint_positions: Iterable[int],
) -> dict[str, object]:
    """Return a deterministic aggregate report from immutable P0 receipts.

    The report contains no prompt, completion, token ID, answer, or local path.
    Its input hashes bind the aggregate to the receipt contents that were read.
    Explicit ``analytic_eligibility`` values are honoured when present; legacy
    receipts without that field are retained as eligible and labelled as such.
    """

    planned_problems = _normalise_identifiers(planned_problem_ids, label="problem")
    planned_rollouts = _normalise_identifiers(planned_rollout_ids, label="rollout")
    positions = _normalise_positions(checkpoint_positions)
    rollouts = _read_receipts(layout.run_root / "raw" / "rollouts")
    checkpoints = _read_receipts(layout.run_root / "raw" / "checkpoints")
    runtime_contract = _read_optional_record(layout.run_root / "runtime_contract.json")

    base_by_rollout_id = _index_base_receipts(rollouts)
    base_eligibility = {
        rollout_id: _analytic_eligibility(record)
        for rollout_id, record in base_by_rollout_id.items()
    }

    checkpoint_accounting = _checkpoint_accounting(
        checkpoints,
        planned_rollout_ids=planned_rollouts,
        checkpoint_positions=positions,
    )
    base_stage = _stage_summary(
        rollouts,
        generation_key="generation",
        evaluation_key="final_evaluation",
    )
    forced_stage = _stage_summary(
        checkpoints,
        generation_key="forced_generation",
        evaluation_key="forced_evaluation",
    )
    final_accuracy = _final_accuracy(rollouts)
    primary_transitions, observed_transitions, transition_integrity = _collect_transitions(
        checkpoints,
        base_by_rollout_id=base_by_rollout_id,
        base_eligibility=base_eligibility,
        declared_positions=positions,
    )
    primary_rates = _transition_rates_by_checkpoint(primary_transitions, positions)
    primary_contrasts = _same_problem_checkpoint_contrasts(primary_transitions, positions)
    primary_concentration = _event_concentration(primary_transitions)
    observed_rates = _transition_rates_by_checkpoint(observed_transitions, positions)
    observed_contrasts = _same_problem_checkpoint_contrasts(observed_transitions, positions)
    observed_concentration = _event_concentration(observed_transitions)

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "record_type": "P0_DERIVED_ANALYSIS",
        "run_id": layout.run_id,
        "manifest_hash": layout.manifest_hash,
        "config_hash": layout.config_hash,
        "scope": (
            "P0 feasibility-only descriptive analysis. No activation analysis, "
            "predictive model, causal claim, or independent-rollout inference."
        ),
        "analysis_unit_note": (
            "Repeated stochastic rollouts are clustered within problem. "
            "Same-problem/same-checkpoint summaries are descriptive contrasts, "
            "not independent-sample estimates."
        ),
        "analysis_input": {
            "base_receipt_count": len(rollouts),
            "checkpoint_receipt_count": len(checkpoints),
            "base_receipts_sha256": sha256_json(rollouts),
            "checkpoint_receipts_sha256": sha256_json(checkpoints),
            "runtime_contract_sha256": (
                sha256_json(runtime_contract) if runtime_contract is not None else None
            ),
        },
        "planned_scope": {
            "planned_problem_count": len(planned_problems),
            "planned_rollout_count": len(planned_rollouts),
            "declared_checkpoint_positions": list(positions),
        },
        "final_answer_accuracy": final_accuracy,
        "base_trajectory_length_tokens": _length_distribution(
            _generated_token_counts(rollouts, generation_key="generation")
        ),
        "checkpoint_receipt_accounting": checkpoint_accounting,
        "stage_accounting": {
            "base": base_stage,
            "forced_checkpoint": forced_stage,
        },
        "vram_bytes": {
            "model_load": _model_load_memory(runtime_contract),
            "base_generation": base_stage["generation_vram_bytes"],
            "forced_checkpoint_generation": forced_stage["generation_vram_bytes"],
        },
        "analytic_eligibility": {
            "base_receipts": _eligibility_summary(
                _analytic_eligibility(record) for record in rollouts
            ),
            "checkpoint_receipts": _eligibility_summary(
                _analytic_eligibility(record) for record in checkpoints
            ),
            "effective_transition_receipts": transition_integrity[
                "effective_eligibility_counts"
            ],
        },
        "transitions": {
            "integrity_and_exclusion_counts": transition_integrity,
            "primary_analytic": {
                "rates_by_checkpoint": primary_rates,
                "same_problem_same_checkpoint_w_to_c_vs_w_to_w": primary_contrasts,
                "event_concentration": primary_concentration,
            },
            "observed_before_analytic_censoring": {
                "rates_by_checkpoint": observed_rates,
                "same_problem_same_checkpoint_w_to_c_vs_w_to_w": observed_contrasts,
                "event_concentration": observed_concentration,
            },
            "rates_by_checkpoint": primary_rates,
            "same_problem_same_checkpoint_w_to_c_vs_w_to_w": primary_contrasts,
            "event_concentration": primary_concentration,
        },
    }


def write_p0_analysis_report(
    layout: RunLayout,
    *,
    planned_problem_ids: Iterable[str],
    planned_rollout_ids: Iterable[str],
    checkpoint_positions: Iterable[int],
) -> Path:
    """Write the aggregate as a content-addressed immutable run summary."""

    report = analyze_p0_run(
        layout,
        planned_problem_ids=planned_problem_ids,
        planned_rollout_ids=planned_rollout_ids,
        checkpoint_positions=checkpoint_positions,
    )
    return layout.write_summary(report)


def format_p0_analysis(report: Mapping[str, Any]) -> str:
    """Render the principal P0 feasibility findings without raw artifacts."""

    accuracy = _mapping(report.get("final_answer_accuracy"))
    lengths = _mapping(report.get("base_trajectory_length_tokens"))
    transitions = _mapping(report.get("transitions"))
    by_checkpoint = _mapping(transitions.get("rates_by_checkpoint"))
    lines = [
        "P0 derived analysis (feasibility-only; descriptive)",
        "final-answer accuracy: "
        f"{accuracy.get('correct_count')}/{accuracy.get('evaluable_denominator')} "
        f"({accuracy.get('accuracy')})",
        "base trajectory lengths: "
        f"n={lengths.get('count')}, p50={lengths.get('p50')}, "
        f"min={lengths.get('min')}, max={lengths.get('max')}",
    ]
    for position, values in sorted(by_checkpoint.items(), key=lambda item: int(item[0])):
        checkpoint = _mapping(values)
        transitions_at_position = _mapping(checkpoint.get("transition_counts"))
        lines.append(
            f"checkpoint {position}: evaluable transitions="
            f"{checkpoint.get('evaluable_transition_denominator')}, "
            f"W_TO_C={transitions_at_position.get(TransitionLabel.W_TO_C.value, 0)}, "
            f"W_TO_W={transitions_at_position.get(TransitionLabel.W_TO_W.value, 0)}"
        )
    return "\n".join(lines)


def _read_receipts(directory: Path) -> tuple[dict[str, Any], ...]:
    if not directory.exists():
        return ()
    return tuple(read_json(path) for path in sorted(directory.glob("*.json")))


def _read_optional_record(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def _normalise_identifiers(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    supplied = tuple(values)
    normalised = tuple(value for value in supplied if isinstance(value, str) and value)
    if len(normalised) != len(supplied):
        raise ValueError(f"planned {label} IDs must be non-empty strings")
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"planned {label} IDs must be unique")
    return normalised


def _normalise_positions(values: Iterable[int]) -> tuple[int, ...]:
    positions = tuple(values)
    if not positions:
        raise ValueError("at least one declared checkpoint position is required")
    if any(type(position) is not int or position <= 0 for position in positions):
        raise ValueError("checkpoint positions must be positive integers")
    if len(set(positions)) != len(positions):
        raise ValueError("checkpoint positions must be unique")
    return tuple(sorted(positions))


def _index_base_receipts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Index only unambiguous base records; malformed duplicates stay visible later."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        rollout_id = record.get("rollout_id")
        if isinstance(rollout_id, str) and rollout_id:
            grouped[rollout_id].append(record)
    return {
        rollout_id: values[0]
        for rollout_id, values in grouped.items()
        if len(values) == 1
    }


def _checkpoint_accounting(
    records: Sequence[Mapping[str, Any]],
    *,
    planned_rollout_ids: Sequence[str],
    checkpoint_positions: Sequence[int],
) -> dict[str, object]:
    expected = {
        (rollout_id, position)
        for rollout_id in planned_rollout_ids
        for position in checkpoint_positions
    }
    observed: Counter[tuple[str, int]] = Counter()
    unexpected = 0
    malformed_identity = 0
    by_position: dict[int, Counter[str]] = {position: Counter() for position in checkpoint_positions}

    for record in records:
        parent = record.get("parent_rollout_id")
        position = record.get("checkpoint_token")
        valid_identity = isinstance(parent, str) and type(position) is int
        if not valid_identity:
            malformed_identity += 1
            continue
        key = (parent, position)
        observed[key] += 1
        if key not in expected:
            unexpected += 1
            continue
        counts = by_position[position]
        counts["recorded_receipts"] += 1
        counts[f"receipt_status:{record.get('status', 'UNKNOWN')}"] += 1
        counts[f"availability_status:{record.get('availability_status', 'UNKNOWN')}"] += 1

    duplicate_logical_receipts = sum(count - 1 for count in observed.values() if count > 1)
    output_by_position: dict[str, dict[str, int]] = {}
    for position in checkpoint_positions:
        counts = by_position[position]
        recorded_unique = sum(
            1
            for rollout_id in planned_rollout_ids
            if observed.get((rollout_id, position), 0) > 0
        )
        counts["expected_receipts"] = len(planned_rollout_ids)
        counts["recorded_unique_expected_receipts"] = recorded_unique
        counts["missing_expected_receipts"] = len(planned_rollout_ids) - recorded_unique
        output_by_position[str(position)] = dict(sorted(counts.items()))

    observed_expected = {key for key in observed if key in expected}
    return {
        "expected_receipt_count": len(expected),
        "recorded_receipt_count": len(records),
        "recorded_unique_expected_receipt_count": len(observed_expected),
        "missing_expected_receipt_count": len(expected - observed_expected),
        "unexpected_receipt_count": unexpected,
        "duplicate_logical_receipt_count": duplicate_logical_receipts,
        "malformed_identity_receipt_count": malformed_identity,
        "by_checkpoint_position": output_by_position,
    }


def _stage_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    generation_key: str,
    evaluation_key: str,
) -> dict[str, object]:
    receipt_statuses = Counter(str(record.get("status", "UNKNOWN")) for record in records)
    evaluations = [_evaluation_payload(record.get(evaluation_key)) for record in records]
    evaluation_statuses = Counter(value["status"] for value in evaluations)
    valid_evaluation_results = [
        value
        for value in evaluations
        if value["status"] in {EvaluationStatus.CORRECT.value, EvaluationStatus.INCORRECT.value}
        and type(value["correct"]) is bool
    ]
    malformed_evaluable_results = sum(
        1
        for value in evaluations
        if value["status"] in {EvaluationStatus.CORRECT.value, EvaluationStatus.INCORRECT.value}
        and type(value["correct"]) is not bool
    )
    failure_records = [
        record
        for record in records
        if record.get("status") in {"FAILED", "INTERRUPTED_UNKNOWN"}
    ]
    failure_types = Counter(
        _failure_type(record) for record in failure_records
    )
    explicit_cuda_oom = sum(_failure_is_explicit_cuda_oom(record) for record in records)
    generations = [
        _generation_payload(record.get(generation_key))
        for record in records
        if record.get("status") == "COMPLETED"
    ]
    completed_without_generation = sum(value is None for value in generations)
    usable_generations = [value for value in generations if value is not None]
    termination_statuses = Counter(
        str(value.get("termination_status", "NOT_RECORDED"))
        for value in usable_generations
    )
    return {
        "receipt_status_counts": dict(sorted(receipt_statuses.items())),
        "evaluation": {
            "status_counts": dict(sorted(evaluation_statuses.items())),
            "evaluable_count": len(valid_evaluation_results),
            "correct_count": sum(value["correct"] is True for value in valid_evaluation_results),
            "incorrect_count": sum(value["correct"] is False for value in valid_evaluation_results),
            "non_evaluable_count": evaluation_statuses.get(
                EvaluationStatus.NON_EVALUABLE.value, 0
            ),
            "error_count": evaluation_statuses.get(EvaluationStatus.ERROR.value, 0),
            "malformed_evaluable_result_count": malformed_evaluable_results,
        },
        "failure": {
            "failed_or_interrupted_receipt_count": len(failure_records),
            "failure_type_counts": dict(sorted(failure_types.items())),
            "explicit_cuda_oom_count": explicit_cuda_oom,
        },
        "generation": _generation_performance(usable_generations),
        "generation_termination_status_counts": dict(sorted(termination_statuses.items())),
        "generation_cap_count": termination_statuses.get("MAX_NEW_TOKENS", 0),
        "completed_receipt_without_generation_count": completed_without_generation,
        "generation_vram_bytes": _generation_vram(usable_generations),
    }


def _evaluation_payload(value: object) -> dict[str, object]:
    payload = _mapping(value)
    status = payload.get("status")
    return {
        "status": str(status) if isinstance(status, str) else "NOT_RECORDED",
        "correct": payload.get("correct"),
    }


def _failure_type(record: Mapping[str, Any]) -> str:
    failure = _mapping(record.get("failure"))
    value = failure.get("error_type")
    return str(value) if isinstance(value, str) and value else "NOT_RECORDED"


def _failure_is_explicit_cuda_oom(record: Mapping[str, Any]) -> bool:
    failure = _mapping(record.get("failure"))
    return failure.get("cuda_oom") is True


def _generation_payload(value: object) -> Mapping[str, Any] | None:
    payload = _mapping(value)
    return payload if payload else None


def _generation_performance(generations: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    token_values = [_nonnegative_int(value.get("generated_token_count")) for value in generations]
    elapsed_values = [_nonnegative_number(value.get("elapsed_seconds")) for value in generations]
    tokens = [value for value in token_values if value is not None]
    elapsed = [value for value in elapsed_values if value is not None]
    paired = [
        (token_count, elapsed_seconds)
        for token_count, elapsed_seconds in zip(token_values, elapsed_values)
        if token_count is not None and elapsed_seconds is not None
    ]
    per_request_tps = [
        token_count / elapsed_seconds
        for token_count, elapsed_seconds in paired
        if elapsed_seconds > 0
    ]
    paired_tokens = sum(token_count for token_count, _ in paired)
    paired_elapsed = sum(elapsed_seconds for _, elapsed_seconds in paired)
    return {
        "completed_generation_count": len(generations),
        "generated_token_total": sum(tokens),
        "token_count_recorded_generation_count": len(tokens),
        "elapsed_seconds_total": sum(elapsed),
        "elapsed_seconds_recorded_generation_count": len(elapsed),
        "runtime_seconds_distribution": _numeric_distribution(elapsed),
        "tokens_per_second": {
            "paired_token_and_runtime_generation_count": len(paired),
            "aggregate": (paired_tokens / paired_elapsed) if paired_elapsed > 0 else None,
            "per_request_distribution": _numeric_distribution(per_request_tps),
            "zero_elapsed_generation_count": sum(
                elapsed_seconds == 0 for _, elapsed_seconds in paired
            ),
        },
    }


def _generation_vram(generations: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    allocated: list[int] = []
    reserved: list[int] = []
    for generation in generations:
        memory_after = _mapping(generation.get("memory_after"))
        allocated_value = _nonnegative_int(memory_after.get("peak_allocated_bytes"))
        reserved_value = _nonnegative_int(memory_after.get("peak_reserved_bytes"))
        if allocated_value is not None:
            allocated.append(allocated_value)
        if reserved_value is not None:
            reserved.append(reserved_value)
    return {
        "peak_allocated_bytes": max(allocated, default=None),
        "peak_reserved_bytes": max(reserved, default=None),
        "allocated_measurement_count": len(allocated),
        "reserved_measurement_count": len(reserved),
    }


def _model_load_memory(runtime_contract: Mapping[str, Any] | None) -> dict[str, object]:
    runtime = _mapping(runtime_contract.get("runtime")) if runtime_contract else {}
    memory = _mapping(runtime.get("memory_after_load"))
    if not memory:
        return {"status": "NOT_RECORDED"}
    return {
        "status": "RECORDED",
        "allocated_bytes": _nonnegative_int(memory.get("allocated_bytes")),
        "reserved_bytes": _nonnegative_int(memory.get("reserved_bytes")),
        "peak_allocated_bytes": _nonnegative_int(memory.get("peak_allocated_bytes")),
        "peak_reserved_bytes": _nonnegative_int(memory.get("peak_reserved_bytes")),
    }


def _generated_token_counts(
    records: Sequence[Mapping[str, Any]], *, generation_key: str
) -> list[int]:
    counts: list[int] = []
    for record in records:
        if record.get("status") != "COMPLETED":
            continue
        generation = _mapping(record.get(generation_key))
        count = _nonnegative_int(generation.get("generated_token_count"))
        if count is not None:
            counts.append(count)
    return counts


def _length_distribution(values: Sequence[int]) -> dict[str, object]:
    return _numeric_distribution([float(value) for value in values], integer_values=True)


def _numeric_distribution(
    values: Sequence[float], *, integer_values: bool = False
) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "total": 0 if integer_values else 0.0,
            "mean": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
            "quantile_method": "nearest-rank",
        }
    ordered = sorted(values)

    def cast(value: float) -> int | float:
        return int(value) if integer_values else value

    return {
        "count": len(ordered),
        "total": cast(sum(ordered)),
        "mean": sum(ordered) / len(ordered),
        "min": cast(ordered[0]),
        "p25": cast(_nearest_rank(ordered, 0.25)),
        "p50": cast(_nearest_rank(ordered, 0.50)),
        "p75": cast(_nearest_rank(ordered, 0.75)),
        "p90": cast(_nearest_rank(ordered, 0.90)),
        "max": cast(ordered[-1]),
        "quantile_method": "nearest-rank",
    }


def _nearest_rank(sorted_values: Sequence[float], proportion: float) -> float:
    index = max(0, ceil(proportion * len(sorted_values)) - 1)
    return sorted_values[index]


def _final_accuracy(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    eligible_completed = [
        record
        for record in records
        if record.get("status") == "COMPLETED" and _analytic_eligibility(record).eligible
    ]
    excluded_completed = sum(
        record.get("status") == "COMPLETED" and not _analytic_eligibility(record).eligible
        for record in records
    )
    evaluations = [_evaluation_payload(record.get("final_evaluation")) for record in eligible_completed]
    valid = [
        result
        for result in evaluations
        if result["status"] in {EvaluationStatus.CORRECT.value, EvaluationStatus.INCORRECT.value}
        and type(result["correct"]) is bool
    ]
    correct_count = sum(result["correct"] is True for result in valid)
    denominator = len(valid)
    statuses = Counter(result["status"] for result in evaluations)
    return {
        "completed_base_receipt_count": sum(record.get("status") == "COMPLETED" for record in records),
        "analytic_eligibility_excluded_completed_base_count": excluded_completed,
        "analytic_eligible_completed_base_count": len(eligible_completed),
        "evaluable_denominator": denominator,
        "correct_count": correct_count,
        "incorrect_count": sum(result["correct"] is False for result in valid),
        "accuracy": (correct_count / denominator) if denominator else None,
        "evaluation_status_counts_among_eligible_completed": dict(sorted(statuses.items())),
        "non_evaluable_count_among_eligible_completed": statuses.get(
            EvaluationStatus.NON_EVALUABLE.value, 0
        ),
        "error_count_among_eligible_completed": statuses.get(EvaluationStatus.ERROR.value, 0),
        "malformed_evaluable_result_count_among_eligible_completed": sum(
            result["status"] in {EvaluationStatus.CORRECT.value, EvaluationStatus.INCORRECT.value}
            and type(result["correct"]) is not bool
            for result in evaluations
        ),
    }


def _analytic_eligibility(record: Mapping[str, Any]) -> _Eligibility:
    if "analytic_eligibility" not in record:
        return _Eligibility(eligible=True, status="NOT_DECLARED_LEGACY_ELIGIBLE")
    value = record.get("analytic_eligibility")
    if type(value) is bool:
        return _Eligibility(
            eligible=value,
            status="EXPLICIT_BOOLEAN_ELIGIBLE" if value else "EXPLICIT_BOOLEAN_EXCLUDED",
            reason_codes=() if value else ("EXPLICIT_BOOLEAN_EXCLUDED",),
        )
    if isinstance(value, Mapping):
        reason_codes = _eligibility_reason_codes(value)
        eligible = value.get("eligible")
        if type(eligible) is bool:
            return _Eligibility(
                eligible=eligible,
                status=(
                    "EXPLICIT_MAPPING_ELIGIBLE"
                    if eligible
                    else "EXPLICIT_MAPPING_EXCLUDED"
                ),
                reason_codes=reason_codes,
            )
        status = value.get("status")
        if status == "ELIGIBLE":
            return _Eligibility(
                eligible=True,
                status="EXPLICIT_MAPPING_STATUS_ELIGIBLE",
                reason_codes=reason_codes,
            )
        if status in {"EXCLUDED", "INELIGIBLE"}:
            return _Eligibility(
                eligible=False,
                status="EXPLICIT_MAPPING_STATUS_EXCLUDED",
                reason_codes=reason_codes,
            )
    return _Eligibility(
        eligible=False,
        status="MALFORMED_EXPLICIT_FIELD_EXCLUDED",
        reason_codes=("MALFORMED_EXPLICIT_FIELD",),
    )


def _eligibility_reason_codes(value: Mapping[str, Any]) -> tuple[str, ...]:
    raw_reasons = value.get("reasons")
    if not isinstance(raw_reasons, list):
        return ()
    codes = [
        reason.get("code")
        for reason in raw_reasons
        if isinstance(reason, Mapping)
        and isinstance(reason.get("code"), str)
        and reason["code"]
    ]
    return tuple(sorted(set(codes)))


def _eligibility_summary(values: Iterable[_Eligibility]) -> dict[str, object]:
    all_values = tuple(values)
    statuses = Counter(value.status for value in all_values)
    reason_codes = Counter(
        reason_code
        for value in all_values
        if not value.eligible
        for reason_code in value.reason_codes
    )
    return {
        "receipt_count": len(all_values),
        "eligible_count": sum(value.eligible for value in all_values),
        "excluded_count": sum(not value.eligible for value in all_values),
        "status_counts": dict(sorted(statuses.items())),
        "exclusion_reason_counts": dict(sorted(reason_codes.items())),
    }


def _collect_transitions(
    records: Sequence[Mapping[str, Any]],
    *,
    base_by_rollout_id: Mapping[str, Mapping[str, Any]],
    base_eligibility: Mapping[str, _Eligibility],
    declared_positions: Sequence[int],
) -> tuple[list[_TransitionObservation], list[_TransitionObservation], dict[str, object]]:
    primary_observations: list[_TransitionObservation] = []
    observed_observations: list[_TransitionObservation] = []
    counts: Counter[str] = Counter()
    effective_eligibility = Counter[str]()
    exclusion_reasons = Counter[str]()
    declared = set(declared_positions)
    for record in records:
        position = record.get("checkpoint_token")
        if type(position) is not int or position not in declared:
            counts["UNDECLARED_OR_MALFORMED_CHECKPOINT_POSITION"] += 1
            continue
        if record.get("status") != "COMPLETED":
            counts["CHECKPOINT_RECEIPT_NOT_COMPLETED"] += 1
            continue
        checkpoint_eligibility = _analytic_eligibility(record)
        parent_id = record.get("parent_rollout_id")
        if not isinstance(parent_id, str) or parent_id not in base_by_rollout_id:
            counts["MISSING_OR_AMBIGUOUS_PARENT_BASE_RECEIPT"] += 1
            effective_eligibility["MISSING_PARENT_BASE_EXCLUDED"] += 1
            continue
        parent_eligibility = base_eligibility[parent_id]
        problem_id = record.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id:
            counts["MALFORMED_PROBLEM_ID"] += 1
            continue
        observed_label, observed_label_status = _validated_transition_label(
            record,
            parent_base=base_by_rollout_id[parent_id],
            label_key="observed_transition_label",
            fallback_label_keys=("transition_label",),
        )
        if observed_label is None:
            counts[f"OBSERVED:{observed_label_status}"] += 1
        else:
            observed_observations.append(
                _TransitionObservation(
                    label=observed_label,
                    problem_id=problem_id,
                    checkpoint_token=position,
                )
            )
            counts["VALID_OBSERVED_TRANSITION"] += 1
        if not checkpoint_eligibility.eligible:
            counts["CHECKPOINT_ANALYTICALLY_EXCLUDED"] += 1
            effective_eligibility[checkpoint_eligibility.status] += 1
            exclusion_reasons.update(checkpoint_eligibility.reason_codes)
            continue
        if not parent_eligibility.eligible:
            counts["PARENT_BASE_ANALYTICALLY_EXCLUDED"] += 1
            effective_eligibility[parent_eligibility.status] += 1
            exclusion_reasons.update(parent_eligibility.reason_codes)
            continue
        effective_eligibility["EFFECTIVELY_ELIGIBLE"] += 1
        primary_label, primary_label_status = _validated_transition_label(
            record,
            parent_base=base_by_rollout_id[parent_id],
            label_key="transition_label",
        )
        if primary_label is None:
            counts[f"PRIMARY:{primary_label_status}"] += 1
            continue
        primary_observations.append(
            _TransitionObservation(
                label=primary_label,
                problem_id=problem_id,
                checkpoint_token=position,
            )
        )
        counts["VALID_PRIMARY_ANALYTIC_TRANSITION"] += 1
    return primary_observations, observed_observations, {
        **dict(sorted(counts.items())),
        "effective_eligibility_counts": dict(sorted(effective_eligibility.items())),
        "analytic_exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
    }


def _validated_transition_label(
    record: Mapping[str, Any],
    *,
    parent_base: Mapping[str, Any],
    label_key: str,
    fallback_label_keys: Sequence[str] = (),
) -> tuple[str | None, str]:
    recorded = record.get(label_key)
    source_key = label_key
    if recorded is None:
        for fallback_key in fallback_label_keys:
            fallback = record.get(fallback_key)
            if fallback is not None:
                recorded = fallback
                source_key = fallback_key
                break
    recorded_label = (
        recorded if isinstance(recorded, str) and recorded in _VALID_TRANSITIONS else None
    )
    forced = _evaluation_payload(record.get("forced_evaluation"))
    final_reference = _evaluation_payload(record.get("final_evaluation_reference"))
    if final_reference["status"] == "NOT_RECORDED":
        final_reference = _evaluation_payload(parent_base.get("final_evaluation"))
    derived = _derive_transition(forced, final_reference)
    if recorded_label is not None and derived is not None and recorded_label != derived:
        return None, f"{source_key.upper()}_AND_DERIVED_TRANSITION_MISMATCH"
    if recorded_label is not None:
        return recorded_label, (
            f"{source_key.upper()}_AND_DERIVED_TRANSITION"
            if derived is not None
            else f"{source_key.upper()}_TRANSITION_ONLY"
        )
    if derived is not None:
        return derived, "DERIVED_TRANSITION"
    if recorded is not None:
        return None, f"INVALID_{source_key.upper()}"
    return None, "NON_EVALUABLE_OR_ERROR_TRANSITION"


def _derive_transition(
    checkpoint: Mapping[str, object], final: Mapping[str, object]
) -> str | None:
    valid = {EvaluationStatus.CORRECT.value, EvaluationStatus.INCORRECT.value}
    if checkpoint.get("status") not in valid or final.get("status") not in valid:
        return None
    checkpoint_correct = checkpoint.get("correct")
    final_correct = final.get("correct")
    if type(checkpoint_correct) is not bool or type(final_correct) is not bool:
        return None
    return {
        (False, True): TransitionLabel.W_TO_C.value,
        (False, False): TransitionLabel.W_TO_W.value,
        (True, True): TransitionLabel.C_TO_C.value,
        (True, False): TransitionLabel.C_TO_W.value,
    }[(checkpoint_correct, final_correct)]


def _transition_rates_by_checkpoint(
    observations: Sequence[_TransitionObservation], positions: Sequence[int]
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for position in positions:
        labels = Counter(
            observation.label
            for observation in observations
            if observation.checkpoint_token == position
        )
        denominator = sum(labels.values())
        transition_counts = {
            label.value: labels.get(label.value, 0) for label in TransitionLabel
        }
        wrong_checkpoint_denominator = (
            transition_counts[TransitionLabel.W_TO_C.value]
            + transition_counts[TransitionLabel.W_TO_W.value]
        )
        output[str(position)] = {
            "evaluable_transition_denominator": denominator,
            "transition_counts": transition_counts,
            "transition_rates": {
                label: (count / denominator) if denominator else None
                for label, count in transition_counts.items()
            },
            "wrong_checkpoint_transition_denominator": wrong_checkpoint_denominator,
            "recovery_rate_given_wrong_checkpoint": (
                transition_counts[TransitionLabel.W_TO_C.value]
                / wrong_checkpoint_denominator
                if wrong_checkpoint_denominator
                else None
            ),
            "non_recovery_rate_given_wrong_checkpoint": (
                transition_counts[TransitionLabel.W_TO_W.value]
                / wrong_checkpoint_denominator
                if wrong_checkpoint_denominator
                else None
            ),
        }
    return output


def _same_problem_checkpoint_contrasts(
    observations: Sequence[_TransitionObservation], positions: Sequence[int]
) -> dict[str, object]:
    groups: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for observation in observations:
        if observation.label in {TransitionLabel.W_TO_C.value, TransitionLabel.W_TO_W.value}:
            groups[(observation.problem_id, observation.checkpoint_token)][observation.label] += 1
    by_position: dict[str, dict[str, int]] = {}
    for position in positions:
        counters = [counts for (_, group_position), counts in groups.items() if group_position == position]
        by_position[str(position)] = _contrast_summary(counters)
    return {
        "grouping": "problem_id plus checkpoint_token across rollout seeds",
        "all_declared_checkpoints": _contrast_summary(list(groups.values())),
        "by_checkpoint_position": by_position,
    }


def _contrast_summary(groups: Sequence[Counter[str]]) -> dict[str, int]:
    with_w_to_c = [group for group in groups if group.get(TransitionLabel.W_TO_C.value, 0) > 0]
    with_w_to_w = [group for group in groups if group.get(TransitionLabel.W_TO_W.value, 0) > 0]
    mixed = [
        group
        for group in groups
        if group.get(TransitionLabel.W_TO_C.value, 0) > 0
        and group.get(TransitionLabel.W_TO_W.value, 0) > 0
    ]
    only_w_to_c = [
        group
        for group in groups
        if group.get(TransitionLabel.W_TO_C.value, 0) > 0
        and group.get(TransitionLabel.W_TO_W.value, 0) == 0
    ]
    only_w_to_w = [
        group
        for group in groups
        if group.get(TransitionLabel.W_TO_W.value, 0) > 0
        and group.get(TransitionLabel.W_TO_C.value, 0) == 0
    ]
    return {
        "groups_with_any_wrong_checkpoint_transition": len(groups),
        "groups_with_at_least_one_w_to_c": len(with_w_to_c),
        "groups_with_at_least_one_w_to_w": len(with_w_to_w),
        "groups_with_both_w_to_c_and_w_to_w": len(mixed),
        "groups_only_w_to_c": len(only_w_to_c),
        "groups_only_w_to_w": len(only_w_to_w),
        "w_to_c_events_in_mixed_groups": sum(
            group.get(TransitionLabel.W_TO_C.value, 0) for group in mixed
        ),
        "w_to_w_events_in_mixed_groups": sum(
            group.get(TransitionLabel.W_TO_W.value, 0) for group in mixed
        ),
    }


def _event_concentration(
    observations: Sequence[_TransitionObservation],
) -> dict[str, dict[str, object]]:
    return {
        label.value: _concentration_for_label(
            [observation for observation in observations if observation.label == label.value]
        )
        for label in TransitionLabel
    }


def _concentration_for_label(
    observations: Sequence[_TransitionObservation],
) -> dict[str, object]:
    by_problem = Counter(observation.problem_id for observation in observations)
    by_problem_checkpoint = Counter(
        (observation.problem_id, observation.checkpoint_token) for observation in observations
    )
    by_checkpoint = Counter(observation.checkpoint_token for observation in observations)
    event_count = len(observations)
    return {
        "event_count": event_count,
        "distinct_problem_count": len(by_problem),
        "distinct_problem_checkpoint_group_count": len(by_problem_checkpoint),
        "distinct_checkpoint_position_count": len(by_checkpoint),
        "max_events_in_one_problem": max(by_problem.values(), default=0),
        "max_problem_event_share": _largest_share(by_problem, event_count),
        "max_events_in_one_problem_checkpoint_group": max(
            by_problem_checkpoint.values(), default=0
        ),
        "max_problem_checkpoint_group_event_share": _largest_share(
            by_problem_checkpoint, event_count
        ),
        "herfindahl_index_by_problem": _herfindahl(by_problem, event_count),
        "herfindahl_index_by_problem_checkpoint_group": _herfindahl(
            by_problem_checkpoint, event_count
        ),
        "event_counts_by_checkpoint_position": {
            str(position): count for position, count in sorted(by_checkpoint.items())
        },
    }


def _largest_share(counts: Counter[Any], total: int) -> float | None:
    return (max(counts.values()) / total) if total else None


def _herfindahl(counts: Counter[Any], total: int) -> float | None:
    return sum((count / total) ** 2 for count in counts.values()) if total else None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _nonnegative_number(value: object) -> float | None:
    return float(value) if type(value) in {int, float} and value >= 0 else None
