"""Operational and descriptive P0 summaries; no predictive analysis lives here."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from .analysis import analyze_p0_run
from .artifacts import RunLayout, read_json
from .evaluation import EvaluationStatus, TransitionLabel


def summarize_run(
    layout: RunLayout,
    *,
    planned_problem_ids: Iterable[str],
    planned_rollout_ids: Iterable[str],
    checkpoint_positions: Iterable[int],
) -> dict[str, object]:
    """Aggregate only recorded raw receipts into the requested P0 operations report."""

    planned_problems = tuple(planned_problem_ids)
    planned_rollouts = tuple(planned_rollout_ids)
    planned_positions = tuple(checkpoint_positions)
    rollouts = _read_records(layout.run_root / "raw" / "rollouts")
    checkpoints = _read_records(layout.run_root / "raw" / "checkpoints")

    trajectory_status = Counter(str(record.get("status", "UNKNOWN")) for record in rollouts)
    checkpoint_receipt_status = Counter(
        str(record.get("status", "UNKNOWN")) for record in checkpoints
    )
    completed_rollout_ids = {
        record.get("rollout_id")
        for record in rollouts
        if record.get("status") == "COMPLETED"
    }
    completed_problem_ids = {
        record.get("problem_id")
        for record in rollouts
        if record.get("status") == "COMPLETED"
    }
    runtime_values = [
        _number_at(record, ("generation", "elapsed_seconds"))
        for record in rollouts
        if record.get("status") == "COMPLETED"
    ]
    runtime_values = [value for value in runtime_values if value is not None]

    base_tokens = sum(
        _integer_at(record, ("generation", "generated_token_count")) or 0
        for record in rollouts
    )
    forced_tokens = sum(
        _integer_at(record, ("forced_generation", "generated_token_count")) or 0
        for record in checkpoints
    )
    transition_counts = Counter(
        str(record.get("transition_label"))
        for record in checkpoints
        if record.get("transition_label") in {label.value for label in TransitionLabel}
    )
    final_evaluation_counts = Counter(
        _nested_status(record, "final_evaluation") for record in rollouts
    )
    checkpoint_evaluation_counts = Counter(
        _nested_status(record, "forced_evaluation") for record in checkpoints
    )
    failure_counts = Counter(
        _failure_kind(record) for record in (*rollouts, *checkpoints)
        if record.get("status") == "FAILED"
    )

    availability = _checkpoint_availability(checkpoints, planned_positions)
    peak_values = _peak_memory_values((*rollouts, *checkpoints))

    derived_analysis = analyze_p0_run(
        layout,
        planned_problem_ids=planned_problems,
        planned_rollout_ids=planned_rollouts,
        checkpoint_positions=planned_positions,
    )

    return {
        "schema_version": 2,
        "run_id": layout.run_id,
        "manifest_hash": layout.manifest_hash,
        "config_hash": layout.config_hash,
        "scope": "P0 operational feasibility summary only; no activation or prediction metrics",
        "problems_attempted": len({record.get("problem_id") for record in rollouts}),
        "problems_planned": len(planned_problems),
        "problems_with_completed_trajectory": len(completed_problem_ids),
        "trajectories_planned": len(planned_rollouts),
        "trajectories_recorded": len(rollouts),
        "trajectories_completed": len(completed_rollout_ids),
        "trajectory_status_counts": dict(sorted(trajectory_status.items())),
        "checkpoint_receipt_status_counts": dict(
            sorted(checkpoint_receipt_status.items())
        ),
        "base_generated_tokens": base_tokens,
        "forced_generated_tokens": forced_tokens,
        "total_generated_tokens": base_tokens + forced_tokens,
        "checkpoint_counts_by_position": availability,
        "transition_counts": {
            label.value: transition_counts.get(label.value, 0)
            for label in TransitionLabel
        },
        "final_evaluation_status_counts": dict(sorted(final_evaluation_counts.items())),
        "checkpoint_evaluation_status_counts": dict(
            sorted(checkpoint_evaluation_counts.items())
        ),
        "non_evaluable_count": (
            final_evaluation_counts.get(EvaluationStatus.NON_EVALUABLE.value, 0)
            + checkpoint_evaluation_counts.get(EvaluationStatus.NON_EVALUABLE.value, 0)
        ),
        "evaluation_error_count": (
            final_evaluation_counts.get(EvaluationStatus.ERROR.value, 0)
            + checkpoint_evaluation_counts.get(EvaluationStatus.ERROR.value, 0)
        ),
        "failure_count": sum(failure_counts.values()),
        "failure_counts_by_kind": dict(sorted(failure_counts.items())),
        "interrupted_unknown_count": (
            trajectory_status.get("INTERRUPTED_UNKNOWN", 0)
            + checkpoint_receipt_status.get("INTERRUPTED_UNKNOWN", 0)
        ),
        "trajectory_runtime_seconds": {
            "mean": (sum(runtime_values) / len(runtime_values)) if runtime_values else None,
            "p50": median(runtime_values) if runtime_values else None,
        },
        "peak_vram_bytes": {
            "allocated": max(peak_values["allocated"], default=None),
            "reserved": max(peak_values["reserved"], default=None),
        },
        "artifact_disk_usage_bytes_before_summary": layout.disk_usage_bytes(),
        "derived_analysis": derived_analysis,
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    """Render the compact human-facing fields required for a P0 handoff."""

    transitions = summary.get("transition_counts", {})
    runtime = summary.get("trajectory_runtime_seconds", {})
    peak = summary.get("peak_vram_bytes", {})
    derived_analysis = summary.get("derived_analysis", {})
    final_accuracy = (
        derived_analysis.get("final_answer_accuracy", {})
        if isinstance(derived_analysis, Mapping)
        else {}
    )
    lengths = (
        derived_analysis.get("base_trajectory_length_tokens", {})
        if isinstance(derived_analysis, Mapping)
        else {}
    )
    lines = [
        f"problems attempted: {summary.get('problems_attempted')}",
        f"problems completed: {summary.get('problems_with_completed_trajectory')}",
        f"trajectories planned: {summary.get('trajectories_planned')}",
        f"trajectories completed: {summary.get('trajectories_completed')}",
        f"total generated tokens: {summary.get('total_generated_tokens')}",
        f"W_TO_C count: {transitions.get(TransitionLabel.W_TO_C.value, 0)}",
        f"W_TO_W count: {transitions.get(TransitionLabel.W_TO_W.value, 0)}",
        f"C_TO_C count: {transitions.get(TransitionLabel.C_TO_C.value, 0)}",
        f"C_TO_W count: {transitions.get(TransitionLabel.C_TO_W.value, 0)}",
        f"non-evaluable count: {summary.get('non_evaluable_count')}",
        f"failures: {summary.get('failure_count')}",
        f"interrupted unknown requests: {summary.get('interrupted_unknown_count')}",
        f"average trajectory runtime seconds: {runtime.get('mean')}",
        f"p50 trajectory runtime seconds: {runtime.get('p50')}",
        "final answer accuracy (eligible evaluable denominator): "
        f"{final_accuracy.get('correct_count')}/"
        f"{final_accuracy.get('evaluable_denominator')}",
        "base trajectory length tokens (p50): "
        f"{lengths.get('p50')}",
        f"peak allocated VRAM bytes: {peak.get('allocated')}",
        f"peak reserved VRAM bytes: {peak.get('reserved')}",
        "artifact disk usage bytes before writing this summary: "
        f"{summary.get('artifact_disk_usage_bytes_before_summary')}",
    ]
    return "\n".join(lines)


def _read_records(directory: Path) -> tuple[dict[str, Any], ...]:
    if not directory.exists():
        return ()
    return tuple(read_json(path) for path in sorted(directory.glob("*.json")))


def _checkpoint_availability(
    records: Iterable[Mapping[str, Any]], checkpoint_positions: Iterable[int]
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for position in checkpoint_positions:
        counts[str(position)]
    for record in records:
        position = record.get("checkpoint_token")
        if isinstance(position, int):
            position_counts = counts[str(position)]
            availability = str(record.get("availability_status", "UNKNOWN"))
            position_counts[availability] += 1
            position_counts[f"receipt_{record.get('status', 'UNKNOWN')}"] += 1
    return {
        position: dict(sorted(position_counts.items()))
        for position, position_counts in sorted(counts.items(), key=lambda item: int(item[0]))
    }


def _nested_status(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if isinstance(value, Mapping) and isinstance(value.get("status"), str):
        return value["status"]
    return "NOT_RECORDED"


def _failure_kind(record: Mapping[str, Any]) -> str:
    value = record.get("failure")
    if isinstance(value, Mapping) and isinstance(value.get("error_type"), str):
        return value["error_type"]
    return "UNKNOWN_FAILURE"


def _number_at(record: Mapping[str, Any], path: tuple[str, str]) -> float | None:
    container = record.get(path[0])
    value = container.get(path[1]) if isinstance(container, Mapping) else None
    return float(value) if isinstance(value, (int, float)) else None


def _integer_at(record: Mapping[str, Any], path: tuple[str, str]) -> int | None:
    container = record.get(path[0])
    value = container.get(path[1]) if isinstance(container, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _peak_memory_values(
    records: Iterable[Mapping[str, Any]]
) -> dict[str, list[int]]:
    values = {"allocated": [], "reserved": []}
    for record in records:
        for generation_key in ("generation", "forced_generation"):
            generation = record.get(generation_key)
            if not isinstance(generation, Mapping):
                continue
            memory = generation.get("memory_after")
            if not isinstance(memory, Mapping):
                continue
            allocated = memory.get("peak_allocated_bytes")
            reserved = memory.get("peak_reserved_bytes")
            if isinstance(allocated, int):
                values["allocated"].append(allocated)
            if isinstance(reserved, int):
                values["reserved"].append(reserved)
    return values
