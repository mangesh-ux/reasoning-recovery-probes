"""Synthetic-only hardware qualification for the frozen V2 workload.

No DeepMath row, rollout identity, evaluator call, scientific receipt, answer,
or activation artifact is used here.  This module exists so an 8 GiB local
machine can be judged against the protocol before it accumulates any V2 study
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from .provenance import sha256_json, sha256_token_ids, utc_now_iso
from .v2_activations import build_activation_input
from .v2_config import V2Config
from .v2_runtime import V2Runtime


class V2PreflightError(RuntimeError):
    """A synthetic qualification cannot be measured under its frozen contract."""


@dataclass(frozen=True)
class V2PreflightResult:
    """Aggregate-safe hardware result suitable for an immutable receipt."""

    payload: Mapping[str, object]

    @property
    def passed(self) -> bool:
        gate = self.payload.get("hardware_gate")
        return isinstance(gate, Mapping) and gate.get("passed") is True


def run_v2_synthetic_preflight(
    *, config: V2Config, runtime: V2Runtime, artifact_root: Path
) -> V2PreflightResult:
    """Measure the required synthetic operations and apply frozen hardware gates."""

    marker_ids, cue_ids = runtime.validate_readout_contract()
    prompt = runtime.tokenize_prompt(
        "Synthetic V2 hardware qualification only. Do not solve a benchmark problem."
    )
    if len(prompt.token_ids) >= min(config.hardware_preflight.activation_input_token_lengths):
        raise V2PreflightError("synthetic template unexpectedly exceeds the smallest activation shape")
    base_measurement = _measure_reasoning(
        runtime=runtime,
        input_ids=prompt.token_ids,
        marker_ids=marker_ids,
        config=config,
    )
    readout_measurement = _measure_readout(
        runtime=runtime,
        input_ids=tuple(prompt.token_ids) + tuple(cue_ids),
        config=config,
    )
    activation_measurements = {
        str(length): _measure_activation_length(
            runtime=runtime,
            prompt_token_ids=prompt.token_ids,
            total_length=length,
            config=config,
        )
        for length in config.hardware_preflight.activation_input_token_lengths
    }
    projection = _project_full_campaign(
        config=config,
        base=base_measurement,
        readout=readout_measurement,
        activations=activation_measurements,
    )
    storage = _storage_projection(config=config, activations=activation_measurements)
    disk = _disk_state(artifact_root)
    gate = _hardware_gate(
        config=config,
        runtime=runtime,
        base=base_measurement,
        readout=readout_measurement,
        activations=activation_measurements,
        projection=projection,
        storage=storage,
        disk=disk,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "V2_SYNTHETIC_HARDWARE_QUALIFICATION",
        "study_protocol_id": config.study.protocol_id,
        "config_hash": config.config_hash,
        "qualification_protocol_id": config.hardware_preflight.protocol_id,
        "synthetic_prompt": {
            "input_token_count": len(prompt.token_ids),
            "input_token_ids_sha256": prompt.token_ids_sha256,
            "synthetic_only": True,
        },
        "control_contract": {
            "close_think_marker_token_ids": list(marker_ids),
            "close_think_marker_sha256": sha256_token_ids(marker_ids),
            "readout_cue_token_ids": list(cue_ids),
            "readout_cue_sha256": sha256_token_ids(cue_ids),
        },
        "runtime": runtime.provenance,
        "reasoning_generation": base_measurement,
        "short_readout_generation": readout_measurement,
        "activation_forwards": activation_measurements,
        "projected_full_campaign": projection,
        "projected_storage": storage,
        "disk": disk,
        "hardware_gate": gate,
        "qualified_at_utc": utc_now_iso(),
    }
    return V2PreflightResult(payload=payload)


def _measure_reasoning(*, runtime: V2Runtime, input_ids: Sequence[int], marker_ids: Sequence[int], config: V2Config) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    total = config.hardware_preflight.warmup_iterations + config.hardware_preflight.measured_iterations
    for iteration in range(total):
        try:
            generated = runtime.generate_synthetic_reasoning(
                input_ids,
                seed=20261100 + iteration,
                close_think_token_ids=marker_ids,
                max_new_tokens=config.hardware_preflight.synthetic_generation_max_new_tokens,
            )
            runs.append({
                "status": "COMPLETED",
                "generated_token_count": len(generated.generated_token_ids),
                "elapsed_seconds": generated.elapsed_seconds,
                "termination_status": generated.termination_status,
                "memory_after": generated.memory_after.to_dict(),
            })
        except Exception as error:
            runs.append({"status": "FAILED", "error_type": type(error).__name__, "message": str(error)[:500], "cuda_oom": bool(getattr(error, "cuda_oom", False))})
    return _summarize_generation_runs(runs, warmups=config.hardware_preflight.warmup_iterations)


def _measure_readout(*, runtime: V2Runtime, input_ids: Sequence[int], config: V2Config) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    total = config.hardware_preflight.warmup_iterations + config.hardware_preflight.measured_iterations
    for _iteration in range(total):
        try:
            readout = runtime.generate_readout(input_ids)
            generation = readout.generation
            runs.append({
                "status": "COMPLETED",
                "generated_token_count": len(generation.generated_token_ids),
                "elapsed_seconds": generation.elapsed_seconds,
                "termination_status": generation.termination_status,
                "memory_after": generation.memory_after.to_dict(),
            })
        except Exception as error:
            runs.append({"status": "FAILED", "error_type": type(error).__name__, "message": str(error)[:500], "cuda_oom": bool(getattr(error, "cuda_oom", False))})
    return _summarize_generation_runs(runs, warmups=config.hardware_preflight.warmup_iterations)


def _measure_activation_length(*, runtime: V2Runtime, prompt_token_ids: Sequence[int], total_length: int, config: V2Config) -> dict[str, object]:
    if total_length <= len(prompt_token_ids):
        raise V2PreflightError("synthetic activation length must exceed the synthetic prompt")
    # Repeating one existing non-special prompt token gives a fixed valid token
    # sequence without fetching a benchmark row or using an answer token.
    filler = prompt_token_ids[-1]
    prefix = tuple(filler for _ in range(total_length - len(prompt_token_ids)))
    activation_input = build_activation_input(prompt_token_ids, prefix)
    runs: list[dict[str, object]] = []
    total = config.hardware_preflight.warmup_iterations + config.hardware_preflight.measured_iterations
    for _iteration in range(total):
        try:
            extraction = runtime.extract_checkpoint_activation(activation_input)
            runs.append({
                "status": "COMPLETED",
                "elapsed_seconds": extraction.elapsed_seconds,
                "matrix_shape": list(extraction.matrix_shape),
                "matrix_dtype": extraction.matrix_dtype,
                "raw_matrix_bytes": extraction.raw_matrix_bytes,
                "memory_after": extraction.memory_after.to_dict(),
            })
        except Exception as error:
            runs.append({"status": "FAILED", "error_type": type(error).__name__, "message": str(error)[:500], "cuda_oom": bool(getattr(error, "cuda_oom", False))})
    result = _summarize_activation_runs(runs, warmups=config.hardware_preflight.warmup_iterations)
    result.update({"synthetic_input_token_count": total_length, "synthetic_input_sha256": activation_input.input_sha256})
    return result


def _summarize_generation_runs(runs: Sequence[Mapping[str, object]], *, warmups: int) -> dict[str, object]:
    measured = list(runs[warmups:])
    successful = [entry for entry in measured if entry.get("status") == "COMPLETED"]
    if successful:
        total_tokens = sum(int(entry["generated_token_count"]) for entry in successful)
        total_seconds = sum(float(entry["elapsed_seconds"]) for entry in successful)
        rate = total_tokens / total_seconds if total_seconds > 0 else None
    else:
        total_tokens = 0
        total_seconds = 0.0
        rate = None
    return {"warmup_iterations": warmups, "measured_iterations": len(measured), "measured_completed": len(successful), "measured_failed": len(measured) - len(successful), "tokens": total_tokens, "elapsed_seconds": total_seconds, "tokens_per_second": rate, "runs": list(runs)}


def _summarize_activation_runs(runs: Sequence[Mapping[str, object]], *, warmups: int) -> dict[str, object]:
    measured = list(runs[warmups:])
    successful = [entry for entry in measured if entry.get("status") == "COMPLETED"]
    seconds = [float(entry["elapsed_seconds"]) for entry in successful]
    return {"warmup_iterations": warmups, "measured_iterations": len(measured), "measured_completed": len(successful), "measured_failed": len(measured) - len(successful), "elapsed_seconds": sum(seconds), "forwards_per_second": (len(seconds) / sum(seconds)) if sum(seconds) > 0 else None, "mean_seconds_per_forward": (sum(seconds) / len(seconds)) if seconds else None, "runs": list(runs)}


def _project_full_campaign(*, config: V2Config, base: Mapping[str, object], readout: Mapping[str, object], activations: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    base_rate = _positive_float(base.get("tokens_per_second"))
    readout_rate = _positive_float(readout.get("tokens_per_second"))
    largest_length = str(max(config.hardware_preflight.activation_input_token_lengths))
    activation_time = _positive_float(activations[largest_length].get("mean_seconds_per_forward"))
    base_seconds = (160 * 6 * config.generation.reasoning_max_new_tokens / base_rate) if base_rate else None
    readout_seconds = ((160 * 6 + 160 * 6 * len(config.generation.checkpoint_token_positions)) * config.readout.max_new_tokens / readout_rate) if readout_rate else None
    activation_seconds = (160 * 6 * len(config.generation.checkpoint_token_positions) * activation_time) if activation_time else None
    total = None if None in {base_seconds, readout_seconds, activation_seconds} else float(base_seconds + readout_seconds + activation_seconds)
    return {"upper_bound_base_generation_seconds": base_seconds, "upper_bound_readout_generation_seconds": readout_seconds, "upper_bound_activation_seconds": activation_seconds, "upper_bound_total_seconds": total, "upper_bound_total_hours": (total / 3600.0) if total is not None else None, "base_rate_tokens_per_second": base_rate, "readout_rate_tokens_per_second": readout_rate, "conservative_activation_seconds_per_forward": activation_time}


def _storage_projection(*, config: V2Config, activations: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    largest = activations[str(max(config.hardware_preflight.activation_input_token_lengths))]
    successful = [item for item in largest.get("runs", []) if isinstance(item, Mapping) and item.get("status") == "COMPLETED"]
    matrix_bytes = int(successful[0]["raw_matrix_bytes"]) if successful and isinstance(successful[0].get("raw_matrix_bytes"), int) else None
    activation_count = 160 * 6 * len(config.generation.checkpoint_token_positions)
    raw_total = matrix_bytes * activation_count if matrix_bytes is not None else None
    with_reserve = int(raw_total * 1.25) if raw_total is not None else None
    return {"activation_matrix_bytes": matrix_bytes, "maximum_checkpoint_activation_count": activation_count, "raw_activation_payload_bytes": raw_total, "raw_activation_payload_bytes_with_25_percent_reserve": with_reserve}


def _disk_state(artifact_root: Path) -> dict[str, object]:
    anchor = artifact_root.resolve()
    while not anchor.exists() and anchor.parent != anchor:
        anchor = anchor.parent
    usage = shutil.disk_usage(anchor)
    return {"filesystem_anchor": str(anchor), "free_bytes": int(usage.free), "total_bytes": int(usage.total)}


def _hardware_gate(*, config: V2Config, runtime: V2Runtime, base: Mapping[str, object], readout: Mapping[str, object], activations: Mapping[str, Mapping[str, object]], projection: Mapping[str, object], storage: Mapping[str, object], disk: Mapping[str, object]) -> dict[str, object]:
    reasons: list[str] = []
    mandatory = [base, readout, *activations.values()]
    for measurement in mandatory:
        if measurement.get("measured_failed") != 0 or measurement.get("measured_completed") != config.hardware_preflight.measured_iterations:
            reasons.append("one or more mandatory synthetic measurements failed")
            break
    runtime_gpu = runtime.provenance.get("gpu")
    total_memory = runtime_gpu.get("total_memory_bytes") if isinstance(runtime_gpu, Mapping) else None
    if not isinstance(total_memory, int) or total_memory <= 0:
        reasons.append("runtime did not report GPU total memory")
    peak_reserved = 0
    for measurement in mandatory:
        for entry in measurement.get("runs", []):
            if isinstance(entry, Mapping):
                memory = entry.get("memory_after")
                if isinstance(memory, Mapping) and isinstance(memory.get("peak_reserved_bytes"), int):
                    peak_reserved = max(peak_reserved, int(memory["peak_reserved_bytes"]))
    headroom = (total_memory - peak_reserved) if isinstance(total_memory, int) else None
    if headroom is None or headroom < config.runtime.minimum_activation_headroom_bytes:
        reasons.append("allocator headroom is below the frozen 1 GiB minimum")
    projected_hours = projection.get("upper_bound_total_hours")
    if not isinstance(projected_hours, (int, float)) or projected_hours > config.runtime.local_max_projected_wall_hours:
        reasons.append("projected full V2 wall time exceeds the local 48-hour limit")
    projected_storage = storage.get("raw_activation_payload_bytes_with_25_percent_reserve")
    free_disk = disk.get("free_bytes")
    if not isinstance(free_disk, int) or free_disk < config.runtime.minimum_free_disk_bytes:
        reasons.append("free disk is below the frozen minimum")
    if isinstance(projected_storage, int) and isinstance(free_disk, int) and projected_storage > free_disk:
        reasons.append("activation storage reserve does not fit on the artifact filesystem")
    return {"passed": not reasons, "reasons": reasons, "gpu_total_memory_bytes": total_memory, "maximum_observed_peak_reserved_bytes": peak_reserved, "minimum_observed_allocator_headroom_bytes": headroom, "required_allocator_headroom_bytes": config.runtime.minimum_activation_headroom_bytes, "projected_full_campaign_hours": projected_hours, "local_max_projected_wall_hours": config.runtime.local_max_projected_wall_hours}


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)
