"""Deterministic grouped train/validation/test split for the V2 manifest."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

from .provenance import sha256_json, sha256_text
from .v2_config import V2Config


class V2SplitContractError(RuntimeError):
    """Raised when a grouped V2 split does not match the frozen contract."""


def build_v2_grouped_split(
    manifest: Mapping[str, object], *, config: V2Config
) -> dict[str, object]:
    """Freeze a 96/32/32 problem split, balanced by frozen difficulty strata."""

    _validate_manifest_compatibility(manifest, config)
    raw_records = manifest["records"]
    assert isinstance(raw_records, list)
    grouped: dict[float, list[Mapping[str, object]]] = defaultdict(list)
    for record in raw_records:
        assert isinstance(record, Mapping)
        difficulty = record.get("difficulty")
        if isinstance(difficulty, bool) or not isinstance(difficulty, (int, float)):
            raise V2SplitContractError("manifest record has invalid difficulty")
        grouped[float(difficulty)].append(record)

    assignments: list[dict[str, object]] = []
    selected_clusters: set[str] = set()
    for stratum in config.dataset.strata:
        records = grouped.get(stratum.difficulty, [])
        if len(records) != stratum.count:
            raise V2SplitContractError("manifest does not satisfy frozen stratum count")
        ranked = sorted(
            records,
            key=lambda record: (
                sha256_text(
                    f"{config.splits.split_seed}\0{stratum.difficulty:.1f}\0"
                    f"{record['duplicate_cluster_sha256']}\0{record['problem_id']}"
                ),
                int(record["source_index"]),
            ),
        )
        for index, record in enumerate(ranked):
            if index < config.splits.train_per_difficulty:
                split = "train"
            elif index < (
                config.splits.train_per_difficulty
                + config.splits.validation_per_difficulty
            ):
                split = "validation"
            else:
                split = "test"
            cluster = record["duplicate_cluster_sha256"]
            if cluster in selected_clusters:
                raise V2SplitContractError(
                    "duplicate normalized-question cluster would cross a V2 split"
                )
            selected_clusters.add(str(cluster))
            assignments.append(
                {
                    "problem_id": record["problem_id"],
                    "source_index": record["source_index"],
                    "difficulty": record["difficulty"],
                    "duplicate_cluster_sha256": cluster,
                    "split": split,
                }
            )

    split_manifest = {
        "schema_version": 1,
        "record_type": "V2_GROUPED_SPLIT_MANIFEST",
        "study_protocol_id": config.study.protocol_id,
        "config_hash": config.config_hash,
        "selection_manifest_sha256": sha256_json(manifest),
        "split_policy": config.splits.policy,
        "split_seed": config.splits.split_seed,
        "duplicate_clusters_must_not_cross_splits": (
            config.splits.duplicate_clusters_must_not_cross_splits
        ),
        "test_unblinding_policy": config.splits.test_unblinding_policy,
        "assignments": sorted(assignments, key=lambda record: str(record["problem_id"])),
    }
    validate_v2_grouped_split(split_manifest, manifest=manifest, config=config)
    return split_manifest


def validate_v2_grouped_split(
    split: Mapping[str, object], *, manifest: Mapping[str, object], config: V2Config
) -> None:
    """Fail closed unless every selected problem and cluster has one split."""

    if split.get("schema_version") != 1:
        raise V2SplitContractError("unsupported V2 split schema")
    if split.get("record_type") != "V2_GROUPED_SPLIT_MANIFEST":
        raise V2SplitContractError("not a V2 grouped split manifest")
    expected = {
        "study_protocol_id": config.study.protocol_id,
        "config_hash": config.config_hash,
        "selection_manifest_sha256": sha256_json(manifest),
        "split_policy": config.splits.policy,
        "split_seed": config.splits.split_seed,
        "duplicate_clusters_must_not_cross_splits": (
            config.splits.duplicate_clusters_must_not_cross_splits
        ),
        "test_unblinding_policy": config.splits.test_unblinding_policy,
    }
    for key, value in expected.items():
        if split.get(key) != value:
            raise V2SplitContractError(f"V2 split differs from frozen contract: {key}")
    assignments = split.get("assignments")
    if not isinstance(assignments, list):
        raise V2SplitContractError("V2 split has no assignment list")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise V2SplitContractError("V2 manifest has no records")
    expected_by_id = {
        str(record["problem_id"]): record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("problem_id"), str)
    }
    if len(expected_by_id) != len(records):
        raise V2SplitContractError("V2 manifest problem identity is invalid")

    actual_ids: set[str] = set()
    clusters: dict[str, str] = {}
    counts: Counter[tuple[str, float]] = Counter()
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise V2SplitContractError("V2 split contains a non-object assignment")
        problem_id = assignment.get("problem_id")
        split_name = assignment.get("split")
        if not isinstance(problem_id, str) or problem_id not in expected_by_id:
            raise V2SplitContractError("V2 split refers to an unknown problem")
        if split_name not in {"train", "validation", "test"}:
            raise V2SplitContractError("V2 split has an invalid split label")
        if problem_id in actual_ids:
            raise V2SplitContractError("V2 split assigns a problem more than once")
        expected_record = expected_by_id[problem_id]
        for key in (
            "source_index",
            "difficulty",
            "duplicate_cluster_sha256",
        ):
            if assignment.get(key) != expected_record.get(key):
                raise V2SplitContractError(
                    f"V2 split assignment differs from manifest for {key}"
                )
        actual_ids.add(problem_id)
        cluster = str(assignment["duplicate_cluster_sha256"])
        prior = clusters.setdefault(cluster, str(split_name))
        if prior != split_name:
            raise V2SplitContractError(
                "one normalized-question cluster appears in more than one V2 split"
            )
        counts[(str(split_name), float(assignment["difficulty"]))] += 1

    if actual_ids != set(expected_by_id):
        raise V2SplitContractError("V2 split does not assign every selected problem")
    for stratum in config.dataset.strata:
        expected_counts = {
            "train": config.splits.train_per_difficulty,
            "validation": config.splits.validation_per_difficulty,
            "test": config.splits.test_per_difficulty,
        }
        for split_name, count in expected_counts.items():
            if counts[(split_name, stratum.difficulty)] != count:
                raise V2SplitContractError(
                    "V2 split quota differs from frozen contract for "
                    f"{split_name}/difficulty-{stratum.difficulty}"
                )


def v2_problem_ids_for_split(
    split: Mapping[str, object], split_name: str
) -> tuple[str, ...]:
    """Return deterministic problem IDs for a frozen V2 split."""

    if split_name not in {"train", "validation", "test"}:
        raise V2SplitContractError("unknown V2 split name")
    assignments = split.get("assignments")
    if not isinstance(assignments, list):
        raise V2SplitContractError("V2 split assignments are unavailable")
    identifiers = [
        assignment.get("problem_id")
        for assignment in assignments
        if isinstance(assignment, Mapping) and assignment.get("split") == split_name
    ]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise V2SplitContractError("V2 split contains an invalid problem ID")
    return tuple(sorted(str(value) for value in identifiers))


def _validate_manifest_compatibility(manifest: Mapping[str, object], config: V2Config) -> None:
    if manifest.get("study_protocol_id") != config.study.protocol_id:
        raise V2SplitContractError("V2 manifest protocol does not match config")
    if manifest.get("config_hash") != config.config_hash:
        raise V2SplitContractError("V2 manifest config hash does not match config")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise V2SplitContractError("V2 manifest has no selected records")
