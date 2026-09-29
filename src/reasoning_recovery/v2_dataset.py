"""DeepMath snapshot and deterministic, stratified V2 manifest selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from .artifacts import load_selection_manifest, write_selection_manifest
from .provenance import canonical_json, sha256_json, sha256_text
from .v2_config import V2Config, V2DatasetConfig


class V2DatasetContractError(RuntimeError):
    """Raised when DeepMath cannot satisfy the frozen V2 data contract."""


@dataclass(frozen=True)
class V2ProblemRecord:
    """One local DeepMath row with stable, non-outcome-derived identity."""

    problem_id: str
    source_index: int
    question: str
    reference_answer: str
    difficulty: float
    topic: str
    duplicate_cluster_sha256: str

    @property
    def question_sha256(self) -> str:
        return sha256_text(self.question)

    @property
    def reference_answer_sha256(self) -> str:
        return sha256_text(self.reference_answer)

    @property
    def metadata_sha256(self) -> str:
        return sha256_text(
            canonical_json({"difficulty": self.difficulty, "topic": self.topic})
        )


@dataclass(frozen=True)
class DeepMathSnapshot:
    """Pinned DeepMath schema and rows before V2 selection."""

    dataset_id: str
    requested_revision: str
    resolved_revision: str
    split: str
    observed_fingerprint: str | None
    records: tuple[V2ProblemRecord, ...]

    def selection_manifest(self, *, config: V2Config) -> dict[str, object]:
        """Select exactly the frozen 80/80 stratum quotas without outcomes."""

        if self.dataset_id != config.dataset.dataset_id:
            raise V2DatasetContractError("snapshot dataset ID differs from V2 config")
        if self.split != config.dataset.split:
            raise V2DatasetContractError("snapshot split differs from V2 config")
        if self.resolved_revision != config.dataset.revision:
            raise V2DatasetContractError(
                "snapshot revision differs from frozen V2 dataset revision"
            )
        selected: list[V2ProblemRecord] = []
        globally_selected_clusters: set[str] = set()
        stratum_summary: list[dict[str, object]] = []
        for stratum in config.dataset.strata:
            eligible = [
                record
                for record in self.records
                if _same_difficulty(record.difficulty, stratum.difficulty)
            ]
            ranked = sorted(
                eligible,
                key=lambda record: (
                    _selection_rank(
                        config.dataset.selection_seed,
                        stratum.difficulty,
                        record.problem_id,
                        record.source_index,
                    ),
                    record.source_index,
                ),
            )
            stratum_selected: list[V2ProblemRecord] = []
            skipped_duplicate_clusters = 0
            for record in ranked:
                if record.duplicate_cluster_sha256 in globally_selected_clusters:
                    skipped_duplicate_clusters += 1
                    continue
                globally_selected_clusters.add(record.duplicate_cluster_sha256)
                stratum_selected.append(record)
                if len(stratum_selected) == stratum.count:
                    break
            if len(stratum_selected) != stratum.count:
                raise V2DatasetContractError(
                    "frozen V2 selection could not obtain "
                    f"{stratum.count} unique rows for difficulty {stratum.difficulty}"
                )
            selected.extend(stratum_selected)
            stratum_summary.append(
                {
                    "difficulty": stratum.difficulty,
                    "requested_count": stratum.count,
                    "eligible_rows": len(eligible),
                    "skipped_duplicate_clusters": skipped_duplicate_clusters,
                    "selected_count": len(stratum_selected),
                }
            )

        records = [
            {
                "problem_id": record.problem_id,
                "source_index": record.source_index,
                "difficulty": record.difficulty,
                "topic": record.topic,
                "question_sha256": record.question_sha256,
                "reference_answer_sha256": record.reference_answer_sha256,
                "metadata_sha256": record.metadata_sha256,
                "duplicate_cluster_sha256": record.duplicate_cluster_sha256,
            }
            for record in selected
        ]
        if len({record["problem_id"] for record in records}) != len(records):
            raise V2DatasetContractError("V2 selection contains duplicate problem IDs")
        if len({record["duplicate_cluster_sha256"] for record in records}) != len(records):
            raise V2DatasetContractError(
                "V2 selection contains duplicate normalized-question clusters"
            )
        manifest = {
            "schema_version": 1,
            "record_type": "V2_SELECTION_MANIFEST",
            "study_protocol_id": config.study.protocol_id,
            "config_hash": config.config_hash,
            "dataset": {
                "id": self.dataset_id,
                "requested_revision": self.requested_revision,
                "resolved_revision": self.resolved_revision,
                "split": self.split,
                "observed_fingerprint": self.observed_fingerprint,
                "problem_field": config.dataset.problem_field,
                "answer_field": config.dataset.answer_field,
                "difficulty_field": config.dataset.difficulty_field,
                "topic_field": config.dataset.topic_field,
                "stable_id_policy": config.dataset.stable_id_policy,
                "duplicate_cluster_policy": config.dataset.duplicate_cluster_policy,
                "selection_method": config.dataset.selection_method,
                "selection_seed": config.dataset.selection_seed,
            },
            "strata": stratum_summary,
            "selection_size": len(records),
            "records": records,
        }
        _validate_manifest(config, manifest)
        return manifest


def load_deepmath_snapshot(
    config: V2DatasetConfig,
    *,
    revision_override: str | None = None,
) -> DeepMathSnapshot:
    """Resolve and load the frozen DeepMath revision without model inference."""

    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ModuleNotFoundError as error:
        raise V2DatasetContractError(
            "datasets and huggingface_hub are required to prepare a V2 manifest"
        ) from error
    requested = revision_override or config.revision
    try:
        info = HfApi().dataset_info(config.dataset_id, revision=requested)
    except Exception as error:
        raise V2DatasetContractError(
            f"could not resolve DeepMath revision: {type(error).__name__}"
        ) from error
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or not resolved:
        raise V2DatasetContractError("DeepMath Hub metadata did not provide a revision SHA")
    if resolved != config.revision:
        raise V2DatasetContractError(
            "DeepMath resolved revision differs from frozen V2 revision"
        )
    try:
        dataset = load_dataset(
            config.dataset_id,
            split=config.split,
            revision=resolved,
        )
    except Exception as error:
        raise V2DatasetContractError(
            f"could not load frozen DeepMath split: {type(error).__name__}: {str(error)[:300]}"
        ) from error

    column_names = set(getattr(dataset, "column_names", ()))
    expected = {
        config.problem_field,
        config.answer_field,
        config.difficulty_field,
        config.topic_field,
    }
    missing = sorted(expected - column_names)
    if missing:
        raise V2DatasetContractError(
            "DeepMath schema lacks frozen required field(s): " + ", ".join(missing)
        )

    records: list[V2ProblemRecord] = []
    for source_index, row in enumerate(dataset):
        if not isinstance(row, Mapping):
            raise V2DatasetContractError(f"DeepMath row {source_index} is not a mapping")
        question = _required_row_string(row, config.problem_field, source_index)
        reference_answer = _required_row_string(row, config.answer_field, source_index)
        topic = _required_row_string(row, config.topic_field, source_index)
        difficulty = _required_row_float(row, config.difficulty_field, source_index)
        question_hash = sha256_text(question)
        records.append(
            V2ProblemRecord(
                problem_id=f"deepmath:{source_index}:{question_hash[:16]}",
                source_index=source_index,
                question=question,
                reference_answer=reference_answer,
                difficulty=difficulty,
                topic=topic,
                duplicate_cluster_sha256=sha256_text(_normalise_question(question)),
            )
        )

    return DeepMathSnapshot(
        dataset_id=config.dataset_id,
        requested_revision=requested,
        resolved_revision=resolved,
        split=config.split,
        observed_fingerprint=_optional_string(getattr(dataset, "_fingerprint", None)),
        records=tuple(records),
    )


def write_v2_selection_manifest(path: Any, manifest: Mapping[str, object]) -> Any:
    """Write one immutable V2 selection manifest before model-backed requests."""

    return write_selection_manifest(path, manifest)


def load_v2_selection_manifest(path: Any, *, config: V2Config) -> dict[str, Any]:
    """Read and validate an immutable V2 selection manifest."""

    manifest = load_selection_manifest(path)
    _validate_manifest(config, manifest)
    return manifest


def select_v2_records_from_manifest(
    snapshot: DeepMathSnapshot, manifest: Mapping[str, object]
) -> tuple[V2ProblemRecord, ...]:
    """Rebind manifest identities to a reloaded frozen snapshot."""

    raw_records = manifest.get("records")
    if not isinstance(raw_records, list):
        raise V2DatasetContractError("V2 manifest records are malformed")
    by_id = {record.problem_id: record for record in snapshot.records}
    selected: list[V2ProblemRecord] = []
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise V2DatasetContractError("V2 manifest contains a non-object record")
        problem_id = raw.get("problem_id")
        if not isinstance(problem_id, str) or problem_id not in by_id:
            raise V2DatasetContractError("V2 manifest problem is absent from snapshot")
        record = by_id[problem_id]
        expected = {
            "problem_id": record.problem_id,
            "source_index": record.source_index,
            "difficulty": record.difficulty,
            "topic": record.topic,
            "question_sha256": record.question_sha256,
            "reference_answer_sha256": record.reference_answer_sha256,
            "metadata_sha256": record.metadata_sha256,
            "duplicate_cluster_sha256": record.duplicate_cluster_sha256,
        }
        if dict(raw) != expected:
            raise V2DatasetContractError(
                "V2 manifest record differs from reloaded frozen DeepMath snapshot"
            )
        selected.append(record)
    if len({record.problem_id for record in selected}) != len(selected):
        raise V2DatasetContractError("V2 manifest rebinding produced duplicate problems")
    return tuple(selected)


def derive_v2_rollout_seed(
    *,
    root_seed: int,
    problem_id: str,
    rollout_index: int,
) -> int:
    """Derive one stable nonzero 31-bit stochastic seed per V2 rollout."""

    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise V2DatasetContractError("V2 rollout seed root must be an integer")
    if not isinstance(problem_id, str) or not problem_id:
        raise V2DatasetContractError("V2 rollout seed needs a problem ID")
    if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
        raise V2DatasetContractError("V2 rollout index must be an integer")
    digest = sha256_text(f"{root_seed}\0{problem_id}\0{rollout_index}")
    return (int(digest[:16], 16) % 2_147_483_646) + 1


def _validate_manifest(config: V2Config, manifest: Mapping[str, object]) -> None:
    if manifest.get("schema_version") != 1:
        raise V2DatasetContractError("unsupported V2 manifest schema")
    if manifest.get("record_type") != "V2_SELECTION_MANIFEST":
        raise V2DatasetContractError("manifest is not a V2 selection manifest")
    if manifest.get("study_protocol_id") != config.study.protocol_id:
        raise V2DatasetContractError("manifest protocol ID differs from V2 config")
    if manifest.get("config_hash") != config.config_hash:
        raise V2DatasetContractError("manifest config hash differs from V2 config")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise V2DatasetContractError("V2 manifest lacks a dataset contract")
    expected_dataset = {
        "id": config.dataset.dataset_id,
        "requested_revision": config.dataset.revision,
        "resolved_revision": config.dataset.revision,
        "split": config.dataset.split,
        "problem_field": config.dataset.problem_field,
        "answer_field": config.dataset.answer_field,
        "difficulty_field": config.dataset.difficulty_field,
        "topic_field": config.dataset.topic_field,
        "stable_id_policy": config.dataset.stable_id_policy,
        "duplicate_cluster_policy": config.dataset.duplicate_cluster_policy,
        "selection_method": config.dataset.selection_method,
        "selection_seed": config.dataset.selection_seed,
    }
    for key, value in expected_dataset.items():
        if dataset.get(key) != value:
            raise V2DatasetContractError(
                f"V2 manifest dataset field differs from config: {key}"
            )
    if not isinstance(dataset.get("observed_fingerprint"), (str, type(None))):
        raise V2DatasetContractError("V2 manifest fingerprint has invalid type")

    records = manifest.get("records")
    if not isinstance(records, list):
        raise V2DatasetContractError("V2 manifest has no record list")
    expected_total = sum(stratum.count for stratum in config.dataset.strata)
    if len(records) != expected_total or manifest.get("selection_size") != expected_total:
        raise V2DatasetContractError("V2 manifest selection size differs from frozen quota")
    ids: set[str] = set()
    clusters: set[str] = set()
    counts = {stratum.difficulty: 0 for stratum in config.dataset.strata}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise V2DatasetContractError("V2 manifest contains a non-object record")
        problem_id = raw.get("problem_id")
        source_index = raw.get("source_index")
        difficulty = raw.get("difficulty")
        topic = raw.get("topic")
        cluster = raw.get("duplicate_cluster_sha256")
        if not isinstance(problem_id, str) or not problem_id:
            raise V2DatasetContractError("V2 manifest record has invalid problem ID")
        if type(source_index) is not int or source_index < 0:
            raise V2DatasetContractError("V2 manifest record has invalid source index")
        if not isinstance(difficulty, (int, float)) or isinstance(difficulty, bool):
            raise V2DatasetContractError("V2 manifest record has invalid difficulty")
        if not isinstance(topic, str) or not topic:
            raise V2DatasetContractError("V2 manifest record has invalid topic")
        if not isinstance(cluster, str) or len(cluster) != 64:
            raise V2DatasetContractError("V2 manifest record has invalid duplicate cluster")
        for hash_name in (
            "question_sha256",
            "reference_answer_sha256",
            "metadata_sha256",
        ):
            value = raw.get(hash_name)
            if not isinstance(value, str) or len(value) != 64:
                raise V2DatasetContractError(
                    f"V2 manifest record has invalid {hash_name}"
                )
        if problem_id in ids or cluster in clusters:
            raise V2DatasetContractError(
                "V2 manifest contains duplicate problem or normalized-question cluster"
            )
        ids.add(problem_id)
        clusters.add(cluster)
        matched = next(
            (
                stratum.difficulty
                for stratum in config.dataset.strata
                if _same_difficulty(float(difficulty), stratum.difficulty)
            ),
            None,
        )
        if matched is None:
            raise V2DatasetContractError("V2 manifest includes an unfrozen difficulty")
        counts[matched] += 1
    for stratum in config.dataset.strata:
        if counts[stratum.difficulty] != stratum.count:
            raise V2DatasetContractError(
                "V2 manifest difficulty quota differs from frozen contract"
            )


def _selection_rank(
    seed: int, difficulty: float, problem_id: str, source_index: int
) -> str:
    return sha256_text(f"{seed}\0{difficulty:.1f}\0{problem_id}\0{source_index}")


def _normalise_question(value: str) -> str:
    return " ".join(value.split())


def _same_difficulty(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0)


def _required_row_string(row: Mapping[str, object], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise V2DatasetContractError(
            f"DeepMath row {index} has missing/empty {field!r}"
        )
    return value


def _required_row_float(row: Mapping[str, object], field: str, index: int) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2DatasetContractError(f"DeepMath row {index} has invalid {field!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise V2DatasetContractError(f"DeepMath row {index} has non-finite {field!r}")
    return numeric


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
