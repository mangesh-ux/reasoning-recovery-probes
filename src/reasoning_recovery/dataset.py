"""Dataset loading and deterministic pre-inference manifest selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import DatasetConfig, PilotConfig
from .provenance import canonical_json, sha256_text


class DatasetContractError(RuntimeError):
    """Raised when a dataset cannot satisfy the reviewed P0 schema."""


@dataclass(frozen=True)
class ProblemRecord:
    """The local fields P0 is allowed to carry into a raw trajectory receipt."""

    problem_id: str
    source_index: int
    problem: str
    reference_answer: str
    metadata: Mapping[str, str | int | None]

    @property
    def problem_sha256(self) -> str:
        return sha256_text(self.problem)

    @property
    def reference_answer_sha256(self) -> str:
        return sha256_text(self.reference_answer)

    @property
    def metadata_sha256(self) -> str:
        return sha256_text(canonical_json(self.metadata))


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset_id: str
    requested_revision: str | None
    split: str
    observed_fingerprint: str | None
    records: tuple[ProblemRecord, ...]

    def selection_manifest(
        self, *, config: PilotConfig, limit: int | None = None
    ) -> dict[str, object]:
        """Build a deterministic manifest without a model request."""

        requested_size = config.dataset.pilot_size if limit is None else limit
        if requested_size <= 0:
            raise DatasetContractError("manifest limit must be positive")
        if requested_size > len(self.records):
            raise DatasetContractError(
                f"requested {requested_size} rows but dataset has only {len(self.records)}"
            )

        ranked = sorted(
            self.records,
            key=lambda record: (
                _selection_rank(
                    config.dataset.selection_seed, record.problem_id, record.source_index
                ),
                record.source_index,
            ),
        )
        selected = ranked[:requested_size]
        return {
            "schema_version": 1,
            "study_protocol_id": config.study.protocol_id,
            "config_hash": config.config_hash,
            "dataset": {
                "id": self.dataset_id,
                "requested_revision": self.requested_revision,
                "split": self.split,
                "observed_fingerprint": self.observed_fingerprint,
                "selection_method": config.dataset.selection_method,
                "selection_seed": config.dataset.selection_seed,
            },
            "selection_size": requested_size,
            "records": [
                {
                    "problem_id": record.problem_id,
                    "source_index": record.source_index,
                    "selection_rank": _selection_rank(
                        config.dataset.selection_seed,
                        record.problem_id,
                        record.source_index,
                    ),
                    "problem_sha256": record.problem_sha256,
                    "reference_answer_sha256": record.reference_answer_sha256,
                    "metadata_sha256": record.metadata_sha256,
                }
                for record in selected
            ],
        }


def load_math500_snapshot(config: DatasetConfig) -> DatasetSnapshot:
    """Load the dataset only after an explicit manifest-preparation command."""

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "datasets is required to prepare a manifest; install .[runtime]"
        ) from error

    try:
        loaded = load_dataset(
            config.dataset_id,
            split=config.split,
            revision=config.revision,
        )
    except Exception as error:
        raise DatasetContractError(
            f"could not load {config.dataset_id!r} split {config.split!r}: {type(error).__name__}"
        ) from error

    column_names = set(getattr(loaded, "column_names", ()))
    required = {config.problem_field, config.answer_field, config.id_field}
    missing = sorted(required - column_names)
    if missing:
        raise DatasetContractError(
            f"dataset lacks required configured fields: {', '.join(missing)}"
        )

    records = tuple(
        _record_from_row(row, source_index, config)
        for source_index, row in enumerate(loaded)
    )
    if not records:
        raise DatasetContractError("dataset split is empty")

    fingerprint = getattr(loaded, "_fingerprint", None)
    return DatasetSnapshot(
        dataset_id=config.dataset_id,
        requested_revision=config.revision,
        split=config.split,
        observed_fingerprint=str(fingerprint) if fingerprint else None,
        records=records,
    )


def select_records_from_manifest(
    snapshot: DatasetSnapshot, manifest: Mapping[str, Any]
) -> tuple[ProblemRecord, ...]:
    """Verify source rows against an immutable manifest before model inference."""

    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise DatasetContractError("manifest records must be a non-empty list")

    selected: list[ProblemRecord] = []
    seen_indexes: set[int] = set()
    for expected in raw_records:
        if not isinstance(expected, Mapping):
            raise DatasetContractError("manifest record is not a mapping")
        source_index = expected.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise DatasetContractError("manifest source_index is invalid")
        if source_index in seen_indexes:
            raise DatasetContractError("manifest repeats a source_index")
        seen_indexes.add(source_index)
        if source_index < 0 or source_index >= len(snapshot.records):
            raise DatasetContractError("manifest source_index is outside loaded dataset")

        record = snapshot.records[source_index]
        _validate_manifest_record(record, expected)
        selected.append(record)
    return tuple(selected)


def _record_from_row(
    row: Mapping[str, Any], source_index: int, config: DatasetConfig
) -> ProblemRecord:
    problem_id = row.get(config.id_field)
    problem = row.get(config.problem_field)
    answer = row.get(config.answer_field)
    if not isinstance(problem_id, (str, int)) or not str(problem_id).strip():
        raise DatasetContractError(f"row {source_index} has an invalid problem ID")
    if not isinstance(problem, str) or not problem.strip():
        raise DatasetContractError(f"row {source_index} has an invalid problem")
    if not isinstance(answer, str) or not answer.strip():
        raise DatasetContractError(f"row {source_index} has an invalid reference answer")

    metadata: dict[str, str | int | None] = {}
    for key in ("subject", "level"):
        value = row.get(key)
        if value is None or isinstance(value, (str, int)):
            metadata[key] = value
        else:
            metadata[key] = str(value)

    return ProblemRecord(
        problem_id=str(problem_id),
        source_index=source_index,
        problem=problem,
        reference_answer=answer,
        metadata=metadata,
    )


def _selection_rank(selection_seed: int, problem_id: str, source_index: int) -> str:
    material = f"{selection_seed}\x00{problem_id}\x00{source_index}"
    return sha256_text(material)


def _validate_manifest_record(
    record: ProblemRecord, expected: Mapping[str, Any]
) -> None:
    comparisons = {
        "problem_id": record.problem_id,
        "problem_sha256": record.problem_sha256,
        "reference_answer_sha256": record.reference_answer_sha256,
        "metadata_sha256": record.metadata_sha256,
    }
    for key, actual in comparisons.items():
        if expected.get(key) != actual:
            raise DatasetContractError(
                f"manifest/source mismatch for source index {record.source_index}: {key}"
            )
