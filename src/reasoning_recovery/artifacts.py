"""Immutable local artifacts and append-only request ledger for P0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .provenance import (
    canonical_json,
    hashed_filename,
    sha256_json,
    utc_now_iso,
)


class ArtifactConflictError(RuntimeError):
    """Raised when an immutable artifact path already holds different content."""


class RunLockError(RuntimeError):
    """Raised when a run cannot prove that it has the only writer lease."""


class IntentState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    STARTED_UNKNOWN = "STARTED_UNKNOWN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED_UNKNOWN = "INTERRUPTED_UNKNOWN"


_TERMINAL_EVENTS = {
    "RUNTIME_LOAD_COMPLETED": IntentState.COMPLETED,
    "RUNTIME_LOAD_FAILED": IntentState.FAILED,
    "BASE_COMPLETED": IntentState.COMPLETED,
    "BASE_FAILED": IntentState.FAILED,
    "CHECKPOINT_COMPLETED": IntentState.COMPLETED,
    "CHECKPOINT_FAILED": IntentState.FAILED,
    "INTERRUPTED_UNKNOWN": IntentState.INTERRUPTED_UNKNOWN,
}


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object with a clear error for malformed local evidence."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"cannot read artifact {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"artifact is malformed JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"artifact is not a JSON object: {path}")
    return value


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Create one JSON file exclusively, never replacing a prior receipt."""

    content = json.dumps(
        _json_ready(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing = read_json(path)
        if canonical_json(existing) != canonical_json(payload):
            raise ArtifactConflictError(
                f"refusing to replace immutable artifact with different content: {path}"
            )
    return path


class AppendOnlyLedger:
    """An append-only event log with conservative interrupted-request handling."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def records(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                raise RuntimeError(f"blank ledger line at {self.path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"malformed ledger record at {self.path}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"non-object ledger record at {self.path}:{line_number}"
                )
            records.append(record)
        return tuple(records)

    def append(
        self,
        event_type: str,
        *,
        intent_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Append a timestamped event; no existing ledger content is changed."""

        record = {
            "schema_version": 1,
            "timestamp_utc": utc_now_iso(),
            "event_type": event_type,
            "intent_id": intent_id,
            "payload": _json_ready(payload or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = canonical_json(record) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def state(self, intent_id: str) -> IntentState:
        """Return a conservative state, rejecting malformed intent histories.

        A terminal event is evidence only when it follows exactly one start.
        Treating orphan events as completed would turn a corrupted ledger into
        apparent experimental evidence, so this method fails closed instead.
        """

        started = False
        terminal: IntentState | None = None
        for record in self.records():
            if record.get("intent_id") != intent_id:
                continue
            event_type = record.get("event_type")
            if event_type == "INTENT_STARTED":
                if started:
                    raise RuntimeError(f"duplicate intent start: {intent_id}")
                if terminal is not None:
                    raise RuntimeError(f"intent start after terminal event: {intent_id}")
                started = True
            elif event_type in _TERMINAL_EVENTS:
                if not started:
                    raise RuntimeError(f"terminal event without intent start: {intent_id}")
                if terminal is not None:
                    raise RuntimeError(f"duplicate terminal event: {intent_id}")
                terminal = _TERMINAL_EVENTS[event_type]
        if terminal is not None:
            return terminal
        if started:
            return IntentState.STARTED_UNKNOWN
        return IntentState.NOT_STARTED

    def preserve_interrupted_unknown(self, intent_id: str) -> IntentState:
        """Terminalize a started-but-unfinished request without rerunning it."""

        state = self.state(intent_id)
        if state is IntentState.STARTED_UNKNOWN:
            self.append(
                "INTERRUPTED_UNKNOWN",
                intent_id=intent_id,
                payload={"reason": "prior run ended without terminal receipt"},
            )
            return IntentState.INTERRUPTED_UNKNOWN
        return state


class RunWriterLock:
    """An exclusive, ownership-checked writer lease for one run directory.

    A crashed process leaves its lock in place deliberately.  The evidence in
    that case is ambiguous, so a later process must not infer that resuming is
    safe merely because the recorded owner PID is no longer live.
    """

    def __init__(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.path = path
        self._payload = dict(payload)
        self._released = False

    def __enter__(self) -> "RunWriterLock":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()

    def release(self) -> None:
        """Release only a lock whose on-disk owner record still matches ours."""

        if self._released:
            return
        if not self.path.exists():
            raise RunLockError(
                f"run writer lock disappeared before release; preserving uncertainty: {self.path}"
            )
        try:
            existing = read_json(self.path)
        except RuntimeError as error:
            raise RunLockError(
                f"cannot verify ownership of run writer lock; preserving it: {self.path}"
            ) from error
        if canonical_json(existing) != canonical_json(self._payload):
            raise RunLockError(
                f"run writer lock ownership changed; preserving it: {self.path}"
            )
        try:
            self.path.unlink()
        except OSError as error:
            raise RunLockError(
                f"cannot release run writer lock; preserving it: {self.path}"
            ) from error
        self._released = True


@dataclass(frozen=True)
class RunLayout:
    """Paths and IDs for one immutable configuration/manifest/campaign run."""

    root: Path
    run_id: str
    manifest_hash: str
    config_hash: str
    campaign_scope_hash: str

    @property
    def run_root(self) -> Path:
        return self.root / "runs" / self.run_id

    @property
    def ledger(self) -> AppendOnlyLedger:
        return AppendOnlyLedger(self.run_root / "ledger.jsonl")

    @property
    def run_manifest_path(self) -> Path:
        return self.run_root / "run_manifest.json"

    @property
    def writer_lock_path(self) -> Path:
        return self.run_root / "writer.lock.json"

    def rollout_path(self, rollout_id: str) -> Path:
        return self.run_root / "raw" / "rollouts" / hashed_filename(rollout_id)

    def checkpoint_path(self, checkpoint_id: str) -> Path:
        return self.run_root / "raw" / "checkpoints" / hashed_filename(checkpoint_id)

    def summary_path(self, payload: Mapping[str, Any]) -> Path:
        """Return a content-addressed summary path without overwriting history."""

        return self.run_root / "summary" / hashed_filename(
            f"summary:{sha256_json(payload)}"
        )

    def write_run_manifest(self, payload: Mapping[str, Any]) -> Path:
        return write_immutable_json(self.run_manifest_path, payload)

    def acquire_writer_lock(self) -> RunWriterLock:
        """Atomically acquire the only writer lease, never breaking an old lock."""

        payload = {
            "schema_version": 1,
            "record_type": "RUN_WRITER_LOCK",
            "run_id": self.run_id,
            "owner_pid": os.getpid(),
            "owner_token": uuid4().hex,
            "acquired_at_utc": utc_now_iso(),
        }
        self.writer_lock_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        try:
            with self.writer_lock_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise RunLockError(
                "run writer lock already exists; it may be active or stale and was "
                f"preserved: {self.writer_lock_path}"
            ) from error
        return RunWriterLock(self.writer_lock_path, payload)

    def write_rollout(self, rollout_id: str, payload: Mapping[str, Any]) -> Path:
        return write_immutable_json(self.rollout_path(rollout_id), payload)

    def write_checkpoint(self, checkpoint_id: str, payload: Mapping[str, Any]) -> Path:
        return write_immutable_json(self.checkpoint_path(checkpoint_id), payload)

    def write_summary(self, payload: Mapping[str, Any]) -> Path:
        return write_immutable_json(self.summary_path(payload), payload)

    def read_rollout(self, rollout_id: str) -> dict[str, Any] | None:
        path = self.rollout_path(rollout_id)
        return read_json(path) if path.exists() else None

    def read_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        path = self.checkpoint_path(checkpoint_id)
        return read_json(path) if path.exists() else None

    def disk_usage_bytes(self) -> int:
        if not self.run_root.exists():
            return 0
        return sum(
            path.stat().st_size
            for path in self.run_root.rglob("*")
            if path.is_file()
        )


def write_selection_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Write a reviewed selection manifest exclusively before model calls."""

    return write_immutable_json(path, manifest)


def load_selection_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest and bind its content hash to later run identities."""

    manifest = read_json(path)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported selection-manifest schema")
    if not isinstance(manifest.get("records"), list) or not manifest["records"]:
        raise RuntimeError("selection manifest must contain records")
    return manifest


def open_run_layout(
    artifact_root: Path,
    *,
    manifest: Mapping[str, Any],
    config_hash: str,
    campaign_scope: Mapping[str, Any],
) -> RunLayout:
    """Derive a stable run ID from immutable study inputs and full campaign scope.

    Invocation subsets intentionally do not belong here: a reviewed first stage
    and a later full-campaign stage must share one receipt namespace.
    """

    manifest_hash = sha256_json(manifest)
    campaign_scope_hash = sha256_json(_json_ready(campaign_scope))
    run_fingerprint = {
        "manifest_hash": manifest_hash,
        "config_hash": config_hash,
        "campaign_scope_hash": campaign_scope_hash,
    }
    run_id = f"run-{sha256_json(run_fingerprint)[:20]}"
    return RunLayout(
        root=artifact_root,
        run_id=run_id,
        manifest_hash=manifest_hash,
        config_hash=config_hash,
        campaign_scope_hash=campaign_scope_hash,
    )


def rollout_id(manifest_hash: str, problem_id: str, rollout_seed: int) -> str:
    """Stable identity for one original sampled trajectory."""

    return f"rollout:{manifest_hash[:16]}:{problem_id}:seed-{rollout_seed}"


def checkpoint_id(rollout_identity: str, checkpoint_token: int) -> str:
    """Stable identity for one child forced-answer request."""

    return f"checkpoint:{rollout_identity}:token-{checkpoint_token}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"artifact payload contains unsupported {type(value).__name__}")
