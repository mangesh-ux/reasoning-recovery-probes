"""Immutable private artifacts and source-bound identities for Protocol V2.

P0 artifacts remain untouched.  V2 uses a separate ledger grammar and run
identity because a continuation must prove the source commit and tree as well
as the frozen config, manifest, and grouped split.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping
from uuid import uuid4

from .artifacts import ArtifactConflictError, RunLockError, read_json, write_immutable_json
from .provenance import canonical_json, hashed_filename, sha256_json, utc_now_iso


class V2IntentState(str, Enum):
    """Conservative lifecycle for any V2 model or activation operation."""

    NOT_STARTED = "NOT_STARTED"
    STARTED_UNKNOWN = "STARTED_UNKNOWN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED_UNKNOWN = "INTERRUPTED_UNKNOWN"


_TERMINAL_EVENT_STATES = {
    "INTENT_COMPLETED": V2IntentState.COMPLETED,
    "INTENT_FAILED": V2IntentState.FAILED,
    "INTERRUPTED_UNKNOWN": V2IntentState.INTERRUPTED_UNKNOWN,
}


def v2_source_identity(repo_root: Path) -> dict[str, object]:
    """Return a source identity that fails closed for an unclean checkout."""

    try:
        commit = _git(repo_root, "rev-parse", "HEAD")
        tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
        status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    except RuntimeError:
        return {
            "commit": None,
            "tree": None,
            "is_clean": None,
        }
    return {
        "commit": commit or None,
        "tree": tree or None,
        "is_clean": not bool(status),
    }


def require_clean_v2_source(repo_root: Path) -> dict[str, object]:
    """Require a committed, clean tree before a V2 model-backed request."""

    identity = v2_source_identity(repo_root)
    if not isinstance(identity.get("commit"), str) or not isinstance(identity.get("tree"), str):
        raise RuntimeError("V2 model-backed execution requires a Git commit and tree")
    if identity.get("is_clean") is not True:
        raise RuntimeError("V2 model-backed execution requires a clean committed source tree")
    return identity


class V2AppendOnlyLedger:
    """Intent-first V2 ledger that preserves unknown remote/model outcomes."""

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
                raise RuntimeError(f"blank V2 ledger line at {self.path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"malformed V2 ledger record at {self.path}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"non-object V2 ledger record at {self.path}:{line_number}"
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
        record = {
            "schema_version": 1,
            "record_type": "V2_LEDGER_EVENT",
            "timestamp_utc": utc_now_iso(),
            "event_type": event_type,
            "intent_id": intent_id,
            "payload": _json_ready(payload or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def state(self, intent_id: str) -> V2IntentState:
        started = False
        terminal: V2IntentState | None = None
        for record in self.records():
            if record.get("intent_id") != intent_id:
                continue
            event_type = record.get("event_type")
            if event_type == "INTENT_STARTED":
                if started or terminal is not None:
                    raise RuntimeError(f"invalid duplicate V2 intent start: {intent_id}")
                started = True
                continue
            state = _TERMINAL_EVENT_STATES.get(str(event_type))
            if state is None:
                continue
            if not started or terminal is not None:
                raise RuntimeError(f"invalid V2 terminal intent history: {intent_id}")
            terminal = state
        if terminal is not None:
            return terminal
        return V2IntentState.STARTED_UNKNOWN if started else V2IntentState.NOT_STARTED

    def preserve_interrupted_unknown(self, intent_id: str) -> V2IntentState:
        state = self.state(intent_id)
        if state is V2IntentState.STARTED_UNKNOWN:
            self.append(
                "INTERRUPTED_UNKNOWN",
                intent_id=intent_id,
                payload={"reason": "prior V2 process ended before a terminal receipt"},
            )
            return V2IntentState.INTERRUPTED_UNKNOWN
        return state


class V2RunWriterLock:
    """Exclusive V2 writer lease that is never automatically broken."""

    def __init__(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.path = path
        self.payload = dict(payload)
        self._released = False

    def __enter__(self) -> "V2RunWriterLock":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        if not self.path.exists():
            raise RunLockError(f"V2 writer lock disappeared: {self.path}")
        existing = read_json(self.path)
        if canonical_json(existing) != canonical_json(self.payload):
            raise RunLockError(f"V2 writer lock ownership changed: {self.path}")
        try:
            self.path.unlink()
        except OSError as error:
            raise RunLockError(f"cannot release V2 writer lock: {self.path}") from error
        self._released = True


@dataclass(frozen=True)
class V2RunLayout:
    """Separate V2 artifact namespace with source-bound stable identity."""

    root: Path
    run_id: str
    config_hash: str
    manifest_hash: str
    split_hash: str
    source_identity: Mapping[str, object]

    @property
    def run_root(self) -> Path:
        return self.root / "runs" / self.run_id

    @property
    def ledger(self) -> V2AppendOnlyLedger:
        return V2AppendOnlyLedger(self.run_root / "ledger.jsonl")

    @property
    def run_manifest_path(self) -> Path:
        return self.run_root / "run_manifest.json"

    @property
    def runtime_contract_path(self) -> Path:
        return self.run_root / "runtime_contract.json"

    @property
    def writer_lock_path(self) -> Path:
        return self.run_root / "writer.lock.json"

    @property
    def model_selection_path(self) -> Path:
        return self.run_root / "selection" / "validation_selection.json"

    @property
    def model_bundle_path(self) -> Path:
        return self.run_root / "selection" / "validated_models.joblib"

    def rollout_path(self, rollout_id: str) -> Path:
        return self.run_root / "raw" / "rollouts" / hashed_filename(rollout_id)

    def terminal_readout_path(self, rollout_id: str) -> Path:
        return self.run_root / "raw" / "terminal_readouts" / hashed_filename(
            f"terminal-readout:{rollout_id}"
        )

    def checkpoint_readout_path(self, checkpoint_id: str) -> Path:
        return self.run_root / "raw" / "checkpoint_readouts" / hashed_filename(
            f"checkpoint-readout:{checkpoint_id}"
        )

    def activation_receipt_path(self, checkpoint_id: str) -> Path:
        return self.run_root / "raw" / "activation_receipts" / hashed_filename(
            f"activation:{checkpoint_id}"
        )

    def activation_tensor_path(self, checkpoint_id: str) -> Path:
        logical_id = f"activation-tensor:{checkpoint_id}"
        return self.run_root / "raw" / "activation_tensors" / hashed_filename(
            logical_id, suffix=".pt"
        )

    def summary_path(self, payload: Mapping[str, Any]) -> Path:
        return self.run_root / "summary" / hashed_filename(
            f"v2-summary:{sha256_json(payload)}"
        )

    def write_run_manifest(self, payload: Mapping[str, Any]) -> Path:
        return write_immutable_json(self.run_manifest_path, payload)

    def write_runtime_contract(self, payload: Mapping[str, Any]) -> Path:
        path = self.runtime_contract_path
        if path.exists():
            existing = read_json(path)
            if canonical_json(existing) != canonical_json(payload):
                raise ArtifactConflictError(
                    "V2 runtime contract differs from the immutable prior contract"
                )
            return path
        return write_immutable_json(path, payload)

    def write_rollout(self, rollout_id: str, payload: Mapping[str, Any]) -> Path:
        return write_immutable_json(self.rollout_path(rollout_id), payload)

    def write_terminal_readout(self, rollout_id: str, payload: Mapping[str, Any]) -> Path:
        return write_immutable_json(self.terminal_readout_path(rollout_id), payload)

    def write_checkpoint_readout(
        self, checkpoint_id: str, payload: Mapping[str, Any]
    ) -> Path:
        return write_immutable_json(self.checkpoint_readout_path(checkpoint_id), payload)

    def write_activation_receipt(
        self, checkpoint_id: str, payload: Mapping[str, Any]
    ) -> Path:
        return write_immutable_json(self.activation_receipt_path(checkpoint_id), payload)

    def write_activation_tensor(self, checkpoint_id: str, content: bytes) -> Path:
        return write_immutable_bytes(self.activation_tensor_path(checkpoint_id), content)

    def write_model_selection(self, payload: Mapping[str, Any]) -> Path:
        return write_immutable_json(self.model_selection_path, payload)

    def write_model_bundle(self, content: bytes) -> Path:
        return write_immutable_bytes(self.model_bundle_path, content)

    def write_summary(self, payload: Mapping[str, Any]) -> Path:
        return write_immutable_json(self.summary_path(payload), payload)

    def acquire_writer_lock(self) -> V2RunWriterLock:
        payload = {
            "schema_version": 1,
            "record_type": "V2_RUN_WRITER_LOCK",
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
                "V2 writer lock already exists and was preserved: "
                f"{self.writer_lock_path}"
            ) from error
        return V2RunWriterLock(self.writer_lock_path, payload)

    def disk_usage_bytes(self) -> int:
        if not self.run_root.exists():
            return 0
        return sum(path.stat().st_size for path in self.run_root.rglob("*") if path.is_file())


def open_v2_run_layout(
    artifact_root: Path,
    *,
    config_hash: str,
    manifest: Mapping[str, Any],
    split: Mapping[str, Any],
    source_identity: Mapping[str, object],
) -> V2RunLayout:
    """Bind V2 evidence to config, data, split, source commit, and source tree."""

    commit = source_identity.get("commit")
    tree = source_identity.get("tree")
    if not isinstance(commit, str) or not isinstance(tree, str):
        raise RuntimeError("V2 run identity requires a source commit and tree")
    manifest_hash = sha256_json(manifest)
    split_hash = sha256_json(split)
    fingerprint = {
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "split_hash": split_hash,
        "source_commit": commit,
        "source_tree": tree,
    }
    run_id = f"v2-run-{sha256_json(fingerprint)[:20]}"
    return V2RunLayout(
        root=artifact_root,
        run_id=run_id,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        split_hash=split_hash,
        source_identity=dict(source_identity),
    )


def v2_rollout_id(manifest_hash: str, problem_id: str, rollout_index: int) -> str:
    """Stable identity for one V2 stochastic reasoning trajectory."""

    return f"v2-rollout:{manifest_hash[:16]}:{problem_id}:rollout-{rollout_index}"


def v2_checkpoint_id(rollout_identity: str, checkpoint_token: int) -> str:
    """Stable identity for one fixed-coordinate V2 checkpoint."""

    return f"v2-checkpoint:{rollout_identity}:token-{checkpoint_token}"


def write_immutable_bytes(path: Path, content: bytes) -> Path:
    """Exclusively create binary evidence or verify byte-identical prior content."""

    if not isinstance(content, bytes):
        raise TypeError("immutable binary content must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = hashlib.sha256(content).hexdigest()
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        actual_hash = _sha256_bytes_file(path)
        if actual_hash != expected_hash:
            raise ArtifactConflictError(
                f"refusing to replace immutable binary artifact with different bytes: {path}"
            )
    return path


def _sha256_bytes_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"could not inspect V2 source identity: {args}") from error
    return completed.stdout.strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"V2 artifact payload has unsupported {type(value).__name__}")
