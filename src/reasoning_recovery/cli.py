"""Explicit command-line entry points for reviewed P0 actions.

No command runs at import time.  Model-backed work additionally requires the
human-visible ``--confirm-run`` flag; that flag is deliberately not inferred
from a manifest or an environment variable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import (
    load_selection_manifest,
    open_run_layout,
    read_json,
    write_immutable_json,
    write_selection_manifest,
)
from .config import PilotConfig, load_config
from .dataset import load_math500_snapshot, select_records_from_manifest
from .evaluation import default_backend
from .metrics import format_summary, summarize_run
from .model_runtime import TransformersRuntime
from .provenance import (
    base_runtime_provenance,
    hashed_filename,
    sha256_json,
    source_git_identity,
    utc_now_iso,
)
from .runner import execute_p0_run


class CommandError(RuntimeError):
    """An action cannot safely proceed under the declared P0 contract."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-manifest":
            _prepare_manifest(args)
        elif args.command == "run":
            _run(args)
        else:  # pragma: no cover - argparse choices make this unreachable.
            raise CommandError(f"unsupported command: {args.command}")
    except (CommandError, OSError, RuntimeError, ValueError) as error:
        print(f"rrp: {error}", file=sys.stderr)
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rrp",
        description="Feasibility-only saved-trajectory reasoning recovery probes.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser(
        "prepare-manifest",
        help="resolve MATH-500 and write a deterministic selection manifest",
    )
    manifest.add_argument("--config", required=True, type=Path)
    manifest.add_argument("--output", required=True, type=Path)
    manifest.add_argument(
        "--limit",
        type=_positive_int,
        help="optional reviewed subset size; defaults to config dataset.pilot_size",
    )

    run = commands.add_parser(
        "run",
        help="run only a reviewed manifest, with exact token-preserving artifacts",
    )
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument(
        "--max-problems",
        type=_positive_int,
        help="run the first N manifest records as an explicitly scoped subset",
    )
    run.add_argument(
        "--max-rollouts",
        type=_positive_int,
        help="run the first N configured rollout seeds as an explicitly scoped subset",
    )
    run.add_argument(
        "--confirm-run",
        action="store_true",
        help="required acknowledgement before model-backed generation",
    )
    return parser


def _prepare_manifest(args: argparse.Namespace) -> None:
    config_path, repo_root = _resolve_config(args.config)
    config = load_config(config_path)
    output = _resolve_repo_path(args.output, repo_root)
    snapshot = load_math500_snapshot(config.dataset)
    manifest = snapshot.selection_manifest(config=config, limit=args.limit)
    manifest.update(
        {
            "prepared_at_utc": utc_now_iso(),
            "source_git_identity": source_git_identity(repo_root),
            "preparation_runtime": base_runtime_provenance(),
        }
    )
    write_selection_manifest(output, manifest)
    print(f"wrote immutable selection manifest: {output}")
    print(f"records selected: {manifest['selection_size']}")
    print(f"dataset revision: {manifest['dataset']['resolved_revision']}")
    print("No model was loaded or invoked.")


def _run(args: argparse.Namespace) -> None:
    if not args.confirm_run:
        raise CommandError(
            "model-backed generation requires --confirm-run after manifest/protocol review"
        )
    config_path, repo_root = _resolve_config(args.config)
    config = load_config(config_path)
    source_identity = source_git_identity(repo_root)
    if source_identity.get("commit") is None:
        raise CommandError("model-backed execution requires a Git commit for source provenance")
    if source_identity.get("is_clean") is not True:
        raise CommandError(
            "model-backed execution requires a clean source tree; commit or preserve changes first"
        )

    manifest_path = _resolve_repo_path(args.manifest, repo_root)
    manifest = load_selection_manifest(manifest_path)
    _validate_manifest(config, manifest)
    dataset = manifest["dataset"]
    assert isinstance(dataset, Mapping)  # _validate_manifest establishes this.
    snapshot = load_math500_snapshot(
        config.dataset, revision_override=str(dataset["resolved_revision"])
    )
    _validate_reloaded_snapshot(config, manifest, snapshot)
    records = select_records_from_manifest(snapshot, manifest)
    selected_records = _bounded(records, args.max_problems, label="--max-problems")
    selected_seeds = _bounded(
        config.generation.rollout_seeds, args.max_rollouts, label="--max-rollouts"
    )

    execution_scope = {
        "selection_manifest_sha256": sha256_json(manifest),
        "problem_ids": [record.problem_id for record in selected_records],
        "source_indexes": [record.source_index for record in selected_records],
        "rollout_seeds": list(selected_seeds),
        "checkpoint_token_positions": list(config.generation.checkpoint_token_positions),
        "human_confirmed_model_run": True,
        "source_git_commit": source_identity["commit"],
    }
    artifact_root = _artifact_root(config, repo_root)
    layout = open_run_layout(
        artifact_root,
        manifest=manifest,
        config_hash=config.config_hash,
        execution_scope=execution_scope,
    )
    layout.write_run_manifest(
        {
            "schema_version": 1,
            "record_type": "RUN_MANIFEST",
            "run_id": layout.run_id,
            "manifest_hash": layout.manifest_hash,
            "config_hash": layout.config_hash,
            "config": config.to_dict(),
            "execution_scope": execution_scope,
            "manifest_filename": manifest_path.name,
        }
    )

    evaluator = default_backend()
    if evaluator is None:
        _write_setup_failure(
            layout,
            stage="EVALUATOR_QUALIFICATION",
            error=CommandError("math-verify is unavailable; no model request was issued"),
        )
        raise CommandError("math-verify is unavailable; install the [runtime] extra")

    runtime: TransformersRuntime | None = None
    try:
        runtime = _load_qualified_runtime(layout, config)
        marker_ids, cue_ids = runtime.validate_forced_cue(
            close_marker_text=config.forced_answer.close_think_marker_text,
            cue_text=config.forced_answer.close_think_text,
        )
        runtime_provenance_hash = _record_runtime_contract(
            layout=layout,
            config=config,
            runtime=runtime,
            close_marker_token_ids=marker_ids,
            close_think_cue_token_ids=cue_ids,
        )
        layout.ledger.append(
            "RUN_EXECUTION_STARTED",
            payload={
                "run_id": layout.run_id,
                "runtime_provenance_sha256": runtime_provenance_hash,
            },
        )
        execution = execute_p0_run(
            config=config,
            records=selected_records,
            rollout_seeds=selected_seeds,
            layout=layout,
            runtime=runtime,
            evaluator=evaluator,
            close_marker_token_ids=marker_ids,
            close_think_cue_token_ids=cue_ids,
            runtime_provenance_hash=runtime_provenance_hash,
            source_git_commit=str(source_identity["commit"]),
        )
        summary = summarize_run(
            layout,
            planned_problem_ids=execution.planned_problem_ids,
            planned_rollout_ids=execution.planned_rollout_ids,
            checkpoint_positions=config.generation.checkpoint_token_positions,
        )
        summary_path = layout.write_summary(summary)
        layout.ledger.append(
            "RUN_EXECUTION_FINISHED",
            payload={"run_id": layout.run_id, "summary_path": summary_path.name},
        )
        print(format_summary(summary))
        print(f"immutable summary: {summary_path}")
    except Exception as error:
        _write_setup_failure(layout, stage="RUN_OR_RUNTIME", error=error)
        raise
    finally:
        if runtime is not None:
            runtime.close()


def _load_qualified_runtime(layout: Any, config: PilotConfig) -> TransformersRuntime:
    layout.ledger.append(
        "RUNTIME_LOAD_STARTED",
        payload={
            "run_id": layout.run_id,
            "model_id": config.model.model_id,
            "requested_revision": config.model.revision,
            "requested_tokenizer_revision": config.model.tokenizer_revision,
        },
    )
    try:
        runtime = TransformersRuntime.load(config.model)
    except Exception as error:
        _write_setup_failure(layout, stage="RUNTIME_LOAD", error=error)
        layout.ledger.append(
            "RUNTIME_LOAD_FAILED",
            payload={"error_type": type(error).__name__, "message": str(error)[:1000]},
        )
        raise
    layout.ledger.append(
        "RUNTIME_LOAD_COMPLETED",
        payload={
            "model_id": config.model.model_id,
            "resolved_model_revision": runtime.provenance["model"]["resolved_revision"],
        },
    )
    return runtime


def _record_runtime_contract(
    *,
    layout: Any,
    config: PilotConfig,
    runtime: TransformersRuntime,
    close_marker_token_ids: Sequence[int],
    close_think_cue_token_ids: Sequence[int],
) -> str:
    """Persist one immutable runtime contract, or verify it on a resume."""

    payload = {
        "schema_version": 1,
        "record_type": "RUNTIME_CONTRACT",
        "run_id": layout.run_id,
        "manifest_hash": layout.manifest_hash,
        "config_hash": layout.config_hash,
        "runtime": runtime.provenance,
        "runtime_identity": _runtime_identity(runtime.provenance),
        "forced_answer": {
            "protocol_id": config.forced_answer.protocol_id,
            "close_think_marker_text": config.forced_answer.close_think_marker_text,
            "close_think_marker_token_ids": list(close_marker_token_ids),
            "close_think_marker_sha256": sha256_json(list(close_marker_token_ids)),
            "close_think_cue_text": config.forced_answer.close_think_text,
            "close_think_cue_token_ids": list(close_think_cue_token_ids),
            "close_think_cue_sha256": sha256_json(list(close_think_cue_token_ids)),
        },
        "recorded_at_utc": utc_now_iso(),
    }
    path = layout.run_root / "runtime_contract.json"
    if path.exists():
        existing = read_json(path)
        if existing.get("runtime_identity") != payload["runtime_identity"]:
            raise CommandError(
                "current runtime identity differs from the immutable run contract; create a new reviewed run"
            )
        if existing.get("forced_answer") != payload["forced_answer"]:
            raise CommandError("forced-answer token contract differs from the immutable run contract")
        return sha256_json(existing)
    write_immutable_json(path, payload)
    return sha256_json(payload)


def _write_setup_failure(layout: Any, *, stage: str, error: Exception) -> Path:
    """Keep a local immutable setup/run failure without replacing evidence."""

    payload = {
        "schema_version": 1,
        "record_type": "SETUP_OR_RUN_FAILURE",
        "run_id": layout.run_id,
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error)[:1000],
        "timestamp_utc": utc_now_iso(),
    }
    path = layout.run_root / "setup" / hashed_filename(
        f"failure:{stage}:{sha256_json(payload)}"
    )
    return write_immutable_json(path, payload)


def _validate_manifest(config: PilotConfig, manifest: Mapping[str, Any]) -> None:
    if manifest.get("config_hash") != config.config_hash:
        raise CommandError("manifest config hash does not match the supplied configuration")
    if manifest.get("study_protocol_id") != config.study.protocol_id:
        raise CommandError("manifest study protocol ID does not match the supplied configuration")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise CommandError("selection manifest has no dataset contract")
    expected = {
        "id": config.dataset.dataset_id,
        "split": config.dataset.split,
        "selection_method": config.dataset.selection_method,
        "selection_seed": config.dataset.selection_seed,
    }
    for key, value in expected.items():
        if dataset.get(key) != value:
            raise CommandError(f"manifest dataset field does not match configuration: {key}")
    if not isinstance(dataset.get("resolved_revision"), str) or not dataset["resolved_revision"]:
        raise CommandError("selection manifest lacks an immutable resolved dataset revision")


def _validate_reloaded_snapshot(
    config: PilotConfig, manifest: Mapping[str, Any], snapshot: Any
) -> None:
    dataset = manifest["dataset"]
    assert isinstance(dataset, Mapping)
    if snapshot.resolved_revision != dataset["resolved_revision"]:
        raise CommandError("reloaded dataset revision differs from the manifest")
    expected_fingerprint = dataset.get("observed_fingerprint")
    if expected_fingerprint != snapshot.observed_fingerprint:
        raise CommandError(
            "reloaded dataset fingerprint differs from the immutable manifest"
        )
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise CommandError("selection manifest has no selected records")
    if manifest.get("selection_size") != len(raw_records):
        raise CommandError("selection manifest has an inconsistent selection size")
    expected = snapshot.selection_manifest(config=config, limit=len(raw_records))
    if expected["records"] != raw_records:
        raise CommandError(
            "selection manifest does not match the deterministic configured rank order"
        )


def _bounded(values: Sequence[Any], limit: int | None, *, label: str) -> tuple[Any, ...]:
    selected = tuple(values)
    if limit is None:
        return selected
    if limit > len(selected):
        raise CommandError(f"{label}={limit} exceeds the available reviewed scope ({len(selected)})")
    return selected[:limit]


def _resolve_config(path: Path) -> tuple[Path, Path]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise CommandError(f"configuration file does not exist: {resolved}")
    repo_root = resolved.parent.parent
    if not (repo_root / "pyproject.toml").is_file():
        repo_root = Path.cwd().resolve()
    return resolved, repo_root


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _artifact_root(config: PilotConfig, repo_root: Path) -> Path:
    return _resolve_repo_path(Path(config.artifacts.root), repo_root)


def _runtime_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Drop volatile allocator snapshots before resume-identity comparison."""

    return {
        str(key): value
        for key, value in provenance.items()
        if key not in {"memory_after_load"}
    }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
