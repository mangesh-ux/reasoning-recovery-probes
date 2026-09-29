"""Command line for the separately frozen V2 recovery-representation study."""

from __future__ import annotations

import argparse
from dataclasses import replace
import io
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .artifacts import read_json, write_immutable_json
from .evaluation import default_backend
from .provenance import sha256_json, utc_now_iso
from .v2_artifacts import (
    open_v2_run_layout,
    require_clean_v2_source,
    v2_source_identity,
)
from .v2_config import V2Config, load_v2_config
from .v2_dataset import (
    load_deepmath_snapshot,
    load_v2_selection_manifest,
    select_v2_records_from_manifest,
    write_v2_selection_manifest,
)
from .v2_preflight import run_v2_synthetic_preflight
from .v2_probes import (
    V2ScientificBlock,
    V2SelectedProbe,
    V2ValidationSelection,
    activation_available_cohort,
    event_gate,
    evaluate_test_models,
    refit_models_for_test,
    select_validation_models,
)
from .v2_runner import execute_v2_collection, load_v2_analysis_inputs
from .v2_runtime import V2Runtime
from .v2_splits import (
    build_v2_grouped_split,
    v2_problem_ids_for_split,
    validate_v2_grouped_split,
)


class V2CommandError(RuntimeError):
    """A V2 CLI action cannot safely satisfy its frozen protocol."""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-manifest":
            _prepare_manifest(args)
        elif args.command == "prepare-split":
            _prepare_split(args)
        elif args.command == "qualify-runtime":
            _qualify_runtime(args)
        elif args.command == "preflight":
            _preflight(args)
        elif args.command == "collect-train-validation":
            _collect_train_validation(args)
        elif args.command == "select-model":
            _select_model(args)
        elif args.command == "collect-test":
            _collect_test(args)
        elif args.command == "analyze-test":
            _analyze_test(args)
        else:  # pragma: no cover - argparse validates choices
            raise V2CommandError(f"unsupported V2 command {args.command!r}")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"rrp-v2: {error}", file=sys.stderr)
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rrp-v2",
        description="Frozen, separately namespaced V2 recovery activation study.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("prepare-manifest", help="freeze the DeepMath V2 selection before model work")
    manifest.add_argument("--config", required=True, type=Path)
    manifest.add_argument("--output", required=True, type=Path)
    split = commands.add_parser("prepare-split", help="freeze the grouped V2 split before model work")
    split.add_argument("--config", required=True, type=Path)
    split.add_argument("--manifest", required=True, type=Path)
    split.add_argument("--output", required=True, type=Path)
    qualify = commands.add_parser("qualify-runtime", help="load and validate pinned V2 model/template without generation")
    qualify.add_argument("--config", required=True, type=Path)
    preflight = commands.add_parser("preflight", help="run synthetic-only generation and activation hardware qualification")
    preflight.add_argument("--config", required=True, type=Path)
    preflight.add_argument("--confirm-run", action="store_true", help="required before synthetic CUDA execution")
    collection = commands.add_parser("collect-train-validation", help="collect only train and validation V2 private receipts")
    _collection_arguments(collection)
    selection = commands.add_parser("select-model", help="write immutable train/validation selection before test collection")
    _data_arguments(selection)
    test = commands.add_parser("collect-test", help="collect test only after the immutable selection record exists")
    _collection_arguments(test)
    analysis = commands.add_parser("analyze-test", help="perform the one terminal aggregate test analysis")
    _data_arguments(analysis)
    return parser


def _data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)


def _collection_arguments(parser: argparse.ArgumentParser) -> None:
    _data_arguments(parser)
    parser.add_argument("--confirm-run", action="store_true", help="required before scientific model-backed collection")


def _prepare_manifest(args: argparse.Namespace) -> None:
    config_path, repo_root = _resolve_config(args.config)
    config = load_v2_config(config_path)
    output = _resolve_repo_path(args.output, repo_root)
    snapshot = load_deepmath_snapshot(config.dataset)
    manifest = snapshot.selection_manifest(config=config)
    manifest["prepared_at_utc"] = utc_now_iso()
    manifest["source_git_identity"] = v2_source_identity(repo_root)
    write_v2_selection_manifest(output, manifest)
    print(f"wrote immutable V2 selection manifest: {output}")
    print(f"records selected: {manifest['selection_size']}")
    print("No model was loaded or invoked.")


def _prepare_split(args: argparse.Namespace) -> None:
    config_path, repo_root = _resolve_config(args.config)
    config = load_v2_config(config_path)
    manifest = load_v2_selection_manifest(_resolve_repo_path(args.manifest, repo_root), config=config)
    split = build_v2_grouped_split(manifest, config=config)
    split["prepared_at_utc"] = utc_now_iso()
    split["source_git_identity"] = v2_source_identity(repo_root)
    output = _resolve_repo_path(args.output, repo_root)
    write_immutable_json(output, split)
    print(f"wrote immutable V2 grouped split: {output}")
    print("No model was loaded or invoked.")


def _qualify_runtime(args: argparse.Namespace) -> None:
    config_path, repo_root = _resolve_config(args.config)
    config = load_v2_config(config_path)
    source = require_clean_v2_source(repo_root)
    artifact_root = _artifact_root(config, repo_root)
    runtime: V2Runtime | None = None
    try:
        runtime = V2Runtime.load(config)
        marker, cue = runtime.validate_readout_contract()
        probe = runtime.tokenize_prompt("V2 runtime qualification only. Do not solve a benchmark problem.")
        payload: dict[str, object] = {
            "schema_version": 1,
            "record_type": "V2_LOAD_ONLY_RUNTIME_QUALIFICATION",
            "study_protocol_id": config.study.protocol_id,
            "config_hash": config.config_hash,
            "source_git_identity": source,
            "runtime": runtime.provenance,
            "control_contract": {"close_marker_token_ids": list(marker), "cue_token_ids": list(cue)},
            "template_probe": probe.safe_summary(),
            "qualified_at_utc": utc_now_iso(),
        }
    except Exception as error:
        payload = {"schema_version": 1, "record_type": "V2_LOAD_ONLY_RUNTIME_QUALIFICATION_FAILURE", "study_protocol_id": config.study.protocol_id, "config_hash": config.config_hash, "source_git_identity": source, "error_type": type(error).__name__, "message": str(error)[:1000], "qualified_at_utc": utc_now_iso()}
        path = _write_qualification(artifact_root, payload)
        print(f"immutable V2 runtime qualification failure: {path}", file=sys.stderr)
        raise
    finally:
        if runtime is not None:
            runtime.close()
    path = _write_qualification(artifact_root, payload)
    print(f"immutable V2 runtime qualification: {path}")
    print("No model trajectory was generated.")


def _preflight(args: argparse.Namespace) -> None:
    if not args.confirm_run:
        raise V2CommandError("synthetic CUDA qualification requires --confirm-run")
    config_path, repo_root = _resolve_config(args.config)
    config = load_v2_config(config_path)
    source = require_clean_v2_source(repo_root)
    artifact_root = _artifact_root(config, repo_root)
    runtime: V2Runtime | None = None
    try:
        runtime = V2Runtime.load(config)
        result = run_v2_synthetic_preflight(config=config, runtime=runtime, artifact_root=artifact_root)
        payload = dict(result.payload)
        payload["source_git_identity"] = source
    except Exception as error:
        payload = {"schema_version": 1, "record_type": "V2_SYNTHETIC_HARDWARE_QUALIFICATION_FAILURE", "study_protocol_id": config.study.protocol_id, "config_hash": config.config_hash, "source_git_identity": source, "error_type": type(error).__name__, "message": str(error)[:1000], "qualified_at_utc": utc_now_iso()}
        path = _write_qualification(artifact_root, payload)
        print(f"immutable V2 hardware qualification failure: {path}", file=sys.stderr)
        raise
    finally:
        if runtime is not None:
            runtime.close()
    path = _write_qualification(artifact_root, payload)
    gate = payload.get("hardware_gate", {})
    print(f"immutable V2 synthetic qualification: {path}")
    print(f"local hardware gate passed: {isinstance(gate, Mapping) and gate.get('passed') is True}")


def _collect_train_validation(args: argparse.Namespace) -> None:
    if not args.confirm_run:
        raise V2CommandError("V2 scientific collection requires --confirm-run")
    context = _load_run_context(args)
    runtime: V2Runtime | None = None
    try:
        runtime = V2Runtime.load(context.config)
        _write_runtime_contract(context, runtime)
        with context.layout.acquire_writer_lock():
            outputs = {
                split: execute_v2_collection(
                    config=context.config,
                    records=_records_for_split(context, split),
                    split=split,
                    layout=context.layout,
                    runtime=runtime,
                    evaluator=default_backend(),
                    source_identity=context.source_identity,
                ).to_dict()
                for split in ("train", "validation")
            }
    finally:
        if runtime is not None:
            runtime.close()
    print("V2 train/validation collection completed; no test trajectory was requested.")
    for split, output in outputs.items():
        print(f"{split}: {output}")


def _select_model(args: argparse.Namespace) -> None:
    context = _load_run_context(args)
    if context.layout.model_selection_path.exists():
        print(f"immutable V2 selection already exists: {context.layout.model_selection_path}")
        return
    train_inputs = load_v2_analysis_inputs(config=context.config, records=_records_for_split(context, "train"), layout=context.layout)
    validation_inputs = load_v2_analysis_inputs(config=context.config, records=_records_for_split(context, "validation"), layout=context.layout)
    selection = select_validation_models(train_rows=train_inputs.rows, validation_rows=validation_inputs.rows, activation_matrices={**train_inputs.activation_matrices, **validation_inputs.activation_matrices}, config=context.config)
    payload = {"schema_version": 1, "record_type": "V2_VALIDATION_SELECTION", "study_protocol_id": context.config.study.protocol_id, "config_hash": context.config.config_hash, "run_id": context.layout.run_id, "manifest_hash": context.layout.manifest_hash, "split_hash": context.layout.split_hash, "source_git_identity": context.source_identity, "input_counts": {"train": train_inputs.safe_counts(), "validation": validation_inputs.safe_counts()}, "selection": selection.to_private_dict(), "created_at_utc": utc_now_iso()}
    context.layout.write_model_selection(payload)
    print(f"immutable V2 validation selection: {context.layout.model_selection_path}")
    print(f"selection status: {selection.status}")
    if not selection.ready_for_test:
        print("Scientific event gate blocked test collection; V2 configuration was not changed.")


def _collect_test(args: argparse.Namespace) -> None:
    if not args.confirm_run:
        raise V2CommandError("V2 scientific collection requires --confirm-run")
    context = _load_run_context(args)
    selection_payload = _selection_payload(context)
    selection_status = _nested_mapping(selection_payload, "selection").get("status")
    if selection_status != "READY_FOR_TEST":
        raise V2CommandError("test collection is forbidden until immutable validation selection is READY_FOR_TEST")
    runtime: V2Runtime | None = None
    try:
        runtime = V2Runtime.load(context.config)
        _write_runtime_contract(context, runtime)
        with context.layout.acquire_writer_lock():
            output = execute_v2_collection(config=context.config, records=_records_for_split(context, "test"), split="test", layout=context.layout, runtime=runtime, evaluator=default_backend(), source_identity=context.source_identity).to_dict()
    finally:
        if runtime is not None:
            runtime.close()
    print("V2 test collection completed under prior immutable validation selection.")
    print(output)


def _analyze_test(args: argparse.Namespace) -> None:
    context = _load_run_context(args)
    summary_dir = context.layout.run_root / "summary"
    if summary_dir.exists() and any(summary_dir.glob("*.json")):
        raise V2CommandError("terminal V2 test analysis already exists; preserving the original result")
    payload = _selection_payload(context)
    selection = _restore_selection(context, payload)
    if not selection.ready_for_test:
        raise V2CommandError("V2 test analysis is forbidden without a READY_FOR_TEST selection")
    train_inputs = load_v2_analysis_inputs(config=context.config, records=_records_for_split(context, "train"), layout=context.layout)
    validation_inputs = load_v2_analysis_inputs(config=context.config, records=_records_for_split(context, "validation"), layout=context.layout)
    test_inputs = load_v2_analysis_inputs(config=context.config, records=_records_for_split(context, "test"), layout=context.layout)
    all_activations = {**train_inputs.activation_matrices, **validation_inputs.activation_matrices, **test_inputs.activation_matrices}
    models = refit_models_for_test(train_rows=train_inputs.rows, validation_rows=validation_inputs.rows, activation_matrices=all_activations, selection=selection, config=context.config)
    try:
        evaluation = evaluate_test_models(models=models, test_rows=test_inputs.rows, activation_matrices=all_activations, config=context.config)
    except V2ScientificBlock as error:
        result = {"schema_version": 1, "record_type": "V2_TERMINAL_TEST_SCIENTIFIC_BLOCK", "study_protocol_id": context.config.study.protocol_id, "config_hash": context.config.config_hash, "run_id": context.layout.run_id, "manifest_hash": context.layout.manifest_hash, "split_hash": context.layout.split_hash, "source_git_identity": context.source_identity, "input_counts": {"train": train_inputs.safe_counts(), "validation": validation_inputs.safe_counts(), "test": test_inputs.safe_counts()}, "selection": selection.to_dict(), "scientific_block": error.payload, "completed_at_utc": utc_now_iso()}
    else:
        _write_model_bundle(context, models)
        result = {"schema_version": 1, "record_type": "V2_TERMINAL_TEST_ANALYSIS", "study_protocol_id": context.config.study.protocol_id, "config_hash": context.config.config_hash, "run_id": context.layout.run_id, "manifest_hash": context.layout.manifest_hash, "split_hash": context.layout.split_hash, "source_git_identity": context.source_identity, "input_counts": {"train": train_inputs.safe_counts(), "validation": validation_inputs.safe_counts(), "test": test_inputs.safe_counts()}, "selection": selection.to_dict(), "test_analysis": evaluation.to_dict(), "completed_at_utc": utc_now_iso()}
    summary_path = context.layout.write_summary(result)
    public = _public_summary(result)
    report_path = _resolve_repo_path(Path(context.config.artifacts.public_summary_path), context.repo_root)
    if report_path.exists():
        existing = read_json(report_path)
        if existing != public:
            # A prior local hardware-block summary is itself evidence and must
            # remain immutable.  A later qualified 24 GiB run gets a separate
            # public aggregate rather than silently replacing that block.
            if existing.get("status") == "LOCAL_HARDWARE_BLOCK":
                report_path = report_path.with_name(
                    f"{report_path.stem}-{context.layout.run_id}{report_path.suffix}"
                )
            else:
                raise V2CommandError(
                    "an immutable public V2 summary already exists for a different result"
                )
    write_immutable_json(report_path, public)
    print(f"immutable V2 terminal test summary: {summary_path}")
    print(f"aggregate-only public V2 summary: {report_path}")


def _load_run_context(args: argparse.Namespace) -> "_RunContext":
    config_path, repo_root = _resolve_config(args.config)
    config = load_v2_config(config_path)
    source = require_clean_v2_source(repo_root)
    manifest = load_v2_selection_manifest(_resolve_repo_path(args.manifest, repo_root), config=config)
    split = read_json(_resolve_repo_path(args.split, repo_root))
    validate_v2_grouped_split(split, manifest=manifest, config=config)
    snapshot = load_deepmath_snapshot(config.dataset, revision_override=config.dataset.revision)
    records = select_v2_records_from_manifest(snapshot, manifest)
    layout = open_v2_run_layout(_artifact_root(config, repo_root), config_hash=config.config_hash, manifest=manifest, split=split, source_identity=source)
    run_manifest = {"schema_version": 1, "record_type": "V2_RUN_MANIFEST", "study_protocol_id": config.study.protocol_id, "config_hash": config.config_hash, "run_id": layout.run_id, "manifest_hash": layout.manifest_hash, "split_hash": layout.split_hash, "source_git_identity": source, "selection_manifest": manifest, "split_manifest": split}
    layout.write_run_manifest(run_manifest)
    return _RunContext(config=config, repo_root=repo_root, source_identity=source, manifest=manifest, split=split, records=records, layout=layout)


def _write_runtime_contract(context: "_RunContext", runtime: V2Runtime) -> None:
    marker, cue = runtime.validate_readout_contract()
    payload = {"schema_version": 1, "record_type": "V2_RUNTIME_CONTRACT", "study_protocol_id": context.config.study.protocol_id, "config_hash": context.config.config_hash, "run_id": context.layout.run_id, "manifest_hash": context.layout.manifest_hash, "split_hash": context.layout.split_hash, "source_git_identity": context.source_identity, "runtime": runtime.provenance, "close_marker_token_ids": list(marker), "readout_cue_token_ids": list(cue)}
    context.layout.write_runtime_contract(payload)


def _records_for_split(context: "_RunContext", split_name: str) -> tuple[Any, ...]:
    ids = set(v2_problem_ids_for_split(context.split, split_name))
    records = tuple(record for record in context.records if record.problem_id in ids)
    if len(records) != len(ids):
        raise V2CommandError(f"could not bind every {split_name} problem to the frozen dataset snapshot")
    return records


def _selection_payload(context: "_RunContext") -> dict[str, Any]:
    if not context.layout.model_selection_path.exists():
        raise V2CommandError("immutable V2 validation selection record is missing")
    payload = read_json(context.layout.model_selection_path)
    if payload.get("record_type") != "V2_VALIDATION_SELECTION" or payload.get("run_id") != context.layout.run_id:
        raise V2CommandError("selection record does not match this immutable V2 run")
    return payload


def _restore_selection(context: "_RunContext", payload: Mapping[str, Any]) -> V2ValidationSelection:
    serialized = _nested_mapping(payload, "selection")
    private_ids = _nested_mapping(serialized, "private_activation_cohort_row_ids")
    train_inputs = load_v2_analysis_inputs(config=context.config, records=_records_for_split(context, "train"), layout=context.layout)
    validation_inputs = load_v2_analysis_inputs(config=context.config, records=_records_for_split(context, "validation"), layout=context.layout)
    all_activations = {**train_inputs.activation_matrices, **validation_inputs.activation_matrices}
    train_cohort = activation_available_cohort(train_inputs.rows, all_activations, config=context.config)
    validation_cohort = activation_available_cohort(validation_inputs.rows, all_activations, config=context.config)
    for split_name, cohort in (("train", train_cohort), ("validation", validation_cohort)):
        expected = private_ids.get(split_name)
        actual = [row.row_id for row in cohort.rows]
        if expected != actual:
            raise V2CommandError("train/validation activation cohort differs from immutable selection")
    selected_a = _selected_probe_from_payload(serialized.get("selected_a"))
    selected_b = _selected_probe_from_payload(serialized.get("selected_b"))
    raw_c = _nested_mapping(serialized, "selected_c_by_layer")
    selected_c = {int(layer): _selected_probe_from_payload(entry) for layer, entry in raw_c.items()}
    if any(value is None for value in selected_c.values()):
        raise V2CommandError("selection has a malformed C-layer model specification")
    gates = {"train": event_gate(train_cohort.rows, split="train", config=context.config), "validation": event_gate(validation_cohort.rows, split="validation", config=context.config)}
    primary_layer = serialized.get("validation_selected_primary_layer")
    if isinstance(primary_layer, bool) or not isinstance(primary_layer, int):
        primary_layer = None
    block_reasons = serialized.get("block_reasons")
    if not isinstance(block_reasons, list) or any(not isinstance(item, str) for item in block_reasons):
        raise V2CommandError("selection has malformed block reasons")
    return V2ValidationSelection(status=str(serialized.get("status")), paired_cohort={"train": train_cohort, "validation": validation_cohort}, event_gates=gates, selected_a=selected_a, selected_b=selected_b, selected_c_by_layer={layer: value for layer, value in selected_c.items() if value is not None}, primary_layer=primary_layer, block_reasons=tuple(block_reasons))


def _selected_probe_from_payload(value: object) -> V2SelectedProbe | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise V2CommandError("selection model specification is malformed")
    metrics = value.get("validation_metrics")
    if not isinstance(metrics, Mapping):
        raise V2CommandError("selection validation metrics are malformed")
    return V2SelectedProbe(model_name=str(value.get("model_name")), feature_set=str(value.get("feature_set")), regularization_c=float(value.get("regularization_c")), layer_index=(None if value.get("layer_index") is None else int(value.get("layer_index"))), validation_metrics={str(key): float(item) for key, item in metrics.items()}, train_row_count=int(value.get("train_row_count")), validation_row_count=int(value.get("validation_row_count")), output_feature_count=int(value.get("output_feature_count")))


def _write_model_bundle(context: "_RunContext", models: Any) -> None:
    if context.layout.model_bundle_path.exists():
        return
    try:
        import joblib
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise V2CommandError("V2 final model serialization requires joblib") from error
    stream = io.BytesIO()
    joblib.dump(models.joblib_payload(), stream)
    context.layout.write_model_bundle(stream.getvalue())


def _public_summary(result: Mapping[str, Any]) -> dict[str, object]:
    # This deliberately whitelists aggregates.  It never copies source paths,
    # private row IDs, selection identifiers, raw receipts, tokens, prompts,
    # answers, model objects, tensors, or the private artifact root.
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "V2_PUBLIC_AGGREGATE_SUMMARY",
        "study_protocol_id": result["study_protocol_id"],
        "config_hash": result["config_hash"],
        "input_counts": result["input_counts"],
        "selection": result["selection"],
    }
    if "test_analysis" in result:
        payload["test_analysis"] = result["test_analysis"]
    if "scientific_block" in result:
        payload["scientific_block"] = result["scientific_block"]
    return payload


def _write_qualification(root: Path, payload: Mapping[str, object]) -> Path:
    path = root / "qualification" / f"qualification-{sha256_json(payload)}.json"
    return write_immutable_json(path, payload)


def _resolve_config(path: Path) -> tuple[Path, Path]:
    config_path = path.resolve()
    if not config_path.exists():
        raise V2CommandError(f"V2 configuration does not exist: {config_path}")
    repo_root = _find_repo_root(config_path.parent)
    return config_path, repo_root


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _find_repo_root(start: Path) -> Path:
    candidate = start.resolve()
    while candidate.parent != candidate:
        if (candidate / ".git").exists():
            return candidate
        candidate = candidate.parent
    raise V2CommandError("could not locate the V2 Git repository root")


def _artifact_root(config: V2Config, repo_root: Path) -> Path:
    return (repo_root / config.artifacts.root).resolve()


def _nested_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise V2CommandError(f"expected mapping field {key!r}")
    return value


class _RunContext:
    def __init__(self, *, config: V2Config, repo_root: Path, source_identity: Mapping[str, object], manifest: Mapping[str, Any], split: Mapping[str, Any], records: Sequence[Any], layout: Any) -> None:
        self.config = config
        self.repo_root = repo_root
        self.source_identity = source_identity
        self.manifest = manifest
        self.split = split
        self.records = tuple(records)
        self.layout = layout


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
