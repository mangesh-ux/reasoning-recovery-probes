"""Portable read-only reproduction of original V2-A1 eligibility aggregates.

Standard library only. No evaluator, model, tensor loading, fitting, network,
answer printing, or test receipt reads. Detailed input paths stay in the
write-once private report; stdout contains only aggregate metadata and hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any


PREFIX = "reasoning-recovery-probes/artifacts/v2a1/"
INVENTORY_SHA = "b7782d22c207e8463fd4b41ecf772282a69f305ddc517c44239805df86198775"
INVENTORY_BYTES = 4082363
VERIFICATION_SHA = "85f24f02eb3e07dc8aaddd013aba65445701d1f25fd29b46c45c8979206fe6fa"
VERIFICATION_BYTES = 1182
PRIVATE_NAMESPACE = Path("artifacts/measurement-reproduction/v1")
STATUS = {"COMPLETED", "FAILED", "UNAVAILABLE", "INTERRUPTED_UNKNOWN"}
AVAILABILITY = {"AVAILABLE", "AFTER_THINK_CLOSE", "TRAJECTORY_TOO_SHORT", "EARLY_NO_CLOSE"}
TERMINATION = {"EOS", "MAX_NEW_TOKENS", "STOPPED_OTHER", "THINK_CLOSE_MARKER"}
EXTRACTION = {"EXTRACTED", "MISSING_BOXED_ANSWER", "MALFORMED_BOXED_ANSWER", "EMPTY_BOXED_ANSWER", "ERROR"}
EVALUATION = {"CORRECT", "INCORRECT", "NON_EVALUABLE", "ERROR"}
LABELS = {"W_TO_C", "W_TO_W", "C_TO_C", "C_TO_W"}
PRIMARY = {"W_TO_C", "W_TO_W"}
REASONS = AVAILABILITY | {
    "CHECKPOINT_READOUT_NOT_EOS", "TERMINAL_READOUT_NOT_COMPLETED",
    "TERMINAL_READOUT_NOT_EOS", "TERMINAL_EVALUATION_UNAVAILABLE",
    "EVALUATION_RECORD_MALFORMED", "NON_EVALUABLE_OR_ERROR", "CHECKPOINT_READOUT_FAILED",
}


class AuditError(RuntimeError):
    """Failure with fixed public-safe text, never a payload or private path."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_hash(value: Any) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8"))


def filename(logical_id: str, suffix: str = ".json") -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", logical_id).strip(".-") or "item"
    return f"{component[:72]}--{digest(logical_id.encode('utf-8'))[:16]}{suffix}"


def enum(value: Any, allowed: set[str]) -> str:
    if value is None:
        return "MISSING"
    require(isinstance(value, str) and value in allowed, "Unexpected metadata enum")
    return value


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def native(path: Path) -> Path:
    if os.name != "nt" or str(path).startswith("\\\\?\\"):
        return path
    if str(path).startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + str(path)[2:])
    return Path("\\\\?\\" + str(path))


@dataclass(frozen=True)
class FrozenIdentity:
    """Production identities are immutable CLI constants; fixtures may instantiate this type."""

    run_id: str = "v2-run-498f9d84f5b45d26d732"
    config_hash: str = "bbab807d3ead00739dcab7482eecb6df763e72fa0308c19b12f0a6e06f30a6b3"
    source_commit: str = "926d076f3baf7b5d31328f8865bd70cefe8396c1"
    source_tree: str = "d44acd46935a0a51911f0cc57c07d776f98f242e"
    manifest_hash: str = "8eb97c603f9731a8f482cfc855ba048d03613de420c846eba5a11f28fa80dc22"
    split_hash: str = "bad1088f2857f421326c2ce4d7ff94d0f913fef2df4ddc3d8ab5b1edd2052685"
    protocol_id: str = "v2-recovery-activation-probe-20261002-a1"
    split_counts: tuple[tuple[str, int], ...] = (("train", 96), ("validation", 32), ("test", 32))
    positions: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    rollouts: int = 6


class VerifiedReader:
    def __init__(self, extraction_root: Path, files: dict[str, Any], allowed: set[str]):
        self.root = extraction_root.resolve(strict=True)
        require(self.root.is_dir(), "Extraction root is not a directory")
        require(isinstance(files, dict), "Malformed inventory file map")
        self.files = files
        self.allowed = set(allowed)
        self.consumed: dict[str, dict[str, Any]] = {}

    def read_json(self, relative: str) -> dict[str, Any]:
        require(relative in self.allowed, "Input outside train/validation audit allowlist")
        logical_path = PurePosixPath(relative)
        require(not logical_path.is_absolute() and ".." not in logical_path.parts
                and "\\" not in relative and ":" not in relative and str(logical_path) == relative,
                "Unsafe inventory-relative path")
        require(relative in self.files, "Expected input missing from inventory")
        target = native(self.root / relative).resolve(strict=True)
        require(target.is_relative_to(native(self.root).resolve(strict=True)), "Input escapes extraction root")
        expected = self.files[relative]
        require(isinstance(expected, dict) and type(expected.get("bytes")) is int
                and expected["bytes"] >= 0 and isinstance(expected.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is not None,
                "Malformed inventory entry")
        content = target.read_bytes()
        require(len(content) == expected["bytes"] and digest(content) == expected["sha256"],
                "Input size or SHA256 mismatch")
        self.consumed[relative] = {"bytes": len(content), "sha256": expected["sha256"]}
        parsed = json.loads(content)
        require(isinstance(parsed, dict), "Audited JSON is not an object")
        return parsed


def identity_paths(contract: FrozenIdentity) -> tuple[str, str, str]:
    return (PREFIX + "manifests/v2_deepmath_selection.json",
            PREFIX + "manifests/v2_deepmath_grouped_split.json",
            PREFIX + "runs/" + contract.run_id + "/run_manifest.json")


def receipt_identity(receipt: dict[str, Any], contract: FrozenIdentity, kind: str,
                     problem: str, split_name: str, rollout_id: str,
                     checkpoint_id: str | None = None, position: int | None = None) -> None:
    require(split_name in {"train", "validation"}, "Test receipt access forbidden")
    expected = {
        "record_type": kind, "config_hash": contract.config_hash, "run_id": contract.run_id,
        "manifest_hash": contract.manifest_hash, "split_hash": contract.split_hash,
        "source_git_commit": contract.source_commit, "study_protocol_id": contract.protocol_id,
        "problem_id": problem, "split": split_name, "rollout_id": rollout_id,
    }
    if checkpoint_id is not None:
        expected.update(checkpoint_id=checkpoint_id, checkpoint_token=position)
    require(all(receipt.get(key) == value for key, value in expected.items()),
            "Train/validation receipt identity mismatch")


def audit(reader: VerifiedReader, contract: FrozenIdentity = FrozenIdentity()) -> dict[str, Any]:
    manifest, split, run = (reader.read_json(path) for path in identity_paths(contract))
    require(canonical_hash(manifest) == contract.manifest_hash
            and canonical_hash(split) == contract.split_hash, "Frozen manifest identity mismatch")
    expected_run = {"config_hash": contract.config_hash, "manifest_hash": contract.manifest_hash,
                    "split_hash": contract.split_hash, "run_id": contract.run_id,
                    "study_protocol_id": contract.protocol_id}
    require(all(run.get(key) == value for key, value in expected_run.items()),
            "Frozen run identity mismatch")
    require(run.get("source_git_identity") == {"commit": contract.source_commit,
            "tree": contract.source_tree, "is_clean": True}, "Frozen source identity mismatch")
    require(run.get("selection_manifest") == manifest and run.get("split_manifest") == split,
            "Run embeds different manifests")
    require(split.get("selection_manifest_sha256") == contract.manifest_hash,
            "Split selection identity mismatch")
    assignments, records = split.get("assignments"), manifest.get("records")
    require(isinstance(assignments, list) and isinstance(records, list), "Malformed manifest rows")
    expected_count = sum(count for _, count in contract.split_counts)
    require(len(records) == expected_count and len(assignments) == expected_count,
            "Frozen problem count mismatch")
    record_ids = {record.get("problem_id") for record in records if isinstance(record, dict)}
    assignment_ids = {row.get("problem_id") for row in assignments if isinstance(row, dict)}
    require(len(record_ids) == expected_count and record_ids == assignment_ids
            and all(isinstance(problem, str) for problem in record_ids),
            "Frozen assignment identity mismatch")
    require(Counter(row.get("split") for row in assignments) == dict(contract.split_counts),
            "Frozen split quotas mismatch")
    raw_root = PREFIX + "runs/" + contract.run_id + "/raw/"
    for row in assignments:
        if row["split"] not in {"train", "validation"}:
            continue
        for rollout_index in range(contract.rollouts):
            rollout_id = f"v2-rollout:{contract.manifest_hash[:16]}:{row['problem_id']}:rollout-{rollout_index}"
            reader.allowed.add(raw_root + "terminal_readouts/" + filename("terminal-readout:" + rollout_id))
            for position in contract.positions:
                checkpoint_id = f"v2-checkpoint:{rollout_id}:token-{position}"
                reader.allowed.add(raw_root + "checkpoint_readouts/" + filename("checkpoint-readout:" + checkpoint_id))
                reader.allowed.add(raw_root + "activation_receipts/" + filename("activation:" + checkpoint_id))

    results: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        selected = sorted(row["problem_id"] for row in assignments if row["split"] == split_name)
        counters: dict[str, Counter] = defaultdict(Counter)
        primary_problems = {label: set() for label in PRIMARY}
        paired_problems = {label: set() for label in PRIMARY}
        for problem in selected:
            for rollout_index in range(contract.rollouts):
                rollout_id = f"v2-rollout:{contract.manifest_hash[:16]}:{problem}:rollout-{rollout_index}"
                terminal = reader.read_json(raw_root + "terminal_readouts/" + filename("terminal-readout:" + rollout_id))
                receipt_identity(terminal, contract, "V2_TERMINAL_READOUT", problem, split_name, rollout_id)
                terminal_status = enum(terminal.get("status"), STATUS)
                terminal_term = enum(mapping(terminal.get("readout")).get("termination_status"), TERMINATION)
                terminal_eval = enum(mapping(terminal.get("evaluation")).get("status"), EVALUATION)
                terminal_extraction = enum(mapping(terminal.get("extraction")).get("status"), EXTRACTION)
                counters["terminal_status"][terminal_status] += 1
                counters["terminal_termination"][terminal_term] += 1
                counters["terminal_extraction"][terminal_extraction] += 1
                counters["terminal_evaluation"][terminal_eval] += 1
                for position in contract.positions:
                    checkpoint_id = f"v2-checkpoint:{rollout_id}:token-{position}"
                    checkpoint = reader.read_json(raw_root + "checkpoint_readouts/" + filename("checkpoint-readout:" + checkpoint_id))
                    receipt_identity(checkpoint, contract, "V2_CHECKPOINT_READOUT", problem,
                                     split_name, rollout_id, checkpoint_id, position)
                    status = enum(checkpoint.get("status"), STATUS)
                    label = enum(checkpoint.get("transition_label"), LABELS)
                    counters["checkpoint_status"][status] += 1
                    counters["checkpoint_availability"][enum(checkpoint.get("availability_status"), AVAILABILITY)] += 1
                    counters["stored_transition_labels"][label] += 1
                    eligibility = mapping(checkpoint.get("analytic_eligibility"))
                    reasons = eligibility.get("reason_codes", [])
                    require(isinstance(reasons, list), "Malformed eligibility reason list")
                    validated_reasons = [enum(reason, REASONS) for reason in reasons]
                    counters["eligibility_reason_codes"].update(validated_reasons)
                    counters["eligibility_reason_combinations"]["+".join(sorted(validated_reasons)) or "NONE"] += 1
                    if status != "COMPLETED":
                        continue
                    readout = mapping(checkpoint.get("readout"))
                    require(bool(readout) and isinstance(readout.get("observable_scores"), dict),
                            "Completed checkpoint lacks feature container")
                    checkpoint_term = enum(readout.get("termination_status"), TERMINATION)
                    checkpoint_eval = enum(mapping(checkpoint.get("evaluation")).get("status"), EVALUATION)
                    checkpoint_extraction = enum(mapping(checkpoint.get("extraction")).get("status"), EXTRACTION)
                    counters["checkpoint_termination"][checkpoint_term] += 1
                    counters["checkpoint_terminal_termination_pairs"][checkpoint_term + "/" + terminal_term] += 1
                    counters["checkpoint_extraction"][checkpoint_extraction] += 1
                    counters["checkpoint_evaluation"][checkpoint_eval] += 1
                    counters["loader_feature_candidates"]["COMPLETED_WITH_OBSERVABLE_CONTAINER"] += 1
                    if label not in PRIMARY:
                        counters["primary_enrollment_exclusions"]["MISSING_TRANSITION_LABEL" if label == "MISSING" else "NON_PRIMARY_" + label] += 1
                        continue
                    expected_terminal = "CORRECT" if label == "W_TO_C" else "INCORRECT"
                    require(eligibility.get("eligible") is True and not reasons
                            and terminal_status == "COMPLETED" and terminal_term == "EOS"
                            and checkpoint_term == "EOS" and checkpoint_eval == "INCORRECT"
                            and terminal_eval == expected_terminal
                            and checkpoint_extraction == "EXTRACTED" and terminal_extraction == "EXTRACTED",
                            "Primary label inconsistent with stored eligibility")
                    counters["primary_rows"][label] += 1
                    primary_problems[label].add(problem)
                    activation_path = raw_root + "activation_receipts/" + filename("activation:" + checkpoint_id)
                    tensor_path = raw_root + "activation_tensors/" + filename("activation-tensor:" + checkpoint_id, ".pt")
                    if activation_path not in reader.files:
                        counters["primary_activation_metadata"]["RECEIPT_MISSING"] += 1
                        continue
                    activation = reader.read_json(activation_path)
                    receipt_identity(activation, contract, "V2_CHECKPOINT_ACTIVATION", problem,
                                     split_name, rollout_id, checkpoint_id, position)
                    activation_status = enum(activation.get("status"), STATUS)
                    counters["primary_activation_metadata"][activation_status] += 1
                    if activation_status == "COMPLETED" and tensor_path in reader.files:
                        counters["primary_activation_metadata_present_rows"][label] += 1
                        paired_problems[label].add(problem)
        row_floor, problem_floor = (50, 15) if split_name == "train" else (20, 8)
        counts = {key: dict(sorted(value.items())) for key, value in sorted(counters.items())}
        counts["primary_rows"] = {label: counters["primary_rows"][label] for label in sorted(PRIMARY)}
        counts["primary_problem_counts"] = {label: len(primary_problems[label]) for label in sorted(PRIMARY)}
        counts["primary_activation_metadata_present_problem_counts"] = {label: len(paired_problems[label]) for label in sorted(PRIMARY)}
        counts["frozen_floors"] = {"rows_per_class": row_floor, "problems_per_class": problem_floor}
        counts["raw_primary_floors_pass"] = all(counters["primary_rows"][label] >= row_floor
                                              and len(primary_problems[label]) >= problem_floor for label in PRIMARY)
        counts["problem_count"] = len(selected)
        counts["intended_terminal_readouts"] = len(selected) * contract.rollouts
        counts["intended_checkpoint_slots"] = len(selected) * contract.rollouts * len(contract.positions)
        results[split_name] = counts
    return results


def checked_bootstrap(root: Path, name: str, expected_size: int, expected_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    target = native(root / name).resolve(strict=True)
    require(target.is_relative_to(native(root).resolve(strict=True)), "Bootstrap input escapes extraction root")
    content = target.read_bytes()
    require(len(content) == expected_size and digest(content) == expected_sha,
            "Bootstrap input size or SHA256 mismatch")
    parsed = json.loads(content)
    require(isinstance(parsed, dict), "Bootstrap JSON is not an object")
    return parsed, {"bytes": len(content), "sha256": expected_sha}


def validate_output(path: Path, repo_root: Path, extraction_root: Path) -> Path:
    output = path.resolve()
    namespace = (repo_root / PRIVATE_NAMESPACE).resolve()
    require(namespace.is_relative_to(repo_root.resolve()), "Private namespace escapes repository")
    require(output.is_relative_to(namespace) and output != namespace and output.suffix == ".json",
            "Private output must be a new JSON file in the measurement-reproduction namespace")
    require(not output.is_relative_to(extraction_root.resolve()), "Private output overlaps verified extraction")
    require(not output.exists(), "Private output already exists; write-once refusal")
    return output


def write_private_report(path: Path, payload: dict[str, Any]) -> str:
    content = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
    return digest(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-root", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--inventory-sha", required=True)
    args = parser.parse_args(argv)
    try:
        require(args.inventory_sha == INVENTORY_SHA, "Inventory authority must match frozen V2-A1")
        root = args.extraction_root.resolve(strict=True)
        output = validate_output(args.private_output, Path(__file__).resolve().parents[1], root)
        inventory, inventory_metadata = checked_bootstrap(root, "a1-scientific-inventory.json", INVENTORY_BYTES, INVENTORY_SHA)
        verification, verification_metadata = checked_bootstrap(root, "backup-verification.json", VERIFICATION_BYTES, VERIFICATION_SHA)
        contract = FrozenIdentity()
        require(all(verification.get(key) is True for key in ("all_payload_hashes_match",
                "scientific_inventory_complete", "operational_inventory_complete")),
                "Recovery verification incomplete")
        require(verification.get("source_commit") == contract.source_commit
                and verification.get("source_tree") == contract.source_tree
                and verification.get("config_hash") == contract.config_hash,
                "Recovery provenance mismatch")
        reader = VerifiedReader(root, inventory.get("files"), set(identity_paths(contract)))
        results = audit(reader, contract)
        public = {
            "schema_version": 1, "record_type": "V2A1_PORTABLE_ELIGIBILITY_REPRODUCTION_V1",
            "config_hash": contract.config_hash, "run_id": contract.run_id,
            "source_commit": contract.source_commit, "source_tree": contract.source_tree,
            "manifest_hash": contract.manifest_hash, "split_hash": contract.split_hash,
            "inventory_sha256": INVENTORY_SHA, "verification_report_sha256": VERIFICATION_SHA,
            "scope": "Stored train/validation metadata only; no test receipt bodies, answer text, numeric confidence, tensor loading, evaluator, model, fitting or network",
            "activation_matrix_validity_reassessed": False,
            "splits": results,
        }
        consumed = {"a1-scientific-inventory.json": inventory_metadata,
                    "backup-verification.json": verification_metadata, **reader.consumed}
        payload = {**public, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "helper_sha256": digest(Path(__file__).read_bytes()),
                   "consumed_input_count": len(consumed), "consumed_inputs": consumed}
        report_hash = write_private_report(output, payload)
        print(json.dumps({**public, "private_report_sha256": report_hash,
                          "consumed_input_count": len(consumed)}, sort_keys=True, allow_nan=False))
        return 0
    except AuditError as error:
        print(json.dumps({"status": "AUDIT_FAILED", "reason": str(error)}))
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"status": "AUDIT_FAILED", "reason": "Invalid or unavailable audited input"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
