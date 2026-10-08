"""Verified CPU-only adapter for a separately frozen completion-forecasting study.

No model, evaluator, network, terminal readout, original TEST body, or original
artifact write is permitted. Audit constructs prefix-only features before
joining future labels. Prepare is a distinct, write-once tensor-loading stage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from reasoning_recovery.v2_features import (  # Pure, no artifact/model imports.
    CheckpointReadoutFeatureInput, build_v2_feature_rows,
    STRUCTURAL_NUMERIC_FEATURES, OBSERVABLE_NUMERIC_FEATURES,
)

PROTOCOL = "completion-forecasting-v1-20261008"
ORIGINAL_SOURCE = "926d076f3baf7b5d31328f8865bd70cefe8396c1"
CONFIG = "bbab807d3ead00739dcab7482eecb6df763e72fa0308c19b12f0a6e06f30a6b3"
MANIFEST = "8eb97c603f9731a8f482cfc855ba048d03613de420c846eba5a11f28fa80dc22"
SPLIT = "bad1088f2857f421326c2ce4d7ff94d0f913fef2df4ddc3d8ab5b1edd2052685"
INVENTORY = "b7782d22c207e8463fd4b41ecf772282a69f305ddc517c44239805df86198775"
RUNTIME_FILE = "f5b4a26040b204fc0f1f671421f09d48ab7e3b39356b4c9335e9cc28b017c92c"
RUN = "v2-run-498f9d84f5b45d26d732"
PREFIX = "reasoning-recovery-probes/artifacts/v2a1/"
RUN_ROOT = PREFIX + "runs/" + RUN + "/"
FEATURE_NAMES = STRUCTURAL_NUMERIC_FEATURES + OBSERVABLE_NUMERIC_FEATURES
PURE_FEATURE_SOURCE = "ca41fbac83225ea2fc1051ecd29b6887ef2bcb11fb64f2e642c45b141a28aa34"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)  # Never include private payloads in errors.


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_sha(value):
    return sha(canonical(value).encode("utf-8"))


def file_sha(path):
    digest = hashlib.sha256()
    with native(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native(path):
    resolved = Path(path).resolve()
    return Path("\\\\?\\" + str(resolved)) if os.name == "nt" else resolved


def utc():
    return datetime.now(timezone.utc).isoformat()


def filename(logical, suffix=".json"):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", logical).strip(".-")[:72] or "item"
    return safe + "--" + sha(logical.encode())[:16] + suffix


def write_once(path, value):
    with native(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


class VerifiedReader:
    """Allowlisted recovered payloads: validate bytes before decoding JSON."""

    def __init__(self, root, inventory_sha):
        self.root = Path(root).resolve()
        require(inventory_sha == INVENTORY, "Unapproved scientific inventory")
        content = native(self.root / "a1-scientific-inventory.json").read_bytes()
        require(sha(content) == inventory_sha, "Scientific inventory changed")
        self.inventory = json.loads(content)["files"]
        self.consumed = {}

    def read_bytes(self, relative):
        require(relative.startswith(PREFIX), "Payload outside original scientific root")
        require("/raw/terminal_readouts/" not in relative, "Terminal readout access forbidden")
        require(relative in self.inventory, "Required payload absent from inventory")
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), "Payload escapes extraction")
        content = native(path).read_bytes()
        expected = self.inventory[relative]
        require(len(content) == expected["bytes"] and sha(content) == expected["sha256"], "Payload hash or size changed")
        self.consumed[relative] = dict(expected)
        return content

    def read_json(self, relative):
        return json.loads(self.read_bytes(relative))

    def exists(self, relative):
        return relative in self.inventory


def token_ids(value):
    require(isinstance(value, list), "Token sequence is not a list")
    require(all(type(t) is int and t >= 0 for t in value), "Token sequence is invalid")
    return value


def first_marker(tokens, marker):
    require(bool(marker), "Empty close marker")
    return next((i for i in range(len(tokens) - len(marker) + 1) if tokens[i:i + len(marker)] == marker), None)


def completion_label(tokens, marker, eos, termination, recorded_boundary):
    """Labels use future tokens only here, never in predictor construction."""
    require(len(tokens) <= 4096, "Base exceeds original reasoning cap")
    close = first_marker(tokens, marker)
    first_eos = next((i for i, t in enumerate(tokens) if t in eos), None)
    valid_close = close is not None and (first_eos is None or first_eos >= close)
    if valid_close:
        require(recorded_boundary["status"] == "CLOSE_MARKER_FOUND", "Recorded closure boundary disagrees")
        require(recorded_boundary["close_marker_start"] == close, "Recorded closure position disagrees")
        require(recorded_boundary["terminal_prefix_sha256"] == object_sha(tokens[:close]), "Recorded closure prefix differs")
        require(close + len(marker) <= 4096, "Closure extends beyond original budget")
        if close < 512:
            return None, "ALREADY_CLOSED"
        return 1, "CLOSE_WITHIN_4096"
    if close is None and termination == "MAX_NEW_TOKENS" and len(tokens) == 4096:
        require(recorded_boundary["status"] == "CAP_PREFIX", "Recorded cap boundary disagrees")
        require(recorded_boundary["terminal_prefix_sha256"] == object_sha(tokens), "Recorded cap prefix differs")
        return 0, "NO_CLOSE_WITHIN_4096"
    require(recorded_boundary["status"] == "EARLY_NO_CLOSE", "Unrecognized original boundary")
    return None, "UNKNOWN_OR_EARLY_TERMINATION"


def lexical_counts(prefix):
    require(len(prefix) == 512, "Lexical predictor needs exact primary prefix")
    vector = [0] * 512
    for token in prefix[-128:]:
        require(type(token) is int and token >= 0, "Invalid lexical token")
        bucket = int(sha(("completion-forecasting-v1|trailing-token-hash|" + str(token)).encode())[:16], 16) % 512
        vector[bucket] += 1
    return vector


def fold_assignment(train_records):
    """Assign original TRAIN identities before label access; never rebalance."""
    output = {}
    for difficulty in (7.0, 8.0):
        ranked = sorted((r for r in train_records if r["difficulty"] == difficulty),
                        key=lambda r: (sha(("completion-forecasting-v1|fold|" + r["problem_id"]).encode()), r["problem_id"]))
        for index, record in enumerate(ranked):
            require(record["problem_id"] not in output, "Fold identities duplicated")
            output[record["problem_id"]] = index % 5
    require(len(output) == len(train_records), "Fold assignment omitted a TRAIN identity")
    return output


def finite_or_none(value):
    if value is None:
        return None
    require(type(value) in (int, float), "Non-numeric observable feature")
    return float(value) if math.isfinite(float(value)) else None


def observation(receipt, metadata, position):
    """Explicit projection: no evaluation, transition, terminal or seed fields."""
    completed = receipt.get("status") == "COMPLETED"
    scores = receipt["readout"]["observable_scores"] if completed else {}
    extraction = receipt.get("extraction", {}) if completed else {}
    answer = extraction.get("extracted_text") if extraction.get("status") == "EXTRACTED" else None
    answer = " ".join(answer.split()) if isinstance(answer, str) else None
    count = scores.get("answer_token_count")
    require(count is None or type(count) is int and count >= 0, "Invalid answer-token count")
    return CheckpointReadoutFeatureInput(
        row_id=receipt["checkpoint_id"], problem_id=metadata["problem_id"],
        rollout_id=receipt["rollout_id"], checkpoint_token=position,
        problem_token_count=metadata["problem_token_count"], difficulty=metadata["difficulty"], topic=metadata["topic"],
        mean_answer_token_logprob=finite_or_none(scores.get("mean_answer_token_logprob")),
        min_answer_token_logprob=finite_or_none(scores.get("min_answer_token_logprob")), answer_token_count=count,
        first_readout_token_entropy_nats=finite_or_none(scores.get("first_readout_token_entropy_nats")),
        first_readout_token_probability_margin=finite_or_none(scores.get("first_readout_token_probability_margin")),
        normalized_extracted_answer=answer or None, readout_available=completed,
    )


def support(rows, split):
    counts = Counter(r["target"] for r in rows)
    problems = {y: {r["problem_id"] for r in rows if r["target"] == y} for y in (0, 1)}
    by_problem = defaultdict(set)
    for row in rows:
        by_problem[row["problem_id"]].add(row["target"])
    row_floor, problem_floor = (50, 15) if split == "train" else (20, 8)
    return {"rows": len(rows), "class_counts": {str(y): counts[y] for y in (0, 1)},
            "class_problem_counts": {str(y): len(problems[y]) for y in (0, 1)},
            "unique_problems": len(by_problem), "mixed_label_problems": sum(len(v) == 2 for v in by_problem.values()),
            "difficulty_class_counts": {str(int(d)): {str(y): sum(r["target"] == y and r["difficulty"] == d for r in rows) for y in (0, 1)} for d in (7.0, 8.0)},
            "class_support_gate_passed": all(counts[y] >= row_floor and len(problems[y]) >= problem_floor for y in (0, 1)),
            "matched_interpretation_gate_passed": sum(len(v) == 2 for v in by_problem.values()) >= 8}


def check_identity(receipt, record, split, rollout, record_type, position=None):
    expected = {"record_type": record_type, "config_hash": CONFIG, "run_id": RUN,
                "manifest_hash": MANIFEST, "split_hash": SPLIT, "source_git_commit": ORIGINAL_SOURCE,
                "split": split, "problem_id": record["problem_id"], "rollout_id": rollout}
    if position is not None:
        expected.update(checkpoint_id=f"v2-checkpoint:{rollout}:token-{position}", checkpoint_token=position)
    if record_type == "V2_BASE_ROLLOUT":
        expected.update(source_index=record["source_index"], difficulty=record["difficulty"], topic=record["topic"], question_sha256=record["question_sha256"])
    require(all(receipt.get(key) == value for key, value in expected.items()), "Original receipt identity mismatch")


def contracts(reader, split):
    manifest = reader.read_json(PREFIX + "manifests/v2_deepmath_selection.json")
    assignments = reader.read_json(PREFIX + "manifests/v2_deepmath_grouped_split.json")
    require(object_sha(manifest) == MANIFEST and object_sha(assignments) == SPLIT, "Original cohort/split identity differs")
    require(Counter(a["split"] for a in assignments["assignments"]) == {"train": 96, "validation": 32, "test": 32}, "Original split quotas differ")
    ids = {a["problem_id"] for a in assignments["assignments"] if a["split"] == split}
    train_ids = {a["problem_id"] for a in assignments["assignments"] if a["split"] == "train"}
    train_records = [r for r in manifest["records"] if r["problem_id"] in train_ids]
    require(len(train_records) == 96, "Fold cohort differs from original TRAIN")
    folds = fold_assignment(train_records)
    records = [r for r in manifest["records"] if r["problem_id"] in ids]
    require(len(records) == (96 if split == "train" else 32) and len(ids) == len(records), "Requested split membership differs")
    runtime_bytes = reader.read_bytes(RUN_ROOT + "runtime_contract.json")
    require(sha(runtime_bytes) == RUNTIME_FILE, "Original runtime identity differs")
    runtime = json.loads(runtime_bytes)
    require(runtime["run_id"] == RUN and runtime["config_hash"] == CONFIG, "Original runtime/config binding differs")
    require(runtime["close_marker_token_ids"] == [151668] and runtime["readout_cue_token_ids"] == [151668, 271], "Original controls differ")
    require(set(runtime["runtime"]["resolved_generation_eos_token_ids"]) == {151645, 151643}, "Original EOS set differs")
    return sorted(records, key=lambda r: r["problem_id"]), runtime, folds


def audit(args, reader):
    records, runtime, folds = contracts(reader, args.split)
    cue, marker = runtime["readout_cue_token_ids"], runtime["close_marker_token_ids"]
    observations, pending, links, lexical = [], [], {}, {}
    exclusions, readout_statuses, nonfinite_features = Counter(), Counter(), Counter()
    for record in records:
        for index in range(6):
            rollout = f"v2-rollout:{MANIFEST[:16]}:{record['problem_id']}:rollout-{index}"
            base_relative = RUN_ROOT + "raw/rollouts/" + filename(rollout)
            base = reader.read_json(base_relative)
            check_identity(base, record, args.split, rollout, "V2_BASE_ROLLOUT")
            if base["status"] != "COMPLETED":
                exclusions["BASE_" + str(base["status"])] += 1
                continue
            generation, prompt = base["generation"], base["prompt"]
            prompt_ids, generated = token_ids(generation["input_token_ids"]), token_ids(generation["generated_token_ids"])
            require(prompt_ids == prompt["prompt_token_ids"] and object_sha(prompt_ids) == prompt["prompt_token_ids_sha256"], "Original prompt hash differs")
            require(object_sha(prompt_ids) == generation["input_token_ids_sha256"], "Original generation prompt hash differs")
            require(object_sha(generated) == generation["generated_token_ids_sha256"] and len(generated) == generation["generated_token_count"], "Original generated sequence differs")
            if len(generated) < 512 or first_marker(generated[:512], marker) is not None:
                exclusions["PRIMARY_PREFIX_UNAVAILABLE"] += 1
                continue
            metadata = {"problem_id": record["problem_id"], "problem_token_count": prompt["problem_token_count"], "difficulty": record["difficulty"], "topic": record["topic"]}
            require(type(metadata["problem_token_count"]) is int and metadata["problem_token_count"] >= 0, "Problem token count differs")
            current = f"v2-checkpoint:{rollout}:token-512"
            for position in (128, 256, 512):
                checkpoint = f"v2-checkpoint:{rollout}:token-{position}"
                relative = RUN_ROOT + "raw/checkpoint_readouts/" + filename("checkpoint-readout:" + checkpoint)
                receipt = reader.read_json(relative)
                check_identity(receipt, record, args.split, rollout, "V2_CHECKPOINT_READOUT", position)
                readout_statuses[str(position) + "/" + receipt["status"]] += 1
                if receipt["status"] == "COMPLETED":
                    prefix = generated[:position]
                    expected_input = prompt_ids + prefix + cue
                    require(receipt["saved_prefix_sha256"] == object_sha(prefix) and receipt["saved_prefix_token_count"] == position, "Saved readout prefix differs")
                    require(receipt["readout_input"]["input_token_ids"] == expected_input and receipt["readout_input"]["input_sha256"] == object_sha(expected_input), "Readout input includes changed/future IDs")
                    require(receipt["readout_input"]["cue_token_ids"] == cue and receipt["readout_input"]["cue_sha256"] == object_sha(cue), "Readout cue differs")
                    require(receipt["readout"]["input_token_ids_sha256"] == object_sha(expected_input), "Readout runtime input differs")
                    scores = receipt["readout"]["observable_scores"]
                    for feature_name in OBSERVABLE_NUMERIC_FEATURES[:5]:
                        value = scores.get(feature_name)
                        if type(value) in (int, float) and not math.isfinite(float(value)):
                            nonfinite_features[str(position) + "/" + feature_name] += 1
                observations.append(observation(receipt, metadata, position))
            prefix = generated[:512]
            lexical[current] = lexical_counts(prefix)
            activation_relative = RUN_ROOT + "raw/activation_receipts/" + filename("activation:" + current)
            activation = reader.read_json(activation_relative)
            check_identity(activation, record, args.split, rollout, "V2_CHECKPOINT_ACTIVATION", 512)
            link = {"status": activation["status"], "receipt_relative": activation_relative}
            if activation["status"] == "COMPLETED":
                inputs = activation["activation_input"]
                exact_input = prompt_ids + prefix
                require(inputs["input_policy"] == "prompt-plus-exact-prefix-no-readout-cue-v1", "Activation policy differs")
                require(inputs["prompt_sha256"] == object_sha(prompt_ids) and inputs["reasoning_prefix_sha256"] == object_sha(prefix), "Activation prompt/prefix differs")
                require(inputs["input_sha256"] == object_sha(exact_input) and inputs["input_token_count"] == len(exact_input), "Activation input includes cue/future IDs")
                require(inputs["reasoning_prefix_token_count"] == 512 and activation["saved_prefix_sha256"] == object_sha(prefix), "Activation coordinate differs")
                require(activation["saved_prefix_token_count"] == 512, "Activation saved coordinate differs")
                matrix = activation["activation"]
                require(matrix["input_sha256"] == inputs["input_sha256"], "Activation input binding differs")
                tensor_relative = RUN_ROOT + "raw/activation_tensors/" + filename("activation-tensor:" + current, ".pt")
                require(activation["tensor_filename"] == Path(tensor_relative).name, "Tensor filename differs")
                if reader.exists(tensor_relative):
                    require(reader.inventory[tensor_relative]["sha256"] == activation["tensor_serialized_sha256"], "Tensor serialized binding differs")
                    link.update(tensor_relative=tensor_relative, tensor_sha256=activation["tensor_serialized_sha256"], matrix_sha256=matrix["matrix_sha256"], input_sha256=inputs["input_sha256"])
                else:
                    link["status"] = "MISSING_TENSOR"
            links[current] = link
            pending.append((current, record["problem_id"], record["difficulty"], generation, base["reasoning_boundary"]))
    feature_rows = [asdict(r) for r in build_v2_feature_rows(observations) if r.checkpoint_token == 512]
    require(len(feature_rows) == len(pending), "Primary feature row count differs")
    feature_payload = {"record_type": "COMPLETION_FORECASTING_PREFIX_FEATURES_V1", "features": feature_rows, "lexical": lexical, "fold_assignment_all_original_train": folds, "numeric_feature_names": list(FEATURE_NAMES), "future_labels_joined": False, "terminal_or_evaluation_fields_used": False}
    write_once(args.output_dir / "features-before-labels.json", feature_payload)
    targets, label_statuses = [], Counter()
    for current, problem, difficulty, generation, boundary in pending:
        target, status = completion_label(generation["generated_token_ids"], marker, runtime["runtime"]["resolved_generation_eos_token_ids"], generation["termination_status"], boundary)
        label_statuses[status] += 1
        if target is None:
            exclusions[status] += 1
            continue
        targets.append({"row_id": current, "problem_id": problem, "difficulty": difficulty, "target": target, "label_status": status, "activation_link": links[current]})
    linked = [r for r in targets if r["activation_link"]["status"] == "COMPLETED"]
    payload = {"record_type": "COMPLETION_FORECASTING_DATA_AUDIT_V1", "protocol_id": PROTOCOL, "split": args.split, "completed_at_utc": utc(), "features_sha256": file_sha(args.output_dir / "features-before-labels.json"), "feature_count_before_label_join": len(feature_rows), "targets": targets, "intended_rollouts": len(records) * 6, "risk_set": support(targets, args.split), "activation_linked_support": support(linked, args.split), "exclusions": dict(exclusions), "label_statuses": dict(label_statuses), "readout_statuses": dict(readout_statuses), "nonfinite_feature_values_encoded_missing": dict(nonfinite_features), "consumed_inputs": reader.consumed, "tensor_values_loaded": 0, "gpu_calls": 0, "test_bodies_read": 0, "gold_or_evaluator_fields_used": False, "source_protocol_lock": lock_metadata(args)}
    write_once(args.output_dir / "audit-report.json", payload)
    print(canonical({"status": "AUDIT_COMPLETE", "split": args.split, "risk_set": payload["risk_set"], "activation_linked_support": payload["activation_linked_support"], "exclusions": dict(exclusions), "audit_sha256": file_sha(args.output_dir / "audit-report.json"), "gpu_calls": 0}), flush=True)


def matrix_hash(matrix, torch):
    header = canonical({"shape": list(matrix.shape), "dtype": "bfloat16", "representation": "post-block-last-reasoning-token-v1"}).encode()
    return sha(header + b"\0" + matrix.detach().contiguous().view(torch.uint8).numpy().tobytes())


def prepare(args, reader):
    require(args.audit_sha256 is not None, "Prepare needs immutable audit SHA")
    report_bytes = native(args.output_dir / "audit-report.json").read_bytes()
    require(sha(report_bytes) == args.audit_sha256, "Data audit changed")
    report = json.loads(report_bytes)
    require(report["split"] == args.split and report["source_protocol_lock"] == lock_metadata(args), "Audit/source/protocol binding differs")
    require(report["activation_linked_support"]["class_support_gate_passed"], "Aggregate class-support gate blocks tensor load")
    feature_bytes = native(args.output_dir / "features-before-labels.json").read_bytes()
    require(sha(feature_bytes) == report["features_sha256"], "Prefix features changed")
    feature_payload = json.loads(feature_bytes)
    features = {r["row_id"]: r for r in feature_payload["features"]}
    require(tuple(feature_payload["numeric_feature_names"]) == FEATURE_NAMES, "Feature whitelist differs")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch  # Never call a CUDA API or instantiate a model.
    require(importlib.metadata.version("numpy") == "2.2.6" and importlib.metadata.version("torch") == "2.11.0+cu128", "CPU export library versions differ")
    matrices, accepted, invalid = [], [], Counter()
    for row in report["targets"]:
        link = row["activation_link"]
        if link["status"] != "COMPLETED":
            invalid[link["status"]] += 1
            continue
        content = reader.read_bytes(link["tensor_relative"])
        require(sha(content) == link["tensor_sha256"], "Tensor serialized hash changed")
        matrix = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        if not isinstance(matrix, torch.Tensor) or matrix.device.type != "cpu" or matrix.dtype != torch.bfloat16 or tuple(matrix.shape) != (28, 2048):
            invalid["INVALID_MATRIX_CONTRACT"] += 1
            continue
        require(matrix_hash(matrix, torch) == link["matrix_sha256"], "Tensor matrix-content hash changed")
        if not bool(torch.isfinite(matrix).all().item()):
            invalid["NONFINITE_MATRIX"] += 1
            continue
        matrices.append(matrix.to(dtype=torch.float32).numpy().copy())
        accepted.append(row)
    enrolled_support = support(accepted, args.split)
    runtime = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "numpy_source": str(Path(np.__file__).resolve()), "torch_source": str(Path(torch.__file__).resolve()), "cuda_visible_devices": "", "tensor_load": "weights_only=True,map_location=cpu", "bf16_to_float32": "lossless"}
    prepared = {"record_type": "COMPLETION_FORECASTING_CPU_PREPARED_V1", "protocol_id": PROTOCOL, "split": args.split, "completed_at_utc": utc(), "audit_sha256": args.audit_sha256, "support": enrolled_support, "activation_exclusions": dict(invalid), "runtime": runtime, "consumed_tensors": reader.consumed, "source_protocol_lock": lock_metadata(args), "gpu_calls": 0, "test_bodies_read": 0}
    if enrolled_support["class_support_gate_passed"]:
        ordered = [features[row["row_id"]] for row in accepted]
        numeric = [[r[name] if r[name] is not None else np.nan for name in FEATURE_NAMES] for r in ordered]
        with native(args.output_dir / "data.npz").open("xb") as stream:
            np.savez_compressed(stream, activations=np.stack(matrices), numeric=np.asarray(numeric, dtype=np.float64), lexical=np.asarray([feature_payload["lexical"][r["row_id"]] for r in accepted], dtype=np.float64), labels=np.asarray([r["target"] for r in accepted], dtype=np.int8), row_ids=np.asarray([r["row_id"] for r in accepted]), groups=np.asarray([r["problem_id"] for r in accepted]), fold_ids=np.asarray([feature_payload["fold_assignment_all_original_train"][r["problem_id"]] if args.split == "train" else -1 for r in accepted], dtype=np.int8), topics=np.asarray([r["topic"] for r in ordered]), difficulty=np.asarray([r["difficulty"] for r in accepted], dtype=np.float64), numeric_feature_names=np.asarray(FEATURE_NAMES))
        prepared["data_npz_sha256"] = file_sha(args.output_dir / "data.npz")
        prepared["data_npz_bytes"] = native(args.output_dir / "data.npz").stat().st_size
        prepared["status"] = "CPU_DATA_READY"
    else:
        prepared["status"] = "ACTIVATION_VALID_CLASS_SUPPORT_BLOCK"
    write_once(args.output_dir / "prepare-report.json", prepared)
    print(canonical({"status": prepared["status"], "split": args.split, "support": enrolled_support, "activation_exclusions": dict(invalid), "prepare_report_sha256": file_sha(args.output_dir / "prepare-report.json"), "gpu_calls": 0}), flush=True)


def lock_metadata(args):
    return {"protocol_sha256": args.protocol_sha256, "helper_sha256": args.helper_sha256, "pure_feature_source_sha256": PURE_FEATURE_SOURCE, "inventory_sha256": args.inventory_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("audit", "prepare"))
    parser.add_argument("--extraction-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "validation"))
    parser.add_argument("--inventory-sha", required=True)
    parser.add_argument("--protocol-path", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--helper-sha256", required=True)
    parser.add_argument("--audit-sha256")
    args = parser.parse_args()
    require(file_sha(Path(__file__)) == args.helper_sha256, "Helper not source-locked")
    require(file_sha(args.protocol_path) == args.protocol_sha256, "Protocol not frozen")
    require(file_sha(REPO / "src/reasoning_recovery/v2_features.py") == PURE_FEATURE_SOURCE, "Original feature implementation changed")
    args.output_dir = args.output_dir.resolve()
    require(args.output_dir.is_relative_to((REPO / "artifacts/completion-forecasting/v1").resolve()), "Output must use new private namespace")
    require(not args.output_dir.is_relative_to(args.extraction_root.resolve()), "Output cannot alter recovered extraction")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intent_path = args.output_dir / (args.mode + "-intent.json")
    write_once(intent_path, {"record_type": "COMPLETION_FORECASTING_CPU_INTENT_V1", "mode": args.mode, "split": args.split, "created_at_utc": utc(), "source_protocol_lock": lock_metadata(args), "gpu_calls": 0, "test_bodies_authorized": False})
    try:
        reader = VerifiedReader(args.extraction_root, args.inventory_sha)
        if args.mode == "audit":
            audit(args, reader)
        else:
            prepare(args, reader)
    except Exception as error:
        write_once(args.output_dir / (args.mode + "-failure.json"), {"status": "CPU_STAGE_FAILED", "error_type": type(error).__name__, "message": str(error) if type(error) is RuntimeError else None, "created_at_utc": utc(), "no_automatic_retry": True})
        print(canonical({"status": "CPU_STAGE_FAILED", "mode": args.mode, "error_type": type(error).__name__}), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
