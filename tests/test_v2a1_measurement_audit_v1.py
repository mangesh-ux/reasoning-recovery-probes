"""Synthetic-only tests: no original artifacts or scientific execution."""
from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v2a1_measurement_audit_v1.py"
spec = importlib.util.spec_from_file_location("v2a1_measurement_audit_v1", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class MeasurementAuditTests(unittest.TestCase):
    def setUp(self):
        parent = SCRIPT.parents[1] / module.PRIVATE_NAMESPACE / "synthetic-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temporary.name)
        self.temporary.name = str(module.native(self.root))
        self.addCleanup(self.temporary.cleanup)
        self.files = {}

    def store(self, relative, payload):
        content = json.dumps(payload, sort_keys=True).encode("utf-8")
        target = module.native(self.root / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self.files[relative] = {"bytes": len(content), "sha256": module.digest(content)}

    def fixture(self):
        records = [{"problem_id": name} for name in ("synthetic-train-a", "synthetic-train-b", "synthetic-val", "synthetic-test")]
        manifest = {"records": records}
        manifest_hash = module.canonical_hash(manifest)
        split = {"selection_manifest_sha256": manifest_hash, "assignments": [
            {"problem_id": "synthetic-train-a", "split": "train"},
            {"problem_id": "synthetic-train-b", "split": "train"},
            {"problem_id": "synthetic-val", "split": "validation"},
            {"problem_id": "synthetic-test", "split": "test"},
        ]}
        contract = replace(module.FrozenIdentity(), run_id="synthetic-run",
                           manifest_hash=manifest_hash, split_hash=module.canonical_hash(split),
                           split_counts=(("train", 2), ("validation", 1), ("test", 1)),
                           positions=(512,), rollouts=1)
        run = {"config_hash": contract.config_hash, "manifest_hash": contract.manifest_hash,
               "split_hash": contract.split_hash, "run_id": contract.run_id,
               "study_protocol_id": contract.protocol_id,
               "source_git_identity": {"commit": contract.source_commit, "tree": contract.source_tree, "is_clean": True},
               "selection_manifest": manifest, "split_manifest": split}
        for relative, payload in zip(module.identity_paths(contract), (manifest, split, run)):
            self.store(relative, payload)
        raw = module.PREFIX + "runs/" + contract.run_id + "/raw/"
        checkpoint_paths = []
        for problem, split_name, terminal_eval, label in (
            ("synthetic-train-a", "train", "CORRECT", "W_TO_C"),
            ("synthetic-train-b", "train", "INCORRECT", "W_TO_W"),
            ("synthetic-val", "validation", "NON_EVALUABLE", None),
            ("synthetic-test", "test", "CORRECT", "W_TO_C"),
        ):
            rollout = f"v2-rollout:{contract.manifest_hash[:16]}:{problem}:rollout-0"
            shared = {"config_hash": contract.config_hash, "run_id": contract.run_id,
                      "manifest_hash": contract.manifest_hash, "split_hash": contract.split_hash,
                      "source_git_commit": contract.source_commit, "study_protocol_id": contract.protocol_id,
                      "problem_id": problem, "split": split_name, "rollout_id": rollout,
                      "generated_text": "PRIVATE_TEXT_MUST_NOT_ESCAPE"}
            terminal = {**shared, "record_type": "V2_TERMINAL_READOUT", "status": "COMPLETED",
                        "readout": {"termination_status": "EOS" if label else "MAX_NEW_TOKENS"},
                        "extraction": {"status": "EXTRACTED" if label else "MISSING_BOXED_ANSWER"},
                        "evaluation": {"status": terminal_eval}}
            self.store(raw + "terminal_readouts/" + module.filename("terminal-readout:" + rollout), terminal)
            checkpoint_id = f"v2-checkpoint:{rollout}:token-512"
            checkpoint = {**shared, "record_type": "V2_CHECKPOINT_READOUT", "checkpoint_id": checkpoint_id,
                          "checkpoint_token": 512, "status": "COMPLETED" if label else "UNAVAILABLE",
                          "availability_status": "AVAILABLE" if label else "AFTER_THINK_CLOSE",
                          "transition_label": label, "analytic_eligibility": {"eligible": bool(label),
                          "reason_codes": [] if label else ["AFTER_THINK_CLOSE"]}}
            if label:
                checkpoint.update(readout={"termination_status": "EOS", "observable_scores": {}},
                                  extraction={"status": "EXTRACTED"}, evaluation={"status": "INCORRECT"})
            checkpoint_path = raw + "checkpoint_readouts/" + module.filename("checkpoint-readout:" + checkpoint_id)
            self.store(checkpoint_path, checkpoint)
            checkpoint_paths.append(checkpoint_path)
            if label:
                self.store(raw + "activation_receipts/" + module.filename("activation:" + checkpoint_id),
                           {**shared, "record_type": "V2_CHECKPOINT_ACTIVATION", "checkpoint_id": checkpoint_id,
                            "checkpoint_token": 512, "status": "COMPLETED"})
                tensor_path = raw + "activation_tensors/" + module.filename("activation-tensor:" + checkpoint_id, ".pt")
                self.files[tensor_path] = {"bytes": 1, "sha256": "a" * 64}
        return contract, checkpoint_paths

    def test_reproduces_counts_without_test_or_tensor_reads_or_text_output(self):
        contract, checkpoint_paths = self.fixture()
        reader = module.VerifiedReader(self.root, self.files, set(module.identity_paths(contract)))
        result = module.audit(reader, contract)
        self.assertEqual(result["train"]["primary_rows"], {"W_TO_C": 1, "W_TO_W": 1})
        self.assertEqual(result["train"]["primary_problem_counts"], {"W_TO_C": 1, "W_TO_W": 1})
        self.assertEqual(result["train"]["checkpoint_termination"], {"EOS": 2})
        self.assertEqual(result["validation"]["checkpoint_status"], {"UNAVAILABLE": 1})
        self.assertEqual(result["validation"]["primary_rows"], {"W_TO_C": 0, "W_TO_W": 0})
        self.assertFalse(result["train"]["raw_primary_floors_pass"])
        self.assertNotIn(checkpoint_paths[-1], reader.allowed)
        self.assertTrue(all(not path.endswith(".pt") for path in reader.consumed))
        printed = json.dumps(result)
        self.assertNotIn("PRIVATE_TEXT_MUST_NOT_ESCAPE", printed)
        self.assertNotIn("synthetic-train", printed)
        with self.assertRaisesRegex(module.AuditError, "allowlist"):
            reader.read_json(checkpoint_paths[-1])

    def test_payload_tampering_is_rejected_before_json_parse(self):
        self.store("payload.json", {"value": 1})
        (self.root / "payload.json").write_bytes(b"not-json")
        reader = module.VerifiedReader(self.root, self.files, {"payload.json"})
        with self.assertRaisesRegex(module.AuditError, "size or SHA256"):
            reader.read_json("payload.json")
        self.assertEqual(reader.consumed, {})

    def test_relative_escape_is_rejected(self):
        reader = module.VerifiedReader(self.root, {}, {"../payload.json"})
        with self.assertRaisesRegex(module.AuditError, "Unsafe"):
            reader.read_json("../payload.json")

    def test_semantic_identity_and_receipt_split_mismatches_fail_closed(self):
        contract, paths = self.fixture()
        wrong = replace(contract, manifest_hash="a" * 64)
        reader = module.VerifiedReader(self.root, self.files, set(module.identity_paths(contract)))
        with self.assertRaisesRegex(module.AuditError, "manifest identity"):
            module.audit(reader, wrong)
        checkpoint = json.loads(module.native(self.root / paths[0]).read_bytes())
        checkpoint["split"] = "test"
        self.store(paths[0], checkpoint)
        reader = module.VerifiedReader(self.root, self.files, set(module.identity_paths(contract)))
        with self.assertRaisesRegex(module.AuditError, "receipt identity"):
            module.audit(reader, contract)

    def test_primary_label_requires_eos_and_correct_transition(self):
        contract, paths = self.fixture()
        checkpoint = json.loads(module.native(self.root / paths[0]).read_bytes())
        checkpoint["readout"]["termination_status"] = "MAX_NEW_TOKENS"
        self.store(paths[0], checkpoint)
        reader = module.VerifiedReader(self.root, self.files, set(module.identity_paths(contract)))
        with self.assertRaisesRegex(module.AuditError, "Primary label"):
            module.audit(reader, contract)

    def test_zero_primary_rows_and_caps_are_not_wrong_labels(self):
        contract, paths = self.fixture()
        for checkpoint_path in paths[:2]:
            checkpoint = json.loads(module.native(self.root / checkpoint_path).read_bytes())
            checkpoint.update(transition_label=None, analytic_eligibility={"eligible": False,
                "reason_codes": ["CHECKPOINT_READOUT_NOT_EOS", "TERMINAL_READOUT_NOT_EOS"]},
                readout={"termination_status": "MAX_NEW_TOKENS", "observable_scores": {}},
                extraction={"status": "MISSING_BOXED_ANSWER"}, evaluation={"status": "NON_EVALUABLE"})
            self.store(checkpoint_path, checkpoint)
            terminal_path = module.PREFIX + "runs/" + contract.run_id + "/raw/terminal_readouts/" + module.filename("terminal-readout:" + checkpoint["rollout_id"])
            terminal = json.loads(module.native(self.root / terminal_path).read_bytes())
            terminal.update(readout={"termination_status": "MAX_NEW_TOKENS"},
                extraction={"status": "MISSING_BOXED_ANSWER"}, evaluation={"status": "NON_EVALUABLE"})
            self.store(terminal_path, terminal)
        reader = module.VerifiedReader(self.root, self.files, set(module.identity_paths(contract)))
        result = module.audit(reader, contract)
        self.assertEqual(result["train"]["primary_rows"], {"W_TO_C": 0, "W_TO_W": 0})
        self.assertEqual(result["train"]["primary_problem_counts"], {"W_TO_C": 0, "W_TO_W": 0})
        self.assertEqual(result["train"]["checkpoint_termination"], {"MAX_NEW_TOKENS": 2})
        self.assertEqual(result["train"]["checkpoint_evaluation"], {"NON_EVALUABLE": 2})
        self.assertFalse(result["train"]["raw_primary_floors_pass"])

    def test_unknown_enum_cannot_escape_as_public_counter_key(self):
        with self.assertRaisesRegex(module.AuditError, "Unexpected metadata enum"):
            module.enum("PRIVATE_TEXT_MUST_NOT_ESCAPE", module.TERMINATION)

    def test_bootstrap_size_and_hash_are_checked_before_parse(self):
        self.store("bootstrap.json", {"status": "synthetic"})
        expected = self.files["bootstrap.json"]
        value, metadata = module.checked_bootstrap(self.root, "bootstrap.json", expected["bytes"], expected["sha256"])
        self.assertEqual(value["status"], "synthetic")
        self.assertEqual(metadata, expected)
        with self.assertRaisesRegex(module.AuditError, "Bootstrap input size"):
            module.checked_bootstrap(self.root, "bootstrap.json", expected["bytes"] + 1, expected["sha256"])

    def test_private_output_is_new_namespace_and_write_once(self):
        repo = self.root / "synthetic-repository"
        extraction = self.root / "synthetic-extraction"
        extraction.mkdir()
        output = repo / module.PRIVATE_NAMESPACE / "result.json"
        self.assertEqual(module.validate_output(output, repo, extraction), output.resolve())
        module.write_private_report(output, {"consumed_inputs": {"private-synthetic-id": {"sha256": "a" * 64}}})
        with self.assertRaisesRegex(module.AuditError, "already exists"):
            module.validate_output(output, repo, extraction)
        with self.assertRaises(FileExistsError):
            module.write_private_report(output, {"changed": True})
        with self.assertRaisesRegex(module.AuditError, "namespace"):
            module.validate_output(repo / "artifacts/v2a1/original.json", repo, extraction)

    def test_cli_refuses_nonfrozen_inventory_authority_without_printing_private_path(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            status = module.main(["--extraction-root", str(self.root), "--private-output",
                                  str(self.root / "private.json"), "--inventory-sha", "a" * 64])
        self.assertEqual(status, 2)
        self.assertNotIn(str(self.root), capture.getvalue())
        self.assertEqual(json.loads(capture.getvalue())["status"], "AUDIT_FAILED")


if __name__ == "__main__":
    unittest.main()
