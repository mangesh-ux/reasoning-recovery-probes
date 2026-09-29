from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from reasoning_recovery.artifacts import ArtifactConflictError
from reasoning_recovery.v2_artifacts import (
    V2IntentState,
    V2AppendOnlyLedger,
    open_v2_run_layout,
    write_immutable_bytes,
)


class V2ArtifactTests(unittest.TestCase):
    def test_binary_artifact_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tensor.pt"
            write_immutable_bytes(path, b"first")
            write_immutable_bytes(path, b"first")
            with self.assertRaises(ArtifactConflictError):
                write_immutable_bytes(path, b"other")

    def test_ledger_preserves_unknown_started_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = V2AppendOnlyLedger(Path(temporary) / "ledger.jsonl")
            ledger.append("INTENT_STARTED", intent_id="request:1")
            self.assertEqual(ledger.state("request:1"), V2IntentState.STARTED_UNKNOWN)
            self.assertEqual(
                ledger.preserve_interrupted_unknown("request:1"),
                V2IntentState.INTERRUPTED_UNKNOWN,
            )
            self.assertEqual(
                ledger.state("request:1"), V2IntentState.INTERRUPTED_UNKNOWN
            )

    def test_run_identity_binds_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arguments = {
                "artifact_root": Path(temporary),
                "config_hash": "c" * 64,
                "manifest": {"records": [{"problem_id": "p1"}]},
                "split": {"assignments": {"p1": "train"}},
            }
            first = open_v2_run_layout(
                **arguments,
                source_identity={"commit": "a" * 40, "tree": "b" * 40, "is_clean": True},
            )
            changed_tree = open_v2_run_layout(
                **arguments,
                source_identity={"commit": "a" * 40, "tree": "d" * 40, "is_clean": True},
            )
            self.assertNotEqual(first.run_id, changed_tree.run_id)

    def test_writer_lock_never_auto_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = open_v2_run_layout(
                Path(temporary),
                config_hash="c" * 64,
                manifest={"records": [{"problem_id": "p1"}]},
                split={"assignments": {"p1": "train"}},
                source_identity={"commit": "a" * 40, "tree": "b" * 40, "is_clean": True},
            )
            first = layout.acquire_writer_lock()
            with self.assertRaises(Exception):
                layout.acquire_writer_lock()
            first.release()


if __name__ == "__main__":
    unittest.main()
