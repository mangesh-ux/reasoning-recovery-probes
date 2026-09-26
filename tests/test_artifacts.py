from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reasoning_recovery.artifacts import (
    AppendOnlyLedger,
    ArtifactConflictError,
    IntentState,
    write_immutable_json,
)


class ArtifactTests(unittest.TestCase):
    def test_immutable_json_allows_same_payload_and_rejects_changed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            write_immutable_json(path, {"status": "COMPLETED", "count": 1})
            write_immutable_json(path, {"count": 1, "status": "COMPLETED"})
            with self.assertRaises(ArtifactConflictError):
                write_immutable_json(path, {"status": "FAILED", "count": 1})

    def test_ledger_rejects_orphan_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = AppendOnlyLedger(Path(temporary) / "ledger.jsonl")
            ledger.append("BASE_COMPLETED", intent_id="intent:test")
            with self.assertRaisesRegex(RuntimeError, "without intent start"):
                ledger.state("intent:test")

    def test_started_request_becomes_interrupted_unknown_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = AppendOnlyLedger(Path(temporary) / "ledger.jsonl")
            ledger.append("INTENT_STARTED", intent_id="intent:test")
            self.assertEqual(ledger.state("intent:test"), IntentState.STARTED_UNKNOWN)
            self.assertEqual(
                ledger.preserve_interrupted_unknown("intent:test"),
                IntentState.INTERRUPTED_UNKNOWN,
            )
            self.assertEqual(ledger.state("intent:test"), IntentState.INTERRUPTED_UNKNOWN)


if __name__ == "__main__":
    unittest.main()
