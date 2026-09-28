from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reasoning_recovery.artifacts import (
    AppendOnlyLedger,
    ArtifactConflictError,
    IntentState,
    RunLockError,
    open_run_layout,
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

    def test_full_campaign_scope_keeps_one_run_id_for_bounded_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = {"schema_version": 1, "records": [{"problem_id": "p-1"}]}
            campaign_scope = {
                "problem_ids": [f"p-{index}" for index in range(1, 31)],
                "rollout_seeds": [101, 102, 103, 104],
                "checkpoint_token_positions": [256, 512, 1024, 2048, 3072],
            }
            first_stage = {
                "problem_ids": [f"p-{index}" for index in range(1, 11)],
                "rollout_seeds": [101, 102, 103, 104],
            }
            full_stage = {
                "problem_ids": [f"p-{index}" for index in range(1, 31)],
                "rollout_seeds": [101, 102, 103, 104],
            }
            first_layout = open_run_layout(
                Path(temporary),
                manifest=manifest,
                config_hash="c" * 64,
                campaign_scope=campaign_scope,
            )
            later_layout = open_run_layout(
                Path(temporary),
                manifest=manifest,
                config_hash="c" * 64,
                campaign_scope=campaign_scope,
            )

            self.assertEqual(first_layout.run_id, later_layout.run_id)
            first_layout.write_run_manifest(
                {
                    "run_id": first_layout.run_id,
                    "campaign_scope": campaign_scope,
                }
            )
            later_layout.write_run_manifest(
                {
                    "run_id": later_layout.run_id,
                    "campaign_scope": campaign_scope,
                }
            )
            first_layout.ledger.append(
                "RUN_INVOCATION_STARTED", payload={"invocation_scope": first_stage}
            )
            later_layout.ledger.append(
                "RUN_INVOCATION_STARTED", payload={"invocation_scope": full_stage}
            )

            records = later_layout.ledger.records()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["payload"]["invocation_scope"], first_stage)
            self.assertEqual(records[1]["payload"]["invocation_scope"], full_stage)

    def test_writer_lock_is_exclusive_and_never_breaks_a_stale_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = open_run_layout(
                Path(temporary),
                manifest={"schema_version": 1, "records": [{"problem_id": "p-1"}]},
                config_hash="c" * 64,
                campaign_scope={"problem_ids": ["p-1"], "rollout_seeds": [101]},
            )
            lock = layout.acquire_writer_lock()
            with self.assertRaisesRegex(RunLockError, "active or stale"):
                layout.acquire_writer_lock()
            lock.release()

            layout.writer_lock_path.parent.mkdir(parents=True, exist_ok=True)
            stale_content = '{"record_type":"RUN_WRITER_LOCK","owner":"unknown"}\n'
            layout.writer_lock_path.write_text(stale_content, encoding="utf-8")
            with self.assertRaisesRegex(RunLockError, "active or stale"):
                layout.acquire_writer_lock()
            self.assertEqual(
                layout.writer_lock_path.read_text(encoding="utf-8"), stale_content
            )


if __name__ == "__main__":
    unittest.main()
