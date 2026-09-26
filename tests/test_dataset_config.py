from __future__ import annotations

import unittest

from reasoning_recovery.config import ConfigurationError, PilotConfig
from reasoning_recovery.dataset import DatasetContractError, select_records_from_manifest
from tests.helpers import pilot_config, snapshot


class DatasetConfigTests(unittest.TestCase):
    def test_manifest_is_deterministic_and_revalidates_source_hashes(self) -> None:
        config = pilot_config()
        source = snapshot()
        manifest = source.selection_manifest(config=config)
        selected = select_records_from_manifest(source, manifest)
        self.assertEqual(len(selected), 2)
        self.assertEqual(manifest["dataset"]["resolved_revision"], "a" * 40)
        tampered = dict(manifest)
        tampered_records = [dict(item) for item in manifest["records"]]
        tampered_records[0]["problem_sha256"] = "0" * 64
        tampered["records"] = tampered_records
        with self.assertRaises(DatasetContractError):
            select_records_from_manifest(source, tampered)

    def test_forced_cue_must_start_with_the_declared_marker(self) -> None:
        raw = pilot_config().to_dict()
        raw["forced_answer"]["close_think_text"] = "answer only"
        with self.assertRaises(ConfigurationError):
            PilotConfig.from_mapping(raw)

    def test_unknown_fields_are_rejected_instead_of_silently_unhashed(self) -> None:
        raw = pilot_config().to_dict()
        raw["generation"]["temperature_typo"] = 0.7
        with self.assertRaises(ConfigurationError):
            PilotConfig.from_mapping(raw)


if __name__ == "__main__":
    unittest.main()
