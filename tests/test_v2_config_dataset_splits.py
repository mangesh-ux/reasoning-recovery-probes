from __future__ import annotations

from pathlib import Path
import unittest

from reasoning_recovery.v2_config import V2ConfigurationError, V2Config, load_v2_config
from reasoning_recovery.v2_dataset import (
    DeepMathSnapshot,
    V2DatasetContractError,
    V2ProblemRecord,
    derive_v2_rollout_seed,
)
from reasoning_recovery.v2_splits import (
    build_v2_grouped_split,
    v2_problem_ids_for_split,
)


def _config() -> V2Config:
    root = Path(__file__).resolve().parents[1]
    return load_v2_config(root / "configs" / "recovery_v2_frozen.yaml")


def _snapshot(config: V2Config, *, duplicate_first_pair: bool = False) -> DeepMathSnapshot:
    records: list[V2ProblemRecord] = []
    source_index = 0
    for difficulty in (7.0, 8.0):
        for item in range(90):
            question = f"problem difficulty {difficulty} item {item}"
            cluster = f"{source_index:064x}"
            if duplicate_first_pair and difficulty == 7.0 and item == 1:
                question = "problem difficulty 7.0 item 0"
                cluster = f"{0:064x}"
            records.append(
                V2ProblemRecord(
                    problem_id=f"deepmath:{source_index}:{source_index:016x}",
                    source_index=source_index,
                    question=question,
                    reference_answer="1",
                    difficulty=difficulty,
                    topic="Mathematics -> Algebra",
                    duplicate_cluster_sha256=cluster,
                )
            )
            source_index += 1
    return DeepMathSnapshot(
        dataset_id=config.dataset.dataset_id,
        requested_revision=config.dataset.revision,
        resolved_revision=config.dataset.revision,
        split=config.dataset.split,
        observed_fingerprint="fixture",
        records=tuple(records),
    )


class V2ConfigDatasetSplitTests(unittest.TestCase):
    def test_frozen_config_loads_and_mutation_is_rejected(self) -> None:
        config = _config()
        raw = config.to_dict()
        raw["generation"]["checkpoint_token_positions"] = [128, 256]
        with self.assertRaises(V2ConfigurationError):
            V2Config.from_mapping(raw)

    def test_stratified_selection_is_deterministic_and_quota_exact(self) -> None:
        config = _config()
        snapshot = _snapshot(config)
        first = snapshot.selection_manifest(config=config)
        second = snapshot.selection_manifest(config=config)
        self.assertEqual(first, second)
        records = first["records"]
        self.assertEqual(len(records), 160)
        self.assertEqual(sum(record["difficulty"] == 7.0 for record in records), 80)
        self.assertEqual(sum(record["difficulty"] == 8.0 for record in records), 80)
        self.assertEqual(
            len({record["duplicate_cluster_sha256"] for record in records}), 160
        )

    def test_selection_skips_duplicate_cluster_without_replacement(self) -> None:
        config = _config()
        manifest = _snapshot(config, duplicate_first_pair=True).selection_manifest(
            config=config
        )
        records = manifest["records"]
        self.assertEqual(len(records), 160)
        self.assertEqual(
            len({record["duplicate_cluster_sha256"] for record in records}), 160
        )

    def test_grouped_split_is_balanced_and_has_no_problem_leakage(self) -> None:
        config = _config()
        manifest = _snapshot(config).selection_manifest(config=config)
        split = build_v2_grouped_split(manifest, config=config)
        assignments = split["assignments"]
        self.assertEqual(len(assignments), 160)
        self.assertEqual(len(v2_problem_ids_for_split(split, "train")), 96)
        self.assertEqual(len(v2_problem_ids_for_split(split, "validation")), 32)
        self.assertEqual(len(v2_problem_ids_for_split(split, "test")), 32)
        for split_name, expected in (("train", 48), ("validation", 16), ("test", 16)):
            self.assertEqual(
                sum(
                    row["split"] == split_name and row["difficulty"] == 7.0
                    for row in assignments
                ),
                expected,
            )
            self.assertEqual(
                sum(
                    row["split"] == split_name and row["difficulty"] == 8.0
                    for row in assignments
                ),
                expected,
            )

    def test_seed_derivation_is_stable_and_problem_specific(self) -> None:
        first = derive_v2_rollout_seed(
            root_seed=20260929, problem_id="problem-a", rollout_index=0
        )
        self.assertEqual(
            first,
            derive_v2_rollout_seed(
                root_seed=20260929, problem_id="problem-a", rollout_index=0
            ),
        )
        self.assertNotEqual(
            first,
            derive_v2_rollout_seed(
                root_seed=20260929, problem_id="problem-b", rollout_index=0
            ),
        )
        self.assertGreater(first, 0)


if __name__ == "__main__":
    unittest.main()
