from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import os
from pathlib import Path
import subprocess
import sys
import unittest

from reasoning_recovery.v2_config import (
    V2ConfigurationError,
    V2Config,
    V2DatasetConfig,
    load_v2_config,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_HASH = "d4903cbaaab23d2fc195b9ecce30f4c5816c3e3027e8cc2e8a09f8eed367b88f"
AMENDED_HASH = "bbab807d3ead00739dcab7482eecb6df763e72fa0308c19b12f0a6e06f30a6b3"
AMENDMENT_PATHS = {
    ("study", "protocol_id"),
    ("dataset", "answer_validation_policy"),
    ("artifacts", "root"),
    ("artifacts", "public_summary_path"),
}


def _config(amended: bool = False) -> V2Config:
    filename = "recovery_v2a1_frozen.yaml" if amended else "recovery_v2_frozen.yaml"
    return load_v2_config(ROOT / "configs" / filename)


def _leaf_paths(value: object, prefix: tuple[object, ...] = ()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaf_paths(item, prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaf_paths(item, prefix + (index,))
    else:
        yield prefix, value


def _set_path(raw: dict, path: tuple[object, ...], value: object) -> None:
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


class V2AmendedConfigTests(unittest.TestCase):
    def test_original_hash_and_default_serialization_are_unchanged(self) -> None:
        original = _config()
        self.assertEqual(original.config_hash, ORIGINAL_HASH)
        self.assertEqual(original.dataset.answer_validation_policy, "all-source-final-answer-v1")
        self.assertNotIn("answer_validation_policy", original.to_dict()["dataset"])
        self.assertEqual(V2Config.from_mapping(original.to_dict()), original)

    def test_explicit_original_policy_canonicalizes_to_original_hash(self) -> None:
        raw = _config().to_dict()
        raw["dataset"]["answer_validation_policy"] = "all-source-final-answer-v1"
        parsed = V2Config.from_mapping(raw)
        self.assertEqual(parsed.config_hash, ORIGINAL_HASH)
        self.assertNotIn("answer_validation_policy", parsed.to_dict()["dataset"])

    def test_policy_is_appended_with_a_backward_compatible_default(self) -> None:
        dataset = _config().dataset
        field_list = fields(V2DatasetConfig)
        self.assertEqual(field_list[-1].name, "answer_validation_policy")
        self.assertEqual(field_list[-1].default, "all-source-final-answer-v1")
        legacy_positional = V2DatasetConfig(
            *(getattr(dataset, field.name) for field in field_list[:-1])
        )
        self.assertEqual(legacy_positional, dataset)

    def test_amended_hash_round_trip_and_only_four_differences(self) -> None:
        original = _config().to_dict()
        amended = _config(True)
        self.assertEqual(amended.config_hash, AMENDED_HASH)
        self.assertNotEqual(amended.config_hash, ORIGINAL_HASH)
        self.assertEqual(amended.dataset.answer_validation_policy, "selected-cohort-final-answer-v1")
        self.assertEqual(V2Config.from_mapping(amended.to_dict()), amended)
        original_leaves = dict(_leaf_paths(original))
        amended_leaves = dict(_leaf_paths(amended.to_dict()))
        changed = {
            path
            for path in original_leaves.keys() | amended_leaves.keys()
            if original_leaves.get(path) != amended_leaves.get(path)
        }
        self.assertEqual(changed, AMENDMENT_PATHS)

    def test_every_hybrid_profile_is_rejected(self) -> None:
        paths = sorted(AMENDMENT_PATHS)
        amended_leaves = dict(_leaf_paths(_config(True).to_dict()))
        for mask in range(1, (1 << len(paths)) - 1):
            with self.subTest(mask=mask):
                hybrid = _config().to_dict()
                for index, path in enumerate(paths):
                    if mask & (1 << index):
                        _set_path(hybrid, path, amended_leaves[path])
                with self.assertRaises(V2ConfigurationError):
                    V2Config.from_mapping(hybrid)

    def test_every_non_amendment_leaf_remains_frozen_in_both_profiles(self) -> None:
        for amended in (False, True):
            frozen = _config(amended).to_dict()
            for path, value in _leaf_paths(frozen):
                if path in AMENDMENT_PATHS:
                    continue
                if isinstance(value, bool):
                    mutation = not value
                elif isinstance(value, int):
                    mutation = value + 1
                elif isinstance(value, float):
                    mutation = value + 0.125
                elif value is None:
                    mutation = "balanced"
                else:
                    mutation = str(value) + "-unapproved"
                with self.subTest(amended=amended, path=path):
                    raw = deepcopy(frozen)
                    _set_path(raw, path, mutation)
                    with self.assertRaises(V2ConfigurationError):
                        V2Config.from_mapping(raw)

    def test_policy_must_be_known_and_nonempty_string(self) -> None:
        for policy in (None, True, 1, "", " ", "skip-invalid-final-answers-v1"):
            with self.subTest(policy=policy):
                raw = _config(True).to_dict()
                raw["dataset"]["answer_validation_policy"] = policy
                with self.assertRaises(V2ConfigurationError):
                    V2Config.from_mapping(raw)

    def test_other_dataset_fields_remain_required_and_unknown_fields_rejected(self) -> None:
        for amended in (False, True):
            raw = _config(amended).to_dict()
            raw["dataset"]["answer_validation_override"] = True
            with self.assertRaises(V2ConfigurationError):
                V2Config.from_mapping(raw)
            raw = _config(amended).to_dict()
            del raw["dataset"]["answer_field"]
            with self.assertRaises(V2ConfigurationError):
                V2Config.from_mapping(raw)

    def test_configuration_loading_imports_no_scientific_runtime(self) -> None:
        script = (
            "from pathlib import Path; import sys; "
            "from reasoning_recovery.v2_config import load_v2_config; "
            "load_v2_config(Path('configs/recovery_v2_frozen.yaml')); "
            "load_v2_config(Path('configs/recovery_v2a1_frozen.yaml')); "
            "forbidden = {'torch', 'transformers', 'datasets', 'math_verify', 'sklearn'}; "
            "assert not forbidden.intersection(name.split('.')[0] for name in sys.modules)"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
