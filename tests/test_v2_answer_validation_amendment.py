"""Synthetic-only tests of the explicitly authorized final-answer amendment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from reasoning_recovery.provenance import sha256_text
from reasoning_recovery.v2_config import load_v2_config
import reasoning_recovery.v2_dataset as dataset_module
from reasoning_recovery.v2_dataset import (
    ALL_SOURCE_FINAL_ANSWER_POLICY,
    SELECTED_COHORT_FINAL_ANSWER_POLICY,
    V2DatasetContractError,
    V2ProblemRecord,
    V2SelectionCandidate,
    load_deepmath_snapshot,
    select_v2_records_from_manifest,
)
from reasoning_recovery.v2_splits import build_v2_grouped_split


def _config(amended: bool = False):
    filename = "recovery_v2a1_frozen.yaml" if amended else "recovery_v2_frozen.yaml"
    return load_v2_config(Path(__file__).resolve().parents[1] / "configs" / filename)


def _rows():
    rows = [
        {"question": f"synthetic outside {item}", "final_answer": "outside answer",
         "difficulty": 6.0, "topic": "synthetic topic"}
        for item in range(8)
    ]
    for difficulty in (7.0, 8.0):
        for item in range(120):
            rows.append(
                {"question": f"synthetic problem {difficulty} number {item}",
                 "final_answer": f"PRIVATE SYNTHETIC ANSWER {difficulty} {item}",
                 "difficulty": difficulty, "topic": "synthetic topic"}
            )
    return rows


class _FixtureDataset:
    column_names = ["question", "final_answer", "difficulty", "topic"]
    _fingerprint = "synthetic-fixture"

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


def _snapshot(rows, config):
    """Exercise the actual loader with fake Hub/dataset modules; never download."""

    datasets = ModuleType("datasets")
    hub = ModuleType("huggingface_hub")
    datasets.load_dataset = lambda *args, **kwargs: _FixtureDataset(rows)
    hub.HfApi = lambda: SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(sha=config.dataset.revision)
    )
    with patch.dict(sys.modules, {"datasets": datasets, "huggingface_hub": hub}):
        return load_deepmath_snapshot(config.dataset)


def _independent_membership(rows):
    """A literal oracle independent of the implementation's selection helpers."""

    digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    selected = []
    clusters = set()
    for difficulty in (7.0, 8.0):
        eligible = []
        for index, row in enumerate(rows):
            if float(row["difficulty"]) != difficulty:
                continue
            problem_id = f"deepmath:{index}:{digest(row['question'])[:16]}"
            cluster = digest(" ".join(row["question"].split()))
            rank = digest(f"20260929\0{difficulty:.1f}\0{problem_id}\0{index}")
            eligible.append((rank, index, problem_id, cluster))
        stratum_selected = []
        for _, index, problem_id, cluster in sorted(eligible):
            if cluster in clusters:
                continue
            clusters.add(cluster)
            stratum_selected.append((index, problem_id, cluster))
            if len(stratum_selected) == 80:
                break
        if len(stratum_selected) != 80:
            raise ValueError("synthetic quota unavailable")
        selected.extend(stratum_selected)
    return selected


class V2AnswerValidationAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.legacy = _config()
        self.amended = _config(True)
        self.rows = _rows()

    def test_policy_defaults_preserve_original_config_hash(self):
        self.assertEqual(self.legacy.dataset.answer_validation_policy,
                         ALL_SOURCE_FINAL_ANSWER_POLICY)
        self.assertNotIn("answer_validation_policy", self.legacy.to_dict()["dataset"])
        self.assertEqual(self.legacy.config_hash,
                         "d4903cbaaab23d2fc195b9ecce30f4c5816c3e3027e8cc2e8a09f8eed367b88f")
        self.assertEqual(self.amended.dataset.answer_validation_policy,
                         SELECTED_COHORT_FINAL_ANSWER_POLICY)
        self.assertNotEqual(self.legacy.config_hash, self.amended.config_hash)

    def test_legacy_loader_still_rejects_bad_unselected_answer(self):
        for value in (None, 123, True, "", " \t\n"):
            with self.subTest(value_type=type(value).__name__):
                rows = deepcopy(self.rows)
                rows[0]["final_answer"] = value
                with self.assertRaisesRegex(V2DatasetContractError, "row 0.*final_answer"):
                    _snapshot(rows, self.legacy)
        rows = deepcopy(self.rows)
        del rows[0]["final_answer"]
        with self.assertRaisesRegex(V2DatasetContractError, "row 0.*final_answer"):
            _snapshot(rows, self.legacy)

    def test_legacy_validation_order_still_checks_answer_before_topic(self):
        rows = deepcopy(self.rows)
        rows[0]["final_answer"] = None
        rows[0]["topic"] = None
        with self.assertRaisesRegex(V2DatasetContractError, "final_answer"):
            _snapshot(rows, self.legacy)

    def test_all_source_rows_and_raw_scientific_fields_retained_without_coercion(self):
        raw_answer = {"synthetic": [None, 5]}
        rows = deepcopy(self.rows)
        rows[0]["final_answer"] = raw_answer
        rows[0]["extra_source_field"] = "retained synthetic metadata"
        del rows[1]["final_answer"]
        snapshot = _snapshot(rows, self.amended)
        self.assertEqual(len(snapshot.records), len(rows))
        self.assertEqual([row.source_index for row in snapshot.records], list(range(len(rows))))
        self.assertTrue(all(isinstance(row, V2SelectionCandidate) for row in snapshot.records))
        self.assertIs(snapshot.records[0].source_row["final_answer"], raw_answer)
        self.assertEqual(snapshot.records[0].source_row,
                         {field: rows[0][field] for field in _FixtureDataset.column_names})
        self.assertNotIn("extra_source_field", snapshot.records[0].source_row)
        self.assertNotIn("final_answer", snapshot.records[1].source_row)
        self.assertFalse(hasattr(snapshot.records[0], "reference_answer_sha256"))

    def test_valid_source_identity_and_split_are_identical_to_original(self):
        old = _snapshot(self.rows, self.legacy).selection_manifest(config=self.legacy)
        new = _snapshot(self.rows, self.amended).selection_manifest(config=self.amended)
        self.assertEqual(old["records"], new["records"])
        self.assertEqual(old["strata"], new["strata"])
        self.assertEqual(
            [(row["source_index"], row["problem_id"], row["duplicate_cluster_sha256"])
             for row in new["records"]], _independent_membership(self.rows)
        )
        self.assertEqual(build_v2_grouped_split(old, config=self.legacy)["assignments"],
                         build_v2_grouped_split(new, config=self.amended)["assignments"])
        self.assertNotIn("answer_validation_policy", old["dataset"])
        self.assertNotIn("answer_validation_summary", old["dataset"])
        self.assertEqual(select_v2_records_from_manifest(_snapshot(self.rows, self.legacy), old),
                         select_v2_records_from_manifest(_snapshot(self.rows, self.amended), new))

    def test_bad_unselected_answers_never_filter_or_change_the_cohort(self):
        baseline = _snapshot(self.rows, self.amended).selection_manifest(config=self.amended)
        selected_indices = {row["source_index"] for row in baseline["records"]}
        unselected_eligible = next(index for index in range(8, len(self.rows))
                                   if index not in selected_indices)
        rows = deepcopy(self.rows)
        del rows[0]["final_answer"]
        for index, value in enumerate((None, 17, "", "\t\n"), start=1):
            rows[index]["final_answer"] = value
        rows[unselected_eligible]["final_answer"] = None
        snapshot = _snapshot(rows, self.amended)
        changed = snapshot.selection_manifest(config=self.amended)
        self.assertEqual(changed["records"], baseline["records"])
        self.assertEqual(len(snapshot.records), len(rows))
        self.assertEqual(snapshot.records[unselected_eligible].source_row["final_answer"], None)
        summary = changed["dataset"]["answer_validation_summary"]
        self.assertEqual(summary["source_row_count"], len(rows))
        self.assertEqual(summary["invalid_source_answer_count"], 6)
        self.assertEqual(summary["source_answer_failure_counts"],
                         {"missing_key": 1, "null": 2, "non_string": 1,
                          "empty_string": 1, "whitespace_only": 1})
        self.assertEqual(summary["selected_invalid_answer_count"], 0)
        self.assertEqual(summary["replacement_count"], 0)
        self.assertEqual(len(select_v2_records_from_manifest(snapshot, changed)), 160)

    def test_selected_bad_answers_stop_with_zero_replacement(self):
        baseline = _snapshot(self.rows, self.amended).selection_manifest(config=self.amended)
        bad_index = baseline["records"][0]["source_index"]
        for reason, value in (("null", None), ("non_string", 12), ("bool", True),
                              ("empty", ""), ("whitespace", " \t\n"), ("missing", None)):
            with self.subTest(reason=reason):
                rows = deepcopy(self.rows)
                if reason == "missing":
                    del rows[bad_index]["final_answer"]
                else:
                    rows[bad_index]["final_answer"] = value
                snapshot = _snapshot(rows, self.amended)
                with patch.object(dataset_module, "_select_structural_records",
                                  wraps=dataset_module._select_structural_records) as selector:
                    with self.assertRaisesRegex(V2DatasetContractError,
                                                "invalid_count=1; replacement_count=0"):
                        snapshot.selection_manifest(config=self.amended)
                    self.assertEqual(selector.call_count, 1)
                fixed, _ = dataset_module._select_structural_records(snapshot.records,
                                                                     self.amended.dataset)
                self.assertEqual([row.problem_id for row in fixed],
                                 [row["problem_id"] for row in baseline["records"]])

    def test_answer_validation_runs_only_after_all_identities_are_fixed(self):
        snapshot = _snapshot(self.rows, self.amended)
        fixed_before_answer = False
        original_selector = dataset_module._select_structural_records
        original_validator = dataset_module._required_row_string
        checked_answer_indices = []

        def selector(*args, **kwargs):
            nonlocal fixed_before_answer
            result = original_selector(*args, **kwargs)
            self.assertEqual(len(result[0]), 160)
            fixed_before_answer = True
            return result

        def validator(row, field, index):
            if field == "final_answer":
                self.assertTrue(fixed_before_answer)
                checked_answer_indices.append(index)
            return original_validator(row, field, index)

        with patch.object(dataset_module, "_select_structural_records", side_effect=selector), \
                patch.object(dataset_module, "_required_row_string", side_effect=validator):
            manifest = snapshot.selection_manifest(config=self.amended)
        self.assertEqual(checked_answer_indices,
                         [row["source_index"] for row in manifest["records"]])

    def test_global_structural_validation_still_fails_outside_eligible_strata(self):
        for field, value in (("question", None), ("question", " \t"),
                             ("topic", None), ("topic", ""), ("difficulty", None),
                             ("difficulty", True), ("difficulty", "6"),
                             ("difficulty", float("nan")), ("difficulty", float("inf"))):
            with self.subTest(field=field, value_type=type(value).__name__):
                rows = deepcopy(self.rows)
                rows[0]["final_answer"] = None
                rows[0][field] = value
                with self.assertRaisesRegex(V2DatasetContractError, field):
                    _snapshot(rows, self.amended)

    def test_raw_question_identity_and_unicode_duplicate_control_are_unchanged(self):
        rows = deepcopy(self.rows)
        rows.append({**rows[8], "question": " \t" + rows[8]["question"].replace(" ", "\u00a0") + "\n"})
        rows.append({**rows[8], "difficulty": 8.0})
        old = _snapshot(rows, self.legacy).selection_manifest(config=self.legacy)
        new = _snapshot(rows, self.amended).selection_manifest(config=self.amended)
        self.assertEqual(old["records"], new["records"])
        self.assertEqual(old["strata"], new["strata"])
        self.assertEqual(len({row["duplicate_cluster_sha256"] for row in new["records"]}), 160)
        self.assertEqual([(row["source_index"], row["problem_id"], row["duplicate_cluster_sha256"])
                          for row in new["records"]], _independent_membership(rows))

    def test_difficulty_remains_exact_without_rounding(self):
        rows = deepcopy(self.rows)
        rows[0]["difficulty"] = 7.000001
        new = _snapshot(rows, self.amended).selection_manifest(config=self.amended)
        self.assertNotIn(0, [row["source_index"] for row in new["records"]])
        self.assertEqual([(row["source_index"], row["problem_id"], row["duplicate_cluster_sha256"])
                          for row in new["records"]], _independent_membership(rows))

    def test_quota_failure_happens_before_any_answer_check(self):
        rows = [row for row in self.rows if row["difficulty"] != 8.0]
        rows[8]["final_answer"] = None
        snapshot = _snapshot(rows, self.amended)
        with patch.object(dataset_module, "_validate_selected_answers") as validator:
            with self.assertRaisesRegex(V2DatasetContractError, "unique rows for difficulty 8"):
                snapshot.selection_manifest(config=self.amended)
            validator.assert_not_called()

    def test_valid_selected_answer_is_never_stripped_or_converted(self):
        baseline = _snapshot(self.rows, self.amended).selection_manifest(config=self.amended)
        index = baseline["records"][0]["source_index"]
        rows = deepcopy(self.rows)
        answer = " \tPRIVATE EXACT SYNTHETIC ANSWER\n"
        rows[index]["final_answer"] = answer
        snapshot = _snapshot(rows, self.amended)
        manifest = snapshot.selection_manifest(config=self.amended)
        selected = select_v2_records_from_manifest(snapshot, manifest)
        self.assertEqual(selected[0].reference_answer, answer)
        self.assertIsInstance(selected[0], V2ProblemRecord)
        self.assertEqual(manifest["records"][0]["reference_answer_sha256"], sha256_text(answer))

    def test_manifest_and_aggregate_summary_never_publish_raw_answers_or_questions(self):
        manifest = _snapshot(self.rows, self.amended).selection_manifest(config=self.amended)
        serialized = json.dumps(manifest)
        for row in self.rows:
            self.assertNotIn(row["question"], serialized)
            self.assertNotIn(row["final_answer"], serialized)
        summary = manifest["dataset"]["answer_validation_summary"]
        self.assertEqual(set(summary), {"source_row_count", "invalid_source_answer_count",
                                      "source_answer_failure_counts", "selected_count",
                                      "selected_invalid_answer_count", "replacement_count"})

    def test_rebinding_recomputes_membership_and_rejects_substitution_or_reordering(self):
        snapshot = _snapshot(self.rows, self.amended)
        manifest = snapshot.selection_manifest(config=self.amended)
        changed = deepcopy(manifest)
        changed["records"][0], changed["records"][1] = changed["records"][1], changed["records"][0]
        with self.assertRaisesRegex(V2DatasetContractError, "fixed cohort membership"):
            select_v2_records_from_manifest(snapshot, changed)
        changed = deepcopy(manifest)
        selected_ids = {row["problem_id"] for row in manifest["records"]}
        alternative = next(row for row in snapshot.records if row.problem_id not in selected_ids)
        changed["records"][0]["problem_id"] = alternative.problem_id
        with self.assertRaisesRegex(V2DatasetContractError, "fixed cohort membership"):
            select_v2_records_from_manifest(snapshot, changed)

    def test_rebinding_rejects_invalid_selected_answer_and_valid_changed_answer_hash(self):
        manifest = _snapshot(self.rows, self.amended).selection_manifest(config=self.amended)
        index = manifest["records"][0]["source_index"]
        rows = deepcopy(self.rows)
        rows[index]["final_answer"] = None
        with self.assertRaisesRegex(V2DatasetContractError, "replacement_count=0"):
            select_v2_records_from_manifest(_snapshot(rows, self.amended), manifest)
        rows[index]["final_answer"] = "different valid synthetic answer"
        with self.assertRaisesRegex(V2DatasetContractError, "differs from reloaded"):
            select_v2_records_from_manifest(_snapshot(rows, self.amended), manifest)

    def test_rebinding_rejects_changed_hash_contract_seed_and_source_summary(self):
        snapshot = _snapshot(self.rows, self.amended)
        manifest = snapshot.selection_manifest(config=self.amended)
        for mutation in ("record_hash", "selection_seed", "summary", "policy"):
            with self.subTest(mutation=mutation):
                changed = deepcopy(manifest)
                if mutation == "record_hash":
                    changed["records"][0]["question_sha256"] = "0" * 64
                elif mutation == "selection_seed":
                    changed["dataset"]["selection_seed"] += 1
                elif mutation == "summary":
                    changed["dataset"]["answer_validation_summary"]["source_row_count"] += 1
                else:
                    del changed["dataset"]["answer_validation_policy"]
                with self.assertRaises(V2DatasetContractError):
                    select_v2_records_from_manifest(snapshot, changed)

    def test_manifest_policy_cannot_cross_the_original_and_amended_namespaces(self):
        old_snapshot = _snapshot(self.rows, self.legacy)
        new_snapshot = _snapshot(self.rows, self.amended)
        old = old_snapshot.selection_manifest(config=self.legacy)
        new = new_snapshot.selection_manifest(config=self.amended)
        with self.assertRaises(V2DatasetContractError):
            select_v2_records_from_manifest(old_snapshot, new)
        with self.assertRaises(V2DatasetContractError):
            select_v2_records_from_manifest(new_snapshot, old)
        with self.assertRaises(V2DatasetContractError):
            old_snapshot.selection_manifest(config=self.amended)
        with self.assertRaises(V2DatasetContractError):
            new_snapshot.selection_manifest(config=self.legacy)

    def test_amended_summary_must_be_safe_complete_consistent_counts(self):
        manifest = _snapshot(self.rows, self.amended).selection_manifest(config=self.amended)
        for mutation in ("replacement", "selected_invalid", "bool", "private", "missing", "sum"):
            with self.subTest(mutation=mutation):
                changed = deepcopy(manifest)
                summary = changed["dataset"]["answer_validation_summary"]
                if mutation == "replacement":
                    summary["replacement_count"] = 1
                elif mutation == "selected_invalid":
                    summary["selected_invalid_answer_count"] = 1
                elif mutation == "bool":
                    summary["source_row_count"] = True
                elif mutation == "private":
                    summary["source_row_ids"] = ["private synthetic identity"]
                elif mutation == "missing":
                    del summary["source_answer_failure_counts"]["null"]
                else:
                    summary["invalid_source_answer_count"] = 1
                with self.assertRaises(V2DatasetContractError):
                    dataset_module._validate_manifest(self.amended, changed)

    def test_unknown_answer_policy_fails_before_loading_any_source(self):
        bad = replace(self.amended.dataset, answer_validation_policy="unsupported-policy")
        with self.assertRaisesRegex(V2DatasetContractError, "unsupported"):
            load_deepmath_snapshot(bad)


if __name__ == "__main__":
    unittest.main()
