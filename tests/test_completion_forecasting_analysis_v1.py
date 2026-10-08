"""Synthetic-only contracts for the new CPU completion-forecasting analysis.

These tests import only analysis code. They never load an original receipt,
benchmark row, activation tensor, validation outcome, or model weight.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "completion_forecasting_analysis_v1_under_test",
    REPOSITORY / "scripts" / "completion_forecasting_analysis_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def synthetic_data(n: int = 8) -> dict[str, np.ndarray]:
    """Create deliberately artificial, tiny A/B/lexical/activation inputs."""

    numeric = np.zeros((n, 11), dtype=np.float64)
    numeric[:, 0] = 512.0
    numeric[:, 1] = 512.0 / 4096.0
    numeric[:, 2] = 20.0 + np.arange(n)
    numeric[:, 3] = 7.0 + (np.arange(n) % 2)
    numeric[:, 4] = -1.0 - np.arange(n) / 10.0
    numeric[:, 5] = -2.0 - np.arange(n) / 10.0
    numeric[:, 6] = 1.0 + np.arange(n) % 3
    numeric[:, 7] = 0.25 + np.arange(n) / 20.0
    numeric[:, 8] = 0.9 - np.arange(n) / 30.0
    numeric[:, 9] = np.nan  # Predeclared all-missing handling is mandatory.
    numeric[:, 10] = np.arange(n) % 2
    lexical = np.zeros((n, 512), dtype=np.float64)
    for row in range(n):
        lexical[row, row % 5] = 128.0
    activations = np.zeros((n, 28, 2048), dtype=np.float32)
    activations[:, 4, 0] = np.arange(n)
    activations[:, 4, 1] = np.arange(n) % 2
    return {
        "numeric": numeric,
        "topics": np.array(["synthetic-algebra" if i % 2 else "synthetic-geometry" for i in range(n)]),
        "groups": np.array([f"synthetic-problem-{i // 2}" for i in range(n)]),
        "lexical": lexical,
        "activations": activations,
        "labels": np.arange(n) % 2,
        "row_ids": np.array([f"synthetic-row-{i}" for i in range(n)]),
        "fold_ids": np.arange(n) % 5,
    }


class CompletionForecastingAnalysisV1Tests(unittest.TestCase):
    def test_problem_weights_have_equal_totals_and_mean_one(self) -> None:
        groups = np.array(["a", "b", "b", "c", "c", "c"])
        weights = ANALYSIS.weights(groups)
        np.testing.assert_allclose(weights, [2.0, 1.0, 1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        for group in set(groups):
            self.assertAlmostEqual(float(weights[groups == group].sum()), 2.0)

    def test_log_loss_averages_problems_not_rows_and_pairs_identical_rows(self) -> None:
        labels = np.array([1, 0, 0, 0])
        groups = np.array(["one-row", "three-rows", "three-rows", "three-rows"])
        baseline = np.array([0.1, 0.2, 0.3, 0.4])
        activation = np.array([0.8, 0.1, 0.2, 0.3])
        baseline_losses = ANALYSIS.binary_losses(labels, baseline)
        expected = (baseline_losses[0] + baseline_losses[1:].mean()) / 2.0
        self.assertAlmostEqual(ANALYSIS.score(labels, baseline, groups), float(expected))
        self.assertNotAlmostEqual(float(expected), float(baseline_losses.mean()))
        _, deltas = ANALYSIS.problem_means(
            baseline_losses - ANALYSIS.binary_losses(labels, activation), groups
        )
        self.assertAlmostEqual(
            float(deltas.mean()),
            ANALYSIS.score(labels, baseline, groups) - ANALYSIS.score(labels, activation, groups),
        )

    def test_log_loss_rejects_invalid_probabilities_instead_of_clipping_them(self) -> None:
        labels = np.array([0, 1])
        for probabilities in (
            np.array([np.nan, 0.5]),
            np.array([np.inf, 0.5]),
            np.array([-0.1, 0.5]),
            np.array([0.5, 1.1]),
        ):
            with self.subTest(probabilities=probabilities):
                with self.assertRaisesRegex(RuntimeError, "Invalid prediction"):
                    ANALYSIS.binary_losses(labels, probabilities)

    def test_fold_hash_matches_protocol_and_keeps_all_problem_rows_together(self) -> None:
        all_groups = np.array([f"synthetic-{difficulty}-{i:02d}" for difficulty in (7, 8) for i in range(12)])
        all_difficulty = np.array([difficulty for difficulty in (7, 8) for _ in range(12)])
        groups = np.repeat(all_groups, 3)
        difficulties = np.repeat(all_difficulty, 3)
        actual = ANALYSIS.fold_assignments(groups, difficulties)
        expected = {}
        for difficulty in (7, 8):
            candidates = [g for g, d in zip(all_groups, all_difficulty) if d == difficulty]
            ranked = sorted(
                candidates,
                key=lambda g: (hashlib.sha256(("completion-forecasting-v1|fold|" + g).encode("utf-8")).hexdigest(), g),
            )
            expected.update({group: rank % 5 for rank, group in enumerate(ranked)})
        self.assertEqual(actual, expected)
        self.assertEqual(actual, ANALYSIS.fold_assignments(groups[::-1], difficulties[::-1]))
        for group in all_groups:
            self.assertEqual(len({actual[g] for g in groups[groups == group]}), 1)

    def test_fold_builder_rejects_inconsistent_problem_difficulty(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "difficulty changed"):
            ANALYSIS.fold_assignments(np.array(["same", "same"]), np.array([7.0, 8.0]))

    def test_all_missing_column_is_train_zero_imputed_without_dropping_rows(self) -> None:
        train = synthetic_data()
        preprocessing = ANALYSIS.Preprocessor("B").fit(train)
        self.assertEqual(preprocessing.medians[9], 0.0)
        self.assertFalse(preprocessing.keep[9])
        self.assertFalse(preprocessing.keep[11 + 9])
        evaluation = synthetic_data(6)
        first = preprocessing.transform(evaluation)
        evaluation["numeric"][:, 9] = 1000000.0
        second = preprocessing.transform(evaluation)
        self.assertEqual(len(first), len(evaluation["labels"]))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)

    def test_unknown_topics_and_extreme_heldout_values_do_not_refit(self) -> None:
        train = synthetic_data()
        train["numeric"][:, 4] = [np.nan, -2.0, -4.0, np.nan, -6.0, -8.0, np.nan, -10.0]
        preprocessing = ANALYSIS.Preprocessor("Blex").fit(train)
        self.assertEqual(preprocessing.medians[4], -6.0)
        snapshot = tuple(x.copy() for x in (preprocessing.medians, preprocessing.mean, preprocessing.scale, preprocessing.keep))
        vocabulary = preprocessing.topics.categories_[0].copy()
        evaluation = synthetic_data(4)
        evaluation["numeric"][:, 4] = 1e9
        evaluation["topics"] = np.array(["unseen-synthetic-topic"] * 4)
        unscaled = preprocessing.unscaled(evaluation)
        np.testing.assert_array_equal(unscaled[:, 22:22 + len(vocabulary)], 0.0)
        self.assertTrue(np.isfinite(preprocessing.transform(evaluation)).all())
        for before, after in zip(snapshot, (preprocessing.medians, preprocessing.mean, preprocessing.scale, preprocessing.keep)):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(vocabulary, preprocessing.topics.categories_[0])

    def test_feature_transform_ignores_labels_and_unknown_future_metadata(self) -> None:
        train = synthetic_data()
        preprocessing = ANALYSIS.Preprocessor("Clex", layer=4).fit(train)
        expected = preprocessing.transform(train)
        altered = {key: value.copy() for key, value in train.items()}
        altered["labels"] = 1 - altered["labels"]
        altered["eventual_close_position"] = np.arange(len(train["labels"])) + 3000
        altered["synthetic_reference_answer"] = np.array(["not-a-model-feature"] * len(train["labels"]))
        np.testing.assert_array_equal(preprocessing.transform(altered), expected)

    def test_only_the_selected_activation_layer_enters_clex(self) -> None:
        train = synthetic_data()
        preprocessing = ANALYSIS.Preprocessor("Clex", layer=4).fit(train)
        expected = preprocessing.transform(train)
        other_layer_changed = {key: value.copy() for key, value in train.items()}
        other_layer_changed["activations"][:, 3, :] = 1e7
        np.testing.assert_array_equal(preprocessing.transform(other_layer_changed), expected)
        selected_changed = {key: value.copy() for key, value in train.items()}
        selected_changed["activations"][:, 4, 0] += 100
        self.assertFalse(np.array_equal(preprocessing.transform(selected_changed), expected))

    def test_saved_train_model_predictions_restore_without_any_fit(self) -> None:
        train = synthetic_data()
        evaluation = synthetic_data(6)
        evaluation["topics"] = np.array(["unseen-topic"] * 6)
        for model in ("A", "B", "Blex", "C", "Clex"):
            with self.subTest(model=model):
                layer = 4 if model in ("C", "Clex") else None
                expected, state = ANALYSIS.predict(train, evaluation, model, 0.01, layer=layer)
                with patch.object(ANALYSIS.LogisticRegression, "fit", side_effect=AssertionError("heldout fit forbidden")), patch.object(ANALYSIS.Preprocessor, "fit", side_effect=AssertionError("heldout preprocessing fit forbidden")):
                    actual = ANALYSIS.predict_frozen(evaluation, state)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                self.assertTrue(np.isfinite(actual).all())
                self.assertEqual(state["model"], model)
                self.assertEqual(state["layer"], layer)

    def test_bootstrap_retains_repeated_problem_draws(self) -> None:
        values = np.array([0.0, 0.1, 1.7])
        seed = 12345
        indices = np.random.default_rng(seed).integers(0, 3, size=(2000, 3))
        self.assertTrue(any(len(set(row)) < 3 for row in indices))
        expected = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
        np.testing.assert_array_equal(ANALYSIS.bootstrap_interval(values, seed=seed), expected)

    def test_paired_bootstrap_counts_invalid_one_class_resamples(self) -> None:
        values = np.linspace(-0.1, 0.2, 8)
        positive = np.array([True] + [False] * 7)
        negative = ~positive
        seed = 91
        indices = np.random.default_rng(seed).integers(0, 8, size=(2000, 8))
        valid = positive[indices].any(axis=1) & negative[indices].any(axis=1)
        result = ANALYSIS.paired_bootstrap(values, positive, negative, seed=seed)
        self.assertEqual(result["valid_resamples"], int(valid.sum()))
        self.assertLess(result["valid_resamples"], 1900)
        self.assertIsNone(result["problem_cluster_bootstrap_95_percentile_ci"])
        self.assertAlmostEqual(result["invalid_fraction"], float(1.0 - valid.mean()))

    def test_paired_bootstrap_mixed_problems_are_all_valid_and_paired(self) -> None:
        values = np.array([-0.1, 0.03, 0.15])
        both_classes = np.array([True, True, True])
        result = ANALYSIS.paired_bootstrap(values, both_classes, both_classes, seed=17)
        self.assertEqual(result["valid_resamples"], 2000)
        self.assertEqual(result["invalid_fraction"], 0.0)
        np.testing.assert_array_equal(
            result["problem_cluster_bootstrap_95_percentile_ci"],
            ANALYSIS.bootstrap_interval(values, seed=17),
        )

    def test_matched_concordance_keeps_same_problem_pairs_and_half_credit_ties(self) -> None:
        data = {
            "groups": np.array(["g1", "g1", "g1", "g1", "g2", "g2", "single-class"]),
            "labels": np.array([1, 1, 0, 0, 1, 0, 1]),
        }
        predictions = np.array([0.8, 0.5, 0.5, 0.2, 0.1, 0.9, 0.9])
        groups, concordance = ANALYSIS.matched(data, predictions)
        self.assertEqual(groups, ["g1", "g2"])
        np.testing.assert_allclose(concordance, [0.875, 0.0])


if __name__ == "__main__":
    unittest.main()
