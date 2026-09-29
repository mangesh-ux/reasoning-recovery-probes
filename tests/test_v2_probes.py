"""Offline contract tests for V2 leakage-safe probe selection and analysis."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from reasoning_recovery.v2_config import load_v2_config
from reasoning_recovery.v2_features import FeatureSet, V2FeatureRow, V2PrimaryRow
from reasoning_recovery.v2_probes import (
    activation_available_cohort,
    evaluate_test_models,
    event_gate,
    probability_metrics,
    refit_models_for_test,
    select_validation_models,
)


class V2ProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        frozen = load_v2_config(root / "configs" / "recovery_v2_frozen.yaml")
        model = replace(frozen.model, expected_num_hidden_layers=2, expected_hidden_size=3)
        probe = replace(frozen.probe, regularization_c_grid=(0.1, 1.0))
        analysis = replace(
            frozen.analysis,
            bootstrap_resamples=25,
            bootstrap_min_valid_fraction=0.5,
            min_train_rows_per_class=4,
            min_train_problems_per_class=2,
            min_validation_rows_per_class=2,
            min_validation_problems_per_class=2,
            min_test_rows_per_class=2,
            min_test_problems_per_class=2,
            min_matched_problem_checkpoint_groups=2,
            min_matched_unique_problems=2,
        )
        self.config = replace(frozen, model=model, probe=probe, analysis=analysis)

    def _rows(self, split: str, problem_count: int) -> tuple[tuple[V2PrimaryRow, ...], dict[str, np.ndarray]]:
        rows: list[V2PrimaryRow] = []
        activations: dict[str, np.ndarray] = {}
        for problem_index in range(problem_count):
            # Same problem and coordinate, distinct stochastic rollouts: this
            # produces the exact matched-control geometry used by V2.
            for rollout_suffix, label in (("positive", "W_TO_C"), ("negative", "W_TO_W")):
                checkpoint = 128
                row_id = f"{split}-{problem_index}-{rollout_suffix}-{checkpoint}"
                feature = V2FeatureRow(
                    row_id=row_id,
                    problem_id=f"{split}-problem-{problem_index}",
                    rollout_id=f"{split}-rollout-{problem_index}-{rollout_suffix}",
                    checkpoint_token=checkpoint,
                    checkpoint_token_position=float(checkpoint),
                    normalized_checkpoint_position=checkpoint / 4096.0,
                    problem_token_count=20.0 + problem_index,
                    difficulty=7.0 if problem_index % 2 else 8.0,
                    topic="algebra" if problem_index % 2 else "geometry",
                    mean_answer_token_logprob=-0.1 * (problem_index + 1),
                    min_answer_token_logprob=-0.2 * (problem_index + 1),
                    answer_token_count=1.0 + (problem_index % 3),
                    first_readout_token_entropy_nats=0.5 + problem_index / 50,
                    first_readout_token_probability_margin=0.2 + problem_index / 100,
                    previous_readout_agreement=0.0,
                    previous_readout_available=0.0,
                )
                rows.append(V2PrimaryRow(feature_row=feature, transition_label=label))
                # A reproducible tiny two-layer vector; no terminal or label
                # value is routed through the feature object itself.
                activations[row_id] = np.asarray(
                    [[problem_index / 10, checkpoint / 1000, 0.1], [0.2, problem_index / 20, checkpoint / 2000]],
                    dtype=np.float32,
                )
        return tuple(rows), activations

    def test_validation_selection_refit_and_terminal_test_are_grouped(self) -> None:
        train, train_matrix = self._rows("train", 4)
        validation, validation_matrix = self._rows("validation", 3)
        test, test_matrix = self._rows("test", 3)
        activations = {**train_matrix, **validation_matrix, **test_matrix}
        selection = select_validation_models(
            train_rows=train,
            validation_rows=validation,
            activation_matrices=activations,
            config=self.config,
        )
        self.assertTrue(selection.ready_for_test)
        self.assertIn(selection.primary_layer, {0, 1})
        self.assertEqual(len(selection.selected_c_by_layer), 2)
        models = refit_models_for_test(
            train_rows=train,
            validation_rows=validation,
            activation_matrices=activations,
            selection=selection,
            config=self.config,
        )
        result = evaluate_test_models(
            models=models,
            test_rows=test,
            activation_matrices=activations,
            config=self.config,
        )
        aggregate = result.to_dict()
        self.assertEqual(aggregate["test_event_gate"]["passed"], True)
        self.assertEqual(aggregate["test_matched_gate"]["passed"], True)
        self.assertEqual(len(aggregate["complete_test_layer_curve"]), 2)
        self.assertEqual(aggregate["clustered_bootstrap"]["requested_resamples"], 25)

    def test_missing_activations_are_attrition_not_imputation(self) -> None:
        rows, matrices = self._rows("train", 3)
        missing = dict(matrices)
        missing.pop(rows[0].row_id)
        cohort = activation_available_cohort(rows, missing, config=self.config)
        self.assertEqual(cohort.total_primary_rows, 6)
        self.assertEqual(cohort.available_primary_rows, 5)
        self.assertEqual(cohort.missing_activation_rows, 1)
        self.assertNotIn(rows[0].row_id, cohort.matrices)

    def test_metric_requires_both_primary_classes(self) -> None:
        with self.assertRaises(Exception):
            probability_metrics([1, 1], np.asarray([0.7, 0.8]), sample_weight=None)

    def test_feature_set_cannot_be_arbitrary(self) -> None:
        self.assertEqual(FeatureSet.OBSERVABLE.value, "B")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
