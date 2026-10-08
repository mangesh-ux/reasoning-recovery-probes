"""Synthetic-only tests: no original artifact or private value access."""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/completion_forecasting_data_v1.py"
SPEC = importlib.util.spec_from_file_location("completion_data", MODULE_PATH)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class DataAdapterTests(unittest.TestCase):
    def boundary(self, tokens, status, close=None):
        return {"status": status, "close_marker_start": close,
                "terminal_prefix_sha256": adapter.object_sha(tokens if close is None else tokens[:close])}

    def test_marker_at_first_future_token_is_positive(self):
        tokens = [1] * 512 + [9]
        label, status = adapter.completion_label(tokens, [9], [99], "STOPPED_OTHER", self.boundary(tokens, "CLOSE_MARKER_FOUND", 512))
        self.assertEqual((label, status), (1, "CLOSE_WITHIN_4096"))

    def test_marker_at_final_budget_token_is_positive(self):
        tokens = [1] * 4095 + [9]
        self.assertEqual(adapter.completion_label(tokens, [9], [99], "MAX_NEW_TOKENS", self.boundary(tokens, "CLOSE_MARKER_FOUND", 4095))[0], 1)

    def test_already_closed_is_not_a_negative(self):
        tokens = [1] * 511 + [9]
        self.assertEqual(adapter.completion_label(tokens, [9], [99], "STOPPED_OTHER", self.boundary(tokens, "CLOSE_MARKER_FOUND", 511)), (None, "ALREADY_CLOSED"))

    def test_real_cap_is_negative(self):
        tokens = [1] * 4096
        self.assertEqual(adapter.completion_label(tokens, [9], [99], "MAX_NEW_TOKENS", self.boundary(tokens, "CAP_PREFIX"))[0], 0)

    def test_short_or_early_eos_is_unknown(self):
        for tokens, termination in (([1] * 512 + [99], "EOS"), ([1] * 600, "MAX_NEW_TOKENS")):
            self.assertIsNone(adapter.completion_label(tokens, [9], [99], termination, self.boundary(tokens, "EARLY_NO_CLOSE"))[0])

    def test_inconsistent_boundary_is_not_repaired(self):
        tokens = [1] * 4096
        with self.assertRaises(RuntimeError):
            adapter.completion_label(tokens, [9], [99], "MAX_NEW_TOKENS", self.boundary(tokens, "CLOSE_MARKER_FOUND"))

    def test_lexical_counts_use_only_trailing_prefix(self):
        first = adapter.lexical_counts([2] * 384 + [3] * 128)
        second = adapter.lexical_counts([7] * 384 + [3] * 128)
        self.assertEqual(first, second)
        self.assertEqual(sum(first), 128)
        self.assertEqual(len(first), 512)

    def test_feature_projection_ignores_poisoned_outcome(self):
        class Poison(dict):
            def __getitem__(self, key):
                if key in ("evaluation", "transition_label", "terminal_evaluation_reference"):
                    raise AssertionError("Future outcome read")
                return super().__getitem__(key)
            def get(self, key, default=None):
                if key in ("evaluation", "transition_label", "terminal_evaluation_reference"):
                    raise AssertionError("Future outcome read")
                return super().get(key, default)
        receipt = Poison(status="COMPLETED", checkpoint_id="row", rollout_id="rollout", readout={"observable_scores": {}}, extraction={"status": "MISSING_BOXED_ANSWER"})
        result = adapter.observation(receipt, {"problem_id": "problem", "problem_token_count": 5, "difficulty": 7.0, "topic": "algebra"}, 512)
        self.assertIsNone(result.mean_answer_token_logprob)
        self.assertNotIn("target", result.__dict__)

    def test_problem_floor_cannot_be_met_by_repeated_rollouts(self):
        rows = [{"target": y, "problem_id": "same", "difficulty": 7.0} for y in (0, 1) for _ in range(100)]
        self.assertFalse(adapter.support(rows, "train")["class_support_gate_passed"])
        self.assertEqual(adapter.support(rows, "train")["mixed_label_problems"], 1)

    def test_feature_whitelist_has_eleven_fields(self):
        self.assertEqual(len(adapter.FEATURE_NAMES), 11)
        self.assertTrue(all(not any(x in name for x in ("target", "terminal", "evaluator", "seed")) for name in adapter.FEATURE_NAMES))

    def test_folds_are_problem_grouped_and_order_independent(self):
        records = [{"problem_id": f"problem-{difficulty}-{i}", "difficulty": difficulty} for difficulty in (7.0, 8.0) for i in range(10)]
        first = adapter.fold_assignment(records)
        self.assertEqual(first, adapter.fold_assignment(list(reversed(records))))
        self.assertEqual(Counter(first.values()), {0: 4, 1: 4, 2: 4, 3: 4, 4: 4})


if __name__ == "__main__":
    unittest.main()
