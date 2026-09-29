"""Offline exact-token tests for V2 boundary behavior."""

from __future__ import annotations

import unittest

from reasoning_recovery.v2_boundary import (
    ReasoningBoundaryStatus,
    V2CheckpointStatus,
    construct_fixed_checkpoints,
    construct_reasoning_boundary,
    normalise_extracted_answer_for_agreement,
)


class V2BoundaryTests(unittest.TestCase):
    def test_close_marker_excludes_marker_and_makes_exact_prefix_available(self) -> None:
        boundary = construct_reasoning_boundary(
            [1, 2, 3, 8, 9, 5],
            close_think_token_ids=[8, 9],
            eos_token_ids=[5],
            max_new_tokens=10,
            termination_status="THINK_CLOSE_MARKER",
        )
        self.assertEqual(boundary.status, ReasoningBoundaryStatus.CLOSE_MARKER_FOUND)
        self.assertEqual(boundary.terminal_reasoning_prefix_token_ids, (1, 2, 3))
        checkpoints = construct_fixed_checkpoints(boundary, [2, 3, 4])
        self.assertEqual(checkpoints[0].status, V2CheckpointStatus.AVAILABLE)
        self.assertEqual(checkpoints[1].status, V2CheckpointStatus.AVAILABLE)
        self.assertEqual(checkpoints[2].status, V2CheckpointStatus.AFTER_THINK_CLOSE)

    def test_cap_prefix_is_terminal_but_early_eos_is_not(self) -> None:
        cap = construct_reasoning_boundary(
            [1, 2, 3, 4], close_think_token_ids=[8], eos_token_ids=[5], max_new_tokens=4, termination_status="MAX_NEW_TOKENS"
        )
        early = construct_reasoning_boundary(
            [1, 5], close_think_token_ids=[8], eos_token_ids=[5], max_new_tokens=4, termination_status="EOS"
        )
        self.assertEqual(cap.status, ReasoningBoundaryStatus.CAP_PREFIX)
        self.assertEqual(early.status, ReasoningBoundaryStatus.EARLY_NO_CLOSE)
        self.assertIsNone(early.terminal_reasoning_prefix_token_ids)

    def test_agreement_normalization_does_not_evaluate_math(self) -> None:
        self.assertEqual(normalise_extracted_answer_for_agreement(" \\boxed{  1 + 1 } \n"), "\\boxed{ 1 + 1 }")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
