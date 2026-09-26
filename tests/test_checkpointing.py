from __future__ import annotations

import unittest

from reasoning_recovery.checkpointing import CheckpointStatus, build_fixed_checkpoints
from reasoning_recovery.forced_answer import build_forced_answer_input


class CheckpointingTests(unittest.TestCase):
    def test_prefix_at_close_marker_start_is_available(self) -> None:
        checkpoints = build_fixed_checkpoints(
            [10, 11, 90, 91, 12],
            [2, 3, 4],
            close_think_token_ids=[90, 91],
        )
        self.assertEqual(checkpoints[0].status, CheckpointStatus.AVAILABLE)
        self.assertEqual(checkpoints[0].prefix_generated_token_ids, (10, 11))
        self.assertEqual(checkpoints[1].status, CheckpointStatus.AFTER_THINK_CLOSE)
        self.assertEqual(checkpoints[2].status, CheckpointStatus.AFTER_THINK_CLOSE)

    def test_short_and_terminal_prefixes_stay_explicit(self) -> None:
        checkpoints = build_fixed_checkpoints(
            [1, 2, 99],
            [2, 3, 4],
            close_think_token_ids=[77],
            eos_token_ids=[99],
        )
        self.assertEqual(checkpoints[0].status, CheckpointStatus.AVAILABLE)
        self.assertEqual(checkpoints[1].status, CheckpointStatus.AFTER_TERMINAL_TOKEN)
        self.assertEqual(checkpoints[2].status, CheckpointStatus.TRAJECTORY_TOO_SHORT)

    def test_forced_input_is_raw_id_concatenation(self) -> None:
        checkpoint = build_fixed_checkpoints(
            [50, 51, 52], [2], close_think_token_ids=[90]
        )[0]
        forced = build_forced_answer_input([7, 8], checkpoint, [90, 91])
        self.assertEqual(forced.prompt_token_ids, (7, 8))
        self.assertEqual(forced.saved_prefix_token_ids, (50, 51))
        self.assertEqual(forced.input_token_ids, (7, 8, 50, 51, 90, 91))


if __name__ == "__main__":
    unittest.main()
