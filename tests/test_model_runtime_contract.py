from __future__ import annotations

import unittest

from reasoning_recovery.model_runtime import (
    DecodingParameters,
    GeneratedSequence,
    MemorySnapshot,
    TransformersRuntime,
)
from tests.helpers import pilot_config


class ModelRuntimeContractTests(unittest.TestCase):
    def test_decoding_modes_remain_separate(self) -> None:
        config = pilot_config().generation
        base = DecodingParameters.base(config, 101)
        forced = DecodingParameters.forced(8)
        self.assertTrue(base.do_sample)
        self.assertEqual(base.seed, 101)
        self.assertFalse(forced.do_sample)
        self.assertIsNone(forced.seed)
        self.assertIsNone(forced.temperature)

    def test_child_serialization_can_avoid_repeating_input_ids(self) -> None:
        memory = MemorySnapshot(1, 2, 3, 4)
        sequence = GeneratedSequence(
            input_token_ids=(1, 2, 3),
            generated_token_ids=(4,),
            generated_text="x",
            elapsed_seconds=0.1,
            termination_status="EOS",
            memory_before=memory,
            memory_after=memory,
            decoding=DecodingParameters.forced(8),
        )
        serialized = sequence.to_dict(include_input_token_ids=False)
        self.assertNotIn("input_token_ids", serialized)
        self.assertEqual(serialized["input_token_count"], 3)

    def test_generation_kwargs_request_no_activation_outputs(self) -> None:
        runtime = TransformersRuntime(
            torch=None,
            tokenizer=type("Tokenizer", (), {"pad_token_id": 0})(),
            model=object(),
            model_config=pilot_config().model,
            device=None,
            context_limit_tokens=128,
            eos_token_ids=(9, 10),
            provenance={},
        )
        kwargs = runtime._generation_kwargs(DecodingParameters.forced(8))
        self.assertFalse(kwargs["output_scores"])
        self.assertFalse(kwargs["output_hidden_states"])
        self.assertFalse(kwargs["output_attentions"])
        self.assertFalse(kwargs["return_dict_in_generate"])
        self.assertEqual(kwargs["eos_token_id"], [9, 10])


if __name__ == "__main__":
    unittest.main()
