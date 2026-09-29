"""Offline receipt-contract tests for isolated V2 collection orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
import unittest

import torch

from reasoning_recovery.v2_activations import ActivationExtraction, CudaMemorySnapshot
from reasoning_recovery.v2_artifacts import open_v2_run_layout
from reasoning_recovery.v2_config import load_v2_config
from reasoning_recovery.v2_dataset import V2ProblemRecord
from reasoning_recovery.v2_runtime import (
    V2DecodingParameters,
    V2GeneratedSequence,
    V2TokenizedPrompt,
)
from reasoning_recovery.v2_runner import execute_v2_collection, load_v2_analysis_inputs


_MEMORY = CudaMemorySnapshot(0, 0, 0, 0)


@dataclass(frozen=True)
class _FakeReadout:
    generation: V2GeneratedSequence

    def to_dict(self, *, include_input_token_ids: bool = True):
        payload = self.generation.to_dict(include_input_token_ids=include_input_token_ids)
        payload["observable_scores"] = {
            "mean_answer_token_logprob": -0.1,
            "min_answer_token_logprob": -0.1,
            "answer_token_count": 1,
            "first_readout_token_entropy_nats": 0.2,
            "first_readout_token_probability_margin": 0.7,
        }
        return payload


class _FakeRuntime:
    def __init__(self) -> None:
        self.base_calls = 0
        self.readout_calls = 0
        self.activation_calls = 0
        self.torch = torch
        self.provenance = {"fake": True}
        self.eos_token_ids = (0,)

    def control_token_ids(self, text: str):
        return {"</think>": (9,), "</think>\n\n": (9, 10)}[text]

    def tokenize_prompt(self, problem: str):
        return V2TokenizedPrompt(
            user_prompt=problem,
            chat_prompt_text=problem,
            token_ids=(11, 12),
            token_ids_sha256="bb7f67fd4e5c2b1772bc718df78bcf6376210500f7bd83c4f22c6c37d898a454",
            problem_token_count=2,
        )

    def generate_reasoning(self, input_token_ids, *, seed: int, close_think_token_ids):
        self.base_calls += 1
        return V2GeneratedSequence(
            input_token_ids=tuple(input_token_ids),
            generated_token_ids=(4, 9),
            generated_text="reasoning</think>",
            elapsed_seconds=0.01,
            termination_status="THINK_CLOSE_MARKER",
            memory_before=_MEMORY,
            memory_after=_MEMORY,
            decoding=V2DecodingParameters("fake_base", 3, True, 0.6, 0.95, 20, 0.0, seed),
        )

    def generate_readout(self, input_token_ids):
        self.readout_calls += 1
        generation = V2GeneratedSequence(
            input_token_ids=tuple(input_token_ids),
            generated_token_ids=(5, 0),
            generated_text="\\boxed{1}",
            elapsed_seconds=0.01,
            termination_status="EOS",
            memory_before=_MEMORY,
            memory_after=_MEMORY,
            decoding=V2DecodingParameters("fake_readout", 64, False, None, None, None, None, None),
        )
        return _FakeReadout(generation)

    def extract_checkpoint_activation(self, activation_input):
        self.activation_calls += 1
        matrix = torch.ones((28, 2048), dtype=torch.bfloat16)
        from reasoning_recovery.v2_activations import activation_matrix_sha256
        return ActivationExtraction(
            matrix=matrix,
            layer_count=28,
            hidden_size=2048,
            matrix_shape=(28, 2048),
            matrix_dtype=str(matrix.dtype),
            matrix_sha256=activation_matrix_sha256(matrix, torch=torch),
            raw_matrix_bytes=matrix.numel() * matrix.element_size(),
            input_sha256=activation_input.input_sha256,
            elapsed_seconds=0.01,
            memory_before=_MEMORY,
            memory_after=_MEMORY,
        )


class V2RunnerTests(unittest.TestCase):
    def test_exact_saved_prefix_receipts_resume_without_regeneration(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        frozen = load_v2_config(repo / "configs" / "recovery_v2_frozen.yaml")
        generation = replace(
            frozen.generation,
            reasoning_max_new_tokens=3,
            checkpoint_token_positions=(1,),
            rollout_indices=(0,),
        )
        config = replace(frozen, generation=generation)
        record = V2ProblemRecord(
            problem_id="synthetic-problem",
            source_index=0,
            question="Synthetic question",
            reference_answer="\\boxed{1}",
            difficulty=7.0,
            topic="algebra",
            duplicate_cluster_sha256="cluster",
        )
        manifest = {"records": [{"problem_id": record.problem_id}]}
        split = {"assignments": [{"problem_id": record.problem_id, "split": "train"}]}
        source = {"commit": "a" * 40, "tree": "b" * 40, "is_clean": True}
        with tempfile.TemporaryDirectory() as directory:
            layout = open_v2_run_layout(
                Path(directory), config_hash=config.config_hash, manifest=manifest, split=split, source_identity=source
            )
            runtime = _FakeRuntime()
            first = execute_v2_collection(
                config=config, records=(record,), split="train", layout=layout,
                runtime=runtime, evaluator=None, source_identity=source,
            )
            self.assertEqual(first.completed_base_rollouts, 1)
            self.assertEqual(first.completed_checkpoint_readouts, 1)
            self.assertEqual(first.completed_activations, 1)
            self.assertEqual((runtime.base_calls, runtime.readout_calls, runtime.activation_calls), (1, 2, 1))
            second = execute_v2_collection(
                config=config, records=(record,), split="train", layout=layout,
                runtime=runtime, evaluator=None, source_identity=source,
            )
            self.assertEqual(second.completed_base_rollouts, 1)
            self.assertEqual((runtime.base_calls, runtime.readout_calls, runtime.activation_calls), (1, 2, 1))
            inputs = load_v2_analysis_inputs(config=config, records=(record,), layout=layout)
            self.assertEqual(inputs.raw_checkpoint_receipt_count, 1)
            self.assertEqual(len(inputs.rows), 0)  # evaluator-none remains non-evaluable, not incorrect


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
