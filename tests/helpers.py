"""Small offline fixtures shared by P0 unit tests."""

from __future__ import annotations

from reasoning_recovery.config import PilotConfig
from reasoning_recovery.dataset import DatasetSnapshot, ProblemRecord


def pilot_config(*, checkpoint_positions: list[int] | None = None) -> PilotConfig:
    return PilotConfig.from_mapping(
        {
            "schema_version": 1,
            "study": {
                "name": "test",
                "protocol_id": "p0-v1-draft",
                "feasibility_only": True,
            },
            "dataset": {
                "id": "HuggingFaceH4/MATH-500",
                "revision": None,
                "split": "test",
                "problem_field": "problem",
                "answer_field": "answer",
                "id_field": "unique_id",
                "selection_method": "sha256-problem-id-source-index-v1",
                "selection_seed": 7,
                "pilot_size": 2,
            },
            "model": {
                "id": "Qwen/Qwen3-1.7B",
                "revision": None,
                "tokenizer_revision": None,
                "dtype": "bfloat16",
                "device": "cuda:0",
                "enable_thinking": True,
            },
            "prompt": {"template": "Solve this: {problem}"},
            "generation": {
                "max_new_tokens": 16,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "rollout_seeds": [101, 102],
                "checkpoint_token_positions": checkpoint_positions or [2, 4, 10],
            },
            "forced_answer": {
                "protocol_id": "append-close-think-greedy-v1",
                "close_think_marker_text": "</think>",
                "close_think_text": "</think>\n\n",
                "max_new_tokens": 8,
                "do_sample": False,
            },
            "answer_extraction": {"policy": "last-balanced-boxed-only-v1"},
            "analysis": {
                "primary_label_policy_id": "p0-primary-label-censoring-v1",
                "require_checkpoint_available": True,
                "required_base_termination_status": "EOS",
                "require_base_close_think_marker": True,
                "required_forced_termination_status": "EOS",
                "evaluable_statuses": ["CORRECT", "INCORRECT"],
            },
            "artifacts": {"root": "artifacts"},
        }
    )


def records() -> tuple[ProblemRecord, ...]:
    return (
        ProblemRecord(
            problem_id="p-1",
            source_index=0,
            problem="What is 2 + 2?",
            reference_answer="4",
            metadata={"subject": "Algebra", "level": 1},
        ),
        ProblemRecord(
            problem_id="p-2",
            source_index=1,
            problem="What is 3 + 3?",
            reference_answer="6",
            metadata={"subject": "Algebra", "level": 1},
        ),
    )


def snapshot() -> DatasetSnapshot:
    return DatasetSnapshot(
        dataset_id="HuggingFaceH4/MATH-500",
        requested_revision=None,
        resolved_revision="a" * 40,
        split="test",
        observed_fingerprint="fixture-fingerprint",
        records=records(),
    )
