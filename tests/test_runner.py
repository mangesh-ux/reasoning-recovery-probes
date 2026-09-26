from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reasoning_recovery.artifacts import open_run_layout, rollout_id
from reasoning_recovery.metrics import summarize_run
from reasoning_recovery.model_runtime import (
    DecodingParameters,
    GeneratedSequence,
    MemorySnapshot,
    TokenizedPrompt,
)
from reasoning_recovery.runner import execute_p0_run
from tests.helpers import pilot_config, records, snapshot


class FakeBackend:
    name = "fake-math"
    version = "test"

    def parse(self, text: str) -> list[str]:
        cleaned = text.replace("\\boxed{", "").replace("}", "").strip()
        return [cleaned] if cleaned else []

    def verify(self, reference: list[str], candidate: list[str]) -> bool:
        return reference == candidate


class FakeRuntime:
    eos_token_ids = (99,)

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], DecodingParameters]] = []

    def tokenize_prompt(self, user_prompt: str) -> TokenizedPrompt:
        return TokenizedPrompt(
            user_prompt=user_prompt,
            chat_prompt_text=f"chat:{user_prompt}",
            token_ids=(11, 12),
            token_ids_sha256="prompt-hash",
        )

    def generate(
        self, input_token_ids: tuple[int, ...], decoding: DecodingParameters
    ) -> GeneratedSequence:
        supplied = tuple(input_token_ids)
        self.calls.append((supplied, decoding))
        if decoding.request_kind == "BASE_STOCHASTIC":
            generated_ids = (1, 2, 90, 91, 3, 4)
            text = r"reasoning </think> \boxed{4}"
        else:
            generated_ids = (5,)
            text = r"\boxed{5}"
        memory = MemorySnapshot(1, 2, 3, 4)
        return GeneratedSequence(
            input_token_ids=supplied,
            generated_token_ids=generated_ids,
            generated_text=text,
            elapsed_seconds=0.25,
            termination_status="EOS",
            memory_before=memory,
            memory_after=memory,
            decoding=decoding,
        )


class FailingRuntime(FakeRuntime):
    def generate(
        self, input_token_ids: tuple[int, ...], decoding: DecodingParameters
    ) -> GeneratedSequence:
        self.calls.append((tuple(input_token_ids), decoding))
        raise RuntimeError("synthetic runtime failure")


class RunnerTests(unittest.TestCase):
    def _layout(self, root: Path):
        config = pilot_config()
        manifest = snapshot().selection_manifest(config=config, limit=1)
        return config, open_run_layout(
            root,
            manifest=manifest,
            config_hash=config.config_hash,
            execution_scope={"problems": ["p-1"], "seeds": [101]},
        )

    def test_exact_prefix_is_used_once_and_resume_does_not_regenerate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, layout = self._layout(Path(temporary))
            runtime = FakeRuntime()
            execution = execute_p0_run(
                config=config,
                records=records()[:1],
                rollout_seeds=(101,),
                layout=layout,
                runtime=runtime,
                evaluator=FakeBackend(),
                close_marker_token_ids=(90, 91),
                close_think_cue_token_ids=(90, 91, 92),
                runtime_provenance_hash="a" * 64,
                source_git_commit="b" * 40,
            )
            self.assertEqual(len(runtime.calls), 2)
            base_input, base_decoding = runtime.calls[0]
            forced_input, forced_decoding = runtime.calls[1]
            self.assertEqual(base_input, (11, 12))
            self.assertTrue(base_decoding.do_sample)
            self.assertEqual(forced_input, (11, 12, 1, 2, 90, 91, 92))
            self.assertFalse(forced_decoding.do_sample)

            checkpoint_receipts = list((layout.run_root / "raw" / "checkpoints").glob("*.json"))
            self.assertEqual(len(checkpoint_receipts), 3)
            self.assertEqual(len(execution.planned_rollout_ids), 1)

            execute_p0_run(
                config=config,
                records=records()[:1],
                rollout_seeds=(101,),
                layout=layout,
                runtime=runtime,
                evaluator=FakeBackend(),
                close_marker_token_ids=(90, 91),
                close_think_cue_token_ids=(90, 91, 92),
                runtime_provenance_hash="a" * 64,
                source_git_commit="b" * 40,
            )
            self.assertEqual(len(runtime.calls), 2)
            summary = summarize_run(
                layout,
                planned_problem_ids=execution.planned_problem_ids,
                planned_rollout_ids=execution.planned_rollout_ids,
                checkpoint_positions=config.generation.checkpoint_token_positions,
            )
            self.assertEqual(summary["transition_counts"]["W_TO_C"], 1)
            self.assertEqual(summary["checkpoint_counts_by_position"]["4"]["AFTER_THINK_CLOSE"], 1)

    def test_failed_base_is_preserved_and_children_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, layout = self._layout(Path(temporary))
            runtime = FailingRuntime()
            execute_p0_run(
                config=config,
                records=records()[:1],
                rollout_seeds=(101,),
                layout=layout,
                runtime=runtime,
                evaluator=FakeBackend(),
                close_marker_token_ids=(90, 91),
                close_think_cue_token_ids=(90, 91, 92),
                runtime_provenance_hash="a" * 64,
                source_git_commit="b" * 40,
            )
            self.assertEqual(len(runtime.calls), 1)
            checkpoint_contents = [
                path.read_text(encoding="utf-8")
                for path in (layout.run_root / "raw" / "checkpoints").glob("*.json")
            ]
            self.assertTrue(all("PARENT_BASE_FAILED" in content for content in checkpoint_contents))

    def test_dangling_base_intent_is_marked_unknown_without_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, layout = self._layout(Path(temporary))
            identity = rollout_id(layout.manifest_hash, "p-1", 101)
            layout.ledger.append("INTENT_STARTED", intent_id=f"intent:{identity}")
            runtime = FakeRuntime()
            execute_p0_run(
                config=config,
                records=records()[:1],
                rollout_seeds=(101,),
                layout=layout,
                runtime=runtime,
                evaluator=FakeBackend(),
                close_marker_token_ids=(90, 91),
                close_think_cue_token_ids=(90, 91, 92),
                runtime_provenance_hash="a" * 64,
                source_git_commit="b" * 40,
            )
            self.assertEqual(runtime.calls, [])
            receipt = layout.read_rollout(identity)
            assert receipt is not None
            self.assertEqual(receipt["status"], "INTERRUPTED_UNKNOWN")


if __name__ == "__main__":
    unittest.main()
