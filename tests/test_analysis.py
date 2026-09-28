from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reasoning_recovery.analysis import (
    analyze_p0_run,
    format_p0_analysis,
    write_p0_analysis_report,
)
from reasoning_recovery.artifacts import open_run_layout, read_json, write_immutable_json


class P0AnalysisTests(unittest.TestCase):
    def test_aggregate_report_preserves_denominators_stages_and_clustered_contrasts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = _layout(Path(temporary))
            _write_fixture_receipts(layout)

            report = analyze_p0_run(
                layout,
                planned_problem_ids=("p-1", "p-2"),
                planned_rollout_ids=("r-1", "r-2", "r-3", "r-4"),
                checkpoint_positions=(2, 4),
            )

            accuracy = report["final_answer_accuracy"]
            self.assertEqual(accuracy["evaluable_denominator"], 3)
            self.assertEqual(accuracy["correct_count"], 2)
            self.assertAlmostEqual(accuracy["accuracy"], 2 / 3)
            self.assertEqual(
                accuracy["analytic_eligibility_excluded_completed_base_count"], 1
            )

            lengths = report["base_trajectory_length_tokens"]
            self.assertEqual(lengths["count"], 4)
            self.assertEqual(lengths["p50"], 20)
            self.assertEqual(lengths["max"], 40)

            checkpoints = report["checkpoint_receipt_accounting"]
            self.assertEqual(checkpoints["expected_receipt_count"], 8)
            self.assertEqual(checkpoints["recorded_receipt_count"], 7)
            self.assertEqual(checkpoints["missing_expected_receipt_count"], 1)
            self.assertEqual(
                checkpoints["by_checkpoint_position"]["2"]["recorded_receipts"], 4
            )

            stages = report["stage_accounting"]
            self.assertEqual(stages["base"]["generation_cap_count"], 1)
            self.assertEqual(stages["forced_checkpoint"]["failure"]["explicit_cuda_oom_count"], 1)
            self.assertEqual(
                stages["base"]["generation"]["generated_token_total"], 100
            )
            self.assertAlmostEqual(
                stages["base"]["generation"]["tokens_per_second"]["aggregate"], 10.0
            )

            vram = report["vram_bytes"]
            self.assertEqual(vram["model_load"]["status"], "RECORDED")
            self.assertEqual(vram["model_load"]["peak_reserved_bytes"], 512)
            self.assertEqual(vram["base_generation"]["peak_allocated_bytes"], 140)
            self.assertEqual(
                vram["forced_checkpoint_generation"]["peak_reserved_bytes"], 380
            )

            at_two = report["transitions"]["rates_by_checkpoint"]["2"]
            self.assertEqual(at_two["evaluable_transition_denominator"], 3)
            self.assertEqual(at_two["transition_counts"]["W_TO_C"], 1)
            self.assertEqual(at_two["transition_counts"]["W_TO_W"], 1)
            self.assertEqual(at_two["transition_counts"]["C_TO_C"], 1)
            self.assertAlmostEqual(at_two["recovery_rate_given_wrong_checkpoint"], 0.5)

            contrasts = report["transitions"][
                "same_problem_same_checkpoint_w_to_c_vs_w_to_w"
            ]["all_declared_checkpoints"]
            self.assertEqual(contrasts["groups_with_both_w_to_c_and_w_to_w"], 1)
            self.assertEqual(contrasts["w_to_c_events_in_mixed_groups"], 1)
            self.assertEqual(contrasts["w_to_w_events_in_mixed_groups"], 1)
            concentration = report["transitions"]["event_concentration"]["W_TO_C"]
            self.assertEqual(concentration["event_count"], 1)
            self.assertEqual(concentration["distinct_problem_count"], 1)
            observed_at_two = report["transitions"]["observed_before_analytic_censoring"][
                "rates_by_checkpoint"
            ]["2"]
            self.assertEqual(observed_at_two["transition_counts"]["W_TO_C"], 2)

            eligibility = report["analytic_eligibility"]
            self.assertEqual(eligibility["base_receipts"]["excluded_count"], 1)
            self.assertEqual(
                eligibility["base_receipts"]["status_counts"]["NOT_DECLARED_LEGACY_ELIGIBLE"],
                3,
            )
            self.assertEqual(
                report["transitions"]["integrity_and_exclusion_counts"][
                    "PARENT_BASE_ANALYTICALLY_EXCLUDED"
                ],
                1,
            )
            self.assertEqual(
                report["transitions"]["integrity_and_exclusion_counts"][
                    "analytic_exclusion_reason_counts"
                ]["BASE_TERMINATION_NOT_NATURAL_EOS"],
                1,
            )

    def test_report_is_content_addressed_and_omits_raw_receipt_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = _layout(Path(temporary))
            _write_fixture_receipts(layout)

            first = write_p0_analysis_report(
                layout,
                planned_problem_ids=("p-1", "p-2"),
                planned_rollout_ids=("r-1", "r-2", "r-3", "r-4"),
                checkpoint_positions=(2, 4),
            )
            second = write_p0_analysis_report(
                layout,
                planned_problem_ids=("p-1", "p-2"),
                planned_rollout_ids=("r-1", "r-2", "r-3", "r-4"),
                checkpoint_positions=(2, 4),
            )

            self.assertEqual(first, second)
            payload = read_json(first)
            rendered = first.read_text(encoding="utf-8")
            self.assertEqual(payload["record_type"], "P0_DERIVED_ANALYSIS")
            self.assertNotIn("private fixture prompt", rendered)
            self.assertNotIn("private completion", rendered)
            self.assertIn("final-answer accuracy", format_p0_analysis(payload))

    def test_malformed_explicit_eligibility_fails_closed_without_erasing_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = _layout(Path(temporary))
            layout.write_rollout(
                "r-1",
                _base_receipt(
                    rollout_id="r-1",
                    problem_id="p-1",
                    final_correct=True,
                    tokens=4,
                    elapsed=1.0,
                    analytic_eligibility="not-a-boolean-or-mapping",
                ),
            )
            layout.write_checkpoint(
                "c-1",
                _checkpoint_receipt(
                    checkpoint_id="c-1",
                    parent_rollout_id="r-1",
                    problem_id="p-1",
                    checkpoint_token=2,
                    forced_correct=False,
                    final_correct=True,
                ),
            )

            report = analyze_p0_run(
                layout,
                planned_problem_ids=("p-1",),
                planned_rollout_ids=("r-1",),
                checkpoint_positions=(2,),
            )

            self.assertEqual(report["final_answer_accuracy"]["evaluable_denominator"], 0)
            self.assertEqual(
                report["analytic_eligibility"]["base_receipts"]["status_counts"][
                    "MALFORMED_EXPLICIT_FIELD_EXCLUDED"
                ],
                1,
            )
            self.assertEqual(
                report["transitions"]["rates_by_checkpoint"]["2"][
                    "evaluable_transition_denominator"
                ],
                0,
            )
            self.assertEqual(
                report["transitions"]["integrity_and_exclusion_counts"][
                    "PARENT_BASE_ANALYTICALLY_EXCLUDED"
                ],
                1,
            )

    def test_explicit_checkpoint_eligibility_keeps_observed_transition_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = _layout(Path(temporary))
            layout.write_rollout(
                "r-1",
                _base_receipt(
                    rollout_id="r-1",
                    problem_id="p-1",
                    final_correct=True,
                    tokens=4,
                    elapsed=1.0,
                ),
            )
            layout.write_checkpoint(
                "c-1",
                _checkpoint_receipt(
                    checkpoint_id="c-1",
                    parent_rollout_id="r-1",
                    problem_id="p-1",
                    checkpoint_token=2,
                    forced_correct=False,
                    final_correct=True,
                    analytic_eligibility={
                        "eligible": False,
                        "reasons": [
                            {"code": "FORCED_TERMINATION_NOT_NATURAL_EOS"}
                        ],
                    },
                ),
            )

            report = analyze_p0_run(
                layout,
                planned_problem_ids=("p-1",),
                planned_rollout_ids=("r-1",),
                checkpoint_positions=(2,),
            )

            primary = report["transitions"]["primary_analytic"]["rates_by_checkpoint"]["2"]
            observed = report["transitions"]["observed_before_analytic_censoring"][
                "rates_by_checkpoint"
            ]["2"]
            self.assertEqual(primary["evaluable_transition_denominator"], 0)
            self.assertEqual(observed["transition_counts"]["W_TO_C"], 1)
            integrity = report["transitions"]["integrity_and_exclusion_counts"]
            self.assertEqual(integrity["CHECKPOINT_ANALYTICALLY_EXCLUDED"], 1)
            self.assertEqual(
                integrity["analytic_exclusion_reason_counts"][
                    "FORCED_TERMINATION_NOT_NATURAL_EOS"
                ],
                1,
            )


def _layout(root: Path):
    return open_run_layout(
        root,
        manifest={"schema_version": 1, "records": [{"id": "fixture"}]},
        config_hash="c" * 64,
        campaign_scope={"fixture": True},
    )


def _write_fixture_receipts(layout) -> None:
    write_immutable_json(
        layout.run_root / "runtime_contract.json",
        {
            "schema_version": 1,
            "runtime": {
                "memory_after_load": {
                    "allocated_bytes": 256,
                    "reserved_bytes": 384,
                    "peak_allocated_bytes": 448,
                    "peak_reserved_bytes": 512,
                }
            },
        },
    )
    layout.write_rollout(
        "r-1",
        _base_receipt(
            rollout_id="r-1",
            problem_id="p-1",
            final_correct=True,
            tokens=10,
            elapsed=1.0,
            termination="MAX_NEW_TOKENS",
            peak_allocated=110,
            peak_reserved=210,
        ),
    )
    layout.write_rollout(
        "r-2",
        _base_receipt(
            rollout_id="r-2",
            problem_id="p-1",
            final_correct=False,
            tokens=20,
            elapsed=2.0,
            peak_allocated=120,
            peak_reserved=220,
        ),
    )
    layout.write_rollout(
        "r-3",
        _base_receipt(
            rollout_id="r-3",
            problem_id="p-2",
            final_correct=True,
            tokens=30,
            elapsed=3.0,
            peak_allocated=130,
            peak_reserved=230,
        ),
    )
    layout.write_rollout(
        "r-4",
        _base_receipt(
            rollout_id="r-4",
            problem_id="p-2",
            final_correct=True,
            tokens=40,
            elapsed=4.0,
            peak_allocated=140,
            peak_reserved=240,
            analytic_eligibility={
                "eligible": False,
                "reasons": [{"code": "BASE_TERMINATION_NOT_NATURAL_EOS"}],
            },
        ),
    )

    layout.write_checkpoint(
        "c-1-2",
        _checkpoint_receipt(
            checkpoint_id="c-1-2",
            parent_rollout_id="r-1",
            problem_id="p-1",
            checkpoint_token=2,
            forced_correct=False,
            final_correct=True,
            tokens=5,
            elapsed=0.5,
            peak_allocated=310,
            peak_reserved=360,
        ),
    )
    layout.write_checkpoint(
        "c-2-2",
        _checkpoint_receipt(
            checkpoint_id="c-2-2",
            parent_rollout_id="r-2",
            problem_id="p-1",
            checkpoint_token=2,
            forced_correct=False,
            final_correct=False,
            tokens=6,
            elapsed=0.6,
            peak_allocated=320,
            peak_reserved=370,
        ),
    )
    layout.write_checkpoint(
        "c-3-2",
        _checkpoint_receipt(
            checkpoint_id="c-3-2",
            parent_rollout_id="r-3",
            problem_id="p-2",
            checkpoint_token=2,
            forced_correct=True,
            final_correct=True,
            tokens=7,
            elapsed=0.7,
            peak_allocated=330,
            peak_reserved=380,
        ),
    )
    layout.write_checkpoint(
        "c-4-2",
        _checkpoint_receipt(
            checkpoint_id="c-4-2",
            parent_rollout_id="r-4",
            problem_id="p-2",
            checkpoint_token=2,
            forced_correct=False,
            final_correct=True,
            tokens=8,
            elapsed=0.8,
        ),
    )
    layout.write_checkpoint(
        "c-1-4",
        _unavailable_checkpoint(parent_rollout_id="r-1", problem_id="p-1", checkpoint_token=4),
    )
    layout.write_checkpoint(
        "c-2-4",
        _failed_checkpoint(parent_rollout_id="r-2", problem_id="p-1", checkpoint_token=4),
    )
    layout.write_checkpoint(
        "c-3-4",
        _checkpoint_receipt(
            checkpoint_id="c-3-4",
            parent_rollout_id="r-3",
            problem_id="p-2",
            checkpoint_token=4,
            forced_correct=None,
            final_correct=True,
            tokens=1,
            elapsed=0.1,
            evaluation_status="NON_EVALUABLE",
        ),
    )


def _base_receipt(
    *,
    rollout_id: str,
    problem_id: str,
    final_correct: bool,
    tokens: int,
    elapsed: float,
    termination: str = "EOS",
    peak_allocated: int = 100,
    peak_reserved: int = 200,
    analytic_eligibility: object | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "BASE_ROLLOUT",
        "rollout_id": rollout_id,
        "problem_id": problem_id,
        "problem": "private fixture prompt",
        "status": "COMPLETED",
        "generation": _generation(
            tokens=tokens,
            elapsed=elapsed,
            termination=termination,
            peak_allocated=peak_allocated,
            peak_reserved=peak_reserved,
        ),
        "final_evaluation": _evaluation(final_correct),
    }
    if analytic_eligibility is not None:
        payload["analytic_eligibility"] = analytic_eligibility
    return payload


def _checkpoint_receipt(
    *,
    checkpoint_id: str,
    parent_rollout_id: str,
    problem_id: str,
    checkpoint_token: int,
    forced_correct: bool | None,
    final_correct: bool,
    tokens: int = 1,
    elapsed: float = 0.1,
    peak_allocated: int = 300,
    peak_reserved: int = 350,
    evaluation_status: str | None = None,
    analytic_eligibility: object | None = None,
) -> dict[str, object]:
    forced_evaluation = _evaluation(forced_correct, status=evaluation_status)
    final_evaluation = _evaluation(final_correct)
    label = _label(forced_correct, final_correct) if evaluation_status is None else None
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "FORCED_CHECKPOINT",
        "checkpoint_id": checkpoint_id,
        "parent_rollout_id": parent_rollout_id,
        "problem_id": problem_id,
        "checkpoint_token": checkpoint_token,
        "status": "COMPLETED",
        "availability_status": "AVAILABLE",
        "forced_generation": _generation(
            tokens=tokens,
            elapsed=elapsed,
            termination="EOS",
            peak_allocated=peak_allocated,
            peak_reserved=peak_reserved,
            text="private completion",
        ),
        "forced_evaluation": forced_evaluation,
        "final_evaluation_reference": final_evaluation,
        "transition_label": label,
    }
    if analytic_eligibility is not None:
        payload["analytic_eligibility"] = analytic_eligibility
    return payload


def _unavailable_checkpoint(
    *, parent_rollout_id: str, problem_id: str, checkpoint_token: int
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "FORCED_CHECKPOINT",
        "checkpoint_id": f"{parent_rollout_id}-{checkpoint_token}",
        "parent_rollout_id": parent_rollout_id,
        "problem_id": problem_id,
        "checkpoint_token": checkpoint_token,
        "status": "UNAVAILABLE",
        "availability_status": "AFTER_THINK_CLOSE",
    }


def _failed_checkpoint(
    *, parent_rollout_id: str, problem_id: str, checkpoint_token: int
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "FORCED_CHECKPOINT",
        "checkpoint_id": f"{parent_rollout_id}-{checkpoint_token}",
        "parent_rollout_id": parent_rollout_id,
        "problem_id": problem_id,
        "checkpoint_token": checkpoint_token,
        "status": "FAILED",
        "availability_status": "AVAILABLE",
        "failure": {"error_type": "OutOfMemoryError", "cuda_oom": True},
    }


def _generation(
    *,
    tokens: int,
    elapsed: float,
    termination: str,
    peak_allocated: int,
    peak_reserved: int,
    text: str = "",
) -> dict[str, object]:
    return {
        "generated_token_count": tokens,
        "elapsed_seconds": elapsed,
        "termination_status": termination,
        "generated_text": text,
        "memory_after": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
    }


def _evaluation(correct: bool | None, *, status: str | None = None) -> dict[str, object]:
    if status is not None:
        return {"status": status, "correct": None}
    return {"status": "CORRECT" if correct else "INCORRECT", "correct": correct}


def _label(forced_correct: bool | None, final_correct: bool) -> str | None:
    if forced_correct is None:
        return None
    return {
        (False, True): "W_TO_C",
        (False, False): "W_TO_W",
        (True, True): "C_TO_C",
        (True, False): "C_TO_W",
    }[(forced_correct, final_correct)]


if __name__ == "__main__":
    unittest.main()
