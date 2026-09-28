from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reasoning_recovery import cli
from tests.helpers import pilot_config, snapshot


class CliCampaignTests(unittest.TestCase):
    def test_bounded_and_full_invocations_share_the_full_campaign_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            config_path = repo_root / "configs" / "pilot.yaml"
            config_path.parent.mkdir()
            config_path.write_text("fixture", encoding="utf-8")
            (repo_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

            config = pilot_config()
            dataset_snapshot = snapshot()
            manifest = dataset_snapshot.selection_manifest(config=config, limit=2)
            captured: list[dict[str, object]] = []

            def capture_stage(**kwargs: object) -> None:
                captured.append(kwargs)

            common = {
                "confirm_run": True,
                "config": config_path,
                "manifest": Path("manifests") / "full-campaign.json",
                "max_rollouts": None,
            }
            source_identity = {"commit": "b" * 40, "is_clean": True}
            with (
                patch.object(cli, "load_config", return_value=config),
                patch.object(cli, "source_git_identity", return_value=source_identity),
                patch.object(cli, "load_selection_manifest", return_value=manifest),
                patch.object(cli, "load_math500_snapshot", return_value=dataset_snapshot),
                patch.object(cli, "_run_campaign_stage", side_effect=capture_stage),
            ):
                cli._run(argparse.Namespace(**common, max_problems=1))
                cli._run(argparse.Namespace(**common, max_problems=None))

            self.assertEqual(len(captured), 2)
            first, later = captured
            first_layout = first["layout"]
            later_layout = later["layout"]
            self.assertEqual(first_layout.run_id, later_layout.run_id)
            self.assertEqual(
                first["campaign_scope"],
                later["campaign_scope"],
            )
            self.assertEqual(
                first["campaign_scope"],
                {
                    "selection_manifest_sha256": cli.sha256_json(manifest),
                    "problem_ids": ["p-1", "p-2"],
                    "source_indexes": [0, 1],
                    "rollout_seeds": [101, 102],
                    "checkpoint_token_positions": [2, 4, 10],
                },
            )
            self.assertEqual(first["invocation_scope"]["problem_ids"], ["p-1"])
            self.assertEqual(later["invocation_scope"]["problem_ids"], ["p-1", "p-2"])

    def test_analyze_uses_manifest_scope_without_reloading_the_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            config_path = repo_root / "configs" / "pilot.yaml"
            config_path.parent.mkdir()
            config_path.write_text("fixture", encoding="utf-8")
            (repo_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

            config = pilot_config()
            manifest = snapshot().selection_manifest(config=config, limit=2)
            captured: dict[str, object] = {}

            def capture_analysis(layout: object, **kwargs: object) -> dict[str, object]:
                captured["layout"] = layout
                captured.update(kwargs)
                return {"record_type": "P0_DERIVED_ANALYSIS", "run_id": "fixture"}

            with (
                patch.object(cli, "load_config", return_value=config),
                patch.object(cli, "load_selection_manifest", return_value=manifest),
                patch.object(cli, "analyze_p0_run", side_effect=capture_analysis),
                patch.object(cli, "format_p0_analysis", return_value="fixture analysis"),
            ):
                cli._analyze(
                    argparse.Namespace(
                        config=config_path,
                        manifest=Path("manifests") / "full-campaign.json",
                        max_problems=1,
                        max_rollouts=None,
                    )
                )

            self.assertEqual(captured["planned_problem_ids"], ("p-1",))
            self.assertEqual(
                captured["planned_rollout_ids"],
                (cli.rollout_id(captured["layout"].manifest_hash, "p-1", 101),
                 cli.rollout_id(captured["layout"].manifest_hash, "p-1", 102)),
            )
            self.assertEqual(captured["checkpoint_positions"], (2, 4, 10))

    def test_runtime_qualification_records_load_only_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            config_path = repo_root / "configs" / "pilot.yaml"
            config_path.parent.mkdir()
            config_path.write_text("fixture", encoding="utf-8")
            (repo_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            config = pilot_config()

            class RuntimeFixture:
                provenance = {
                    "memory_after_load": {"peak_reserved_bytes": 123},
                    "model": {"resolved_revision": "b" * 40},
                }
                closed = False

                def validate_forced_cue(
                    self, *, close_marker_text: str, cue_text: str
                ) -> tuple[tuple[int, ...], tuple[int, ...]]:
                    self.close_marker_text = close_marker_text
                    self.cue_text = cue_text
                    return (1,), (1, 2)

                def tokenize_prompt(self, user_prompt: str) -> SimpleNamespace:
                    self.user_prompt = user_prompt
                    return SimpleNamespace(token_ids=(11, 12), token_ids_sha256="c" * 64)

                def close(self) -> None:
                    self.closed = True

            runtime = RuntimeFixture()
            with (
                patch.object(cli, "load_config", return_value=config),
                patch.object(
                    cli,
                    "source_git_identity",
                    return_value={"commit": "d" * 40, "is_clean": True},
                ),
                patch.object(cli.TransformersRuntime, "load", return_value=runtime),
            ):
                cli._qualify_runtime(argparse.Namespace(config=config_path))

            qualification_files = tuple((repo_root / "artifacts" / "qualification").glob("*.json"))
            self.assertEqual(len(qualification_files), 1)
            self.assertTrue(runtime.closed)
            self.assertEqual(runtime.close_marker_text, "</think>")
            self.assertEqual(runtime.cue_text, "</think>\n\n")


if __name__ == "__main__":
    unittest.main()
