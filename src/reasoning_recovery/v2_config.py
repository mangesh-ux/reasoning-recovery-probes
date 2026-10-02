"""Strict, typed configuration contract for the frozen V2 study.

This module intentionally has no model, dataset, or evaluator imports. It
turns reviewed YAML into immutable Python values and rejects both unknown
fields and semantic drift from the frozen protocol before any scientific
operation can be started.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from .provenance import sha256_json


class V2ConfigurationError(ValueError):
    """Raised when a V2 configuration is incomplete, mistyped, or not frozen."""


@dataclass(frozen=True)
class V2StudyConfig:
    name: str
    protocol_id: str
    phase: str
    p0_artifact_reuse: str
    source_identity_policy: str


@dataclass(frozen=True)
class V2DifficultyStratum:
    difficulty: float
    count: int


@dataclass(frozen=True)
class V2DatasetConfig:
    dataset_id: str
    revision: str
    split: str
    problem_field: str
    answer_field: str
    difficulty_field: str
    topic_field: str
    stable_id_policy: str
    duplicate_cluster_policy: str
    selection_method: str
    selection_seed: int
    strata: tuple[V2DifficultyStratum, ...]
    answer_validation_policy: str = "all-source-final-answer-v1"


@dataclass(frozen=True)
class V2ModelConfig:
    model_id: str
    revision: str
    tokenizer_revision: str
    dtype: str
    device: str
    enable_thinking: bool
    expected_model_type: str
    expected_num_hidden_layers: int
    expected_hidden_size: int


@dataclass(frozen=True)
class V2RuntimeConfig:
    require_cuda: bool
    require_bfloat16: bool
    expected_python: str
    expected_torch: str
    expected_transformers: str
    expected_datasets: str
    expected_math_verify: str
    expected_sklearn: str
    expected_accelerate: str
    expected_huggingface_hub: str
    expected_safetensors: str
    minimum_free_disk_bytes: int
    local_max_projected_wall_hours: float
    minimum_activation_headroom_bytes: int


@dataclass(frozen=True)
class V2PromptConfig:
    template: str

    def render(self, problem: str) -> str:
        """Render only the frozen problem placeholder.

        The normal formatting API is deliberately avoided because the prompt
        contains LaTeX braces that are not substitution fields.
        """

        if not isinstance(problem, str) or not problem.strip():
            raise V2ConfigurationError("problem must be a non-empty string")
        return self.template.replace("{problem}", problem)


@dataclass(frozen=True)
class V2GenerationConfig:
    reasoning_max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    rollout_seed_policy: str
    rollout_seed_root: int
    rollout_indices: tuple[int, ...]
    checkpoint_token_positions: tuple[int, ...]
    close_think_marker_text: str
    coordinate_policy: str
    early_no_close_policy: str
    terminal_cap_policy: str


@dataclass(frozen=True)
class V2ReadoutConfig:
    protocol_id: str
    cue_text: str
    max_new_tokens: int
    do_sample: bool
    required_termination_status: str
    answer_extraction_policy: str
    observable_feature_policy: str
    previous_readout_policy: str


@dataclass(frozen=True)
class V2ActivationConfig:
    protocol_id: str
    layer_indexing: str
    storage_dtype: str
    storage_format: str
    include_embedding_state: bool
    include_final_norm_state: bool
    input_policy: str
    use_cache: bool
    output_attentions: bool
    retain_full_sequence_layer_tensors: bool


@dataclass(frozen=True)
class V2SplitsConfig:
    policy: str
    split_seed: int
    train_per_difficulty: int
    validation_per_difficulty: int
    test_per_difficulty: int
    duplicate_clusters_must_not_cross_splits: bool
    test_unblinding_policy: str


@dataclass(frozen=True)
class V2FeaturesConfig:
    structural_feature_policy: str
    observable_feature_policy: str
    normalized_checkpoint_denominator: int
    topic_encoding: str
    numeric_missing_policy: str
    previous_agreement_missing_encoding: str
    forbid_future_terminal_gold_evaluator_features: bool


@dataclass(frozen=True)
class V2ProbeConfig:
    family: str
    solver: str
    max_iter: int
    class_weight: None
    regularization_c_grid: tuple[float, ...]
    hyperparameter_selection_metric: str
    hyperparameter_tiebreak: str
    primary_layer_selection_metric: str
    training_problem_weighting: str
    activation_comparison_cohort: str


@dataclass(frozen=True)
class V2AnalysisConfig:
    positive_label: str
    negative_label: str
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    bootstrap_policy: str
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_min_valid_fraction: float
    calibration_bins: int
    report_unweighted_row_sensitivity: bool
    min_train_rows_per_class: int
    min_train_problems_per_class: int
    min_validation_rows_per_class: int
    min_validation_problems_per_class: int
    min_test_rows_per_class: int
    min_test_problems_per_class: int
    min_matched_problem_checkpoint_groups: int
    min_matched_unique_problems: int
    primary_result_estimand: str


@dataclass(frozen=True)
class V2ArtifactsConfig:
    root: str
    public_summary_path: str
    raw_artifacts_publication: str


@dataclass(frozen=True)
class V2HardwarePreflightConfig:
    protocol_id: str
    synthetic_generation_max_new_tokens: int
    synthetic_readout_max_new_tokens: int
    activation_input_token_lengths: tuple[int, ...]
    warmup_iterations: int
    measured_iterations: int


@dataclass(frozen=True)
class V2ExploratoryExtensionConfig:
    status: str
    protocol_requirement: str


@dataclass(frozen=True)
class V2Config:
    """All semantically relevant V2 settings in one immutable object."""

    schema_version: int
    study: V2StudyConfig
    dataset: V2DatasetConfig
    model: V2ModelConfig
    runtime: V2RuntimeConfig
    prompt: V2PromptConfig
    generation: V2GenerationConfig
    readout: V2ReadoutConfig
    activations: V2ActivationConfig
    splits: V2SplitsConfig
    features: V2FeaturesConfig
    probe: V2ProbeConfig
    analysis: V2AnalysisConfig
    artifacts: V2ArtifactsConfig
    hardware_preflight: V2HardwarePreflightConfig
    exploratory_extension: V2ExploratoryExtensionConfig

    @property
    def config_hash(self) -> str:
        """Hash the parsed semantic configuration rather than YAML syntax."""

        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Round-trip to the frozen YAML-shaped semantic representation."""

        return {
            "schema_version": self.schema_version,
            "study": {
                "name": self.study.name,
                "protocol_id": self.study.protocol_id,
                "phase": self.study.phase,
                "p0_artifact_reuse": self.study.p0_artifact_reuse,
                "source_identity_policy": self.study.source_identity_policy,
            },
            "dataset": {
                "id": self.dataset.dataset_id,
                "revision": self.dataset.revision,
                "split": self.dataset.split,
                "problem_field": self.dataset.problem_field,
                "answer_field": self.dataset.answer_field,
                "difficulty_field": self.dataset.difficulty_field,
                "topic_field": self.dataset.topic_field,
                "stable_id_policy": self.dataset.stable_id_policy,
                "duplicate_cluster_policy": self.dataset.duplicate_cluster_policy,
                "selection_method": self.dataset.selection_method,
                "selection_seed": self.dataset.selection_seed,
                "strata": [
                    {"difficulty": item.difficulty, "count": item.count}
                    for item in self.dataset.strata
                ],
                # The original semantic representation and hash stay frozen.
                **(
                    {}
                    if self.dataset.answer_validation_policy == "all-source-final-answer-v1"
                    else {"answer_validation_policy": self.dataset.answer_validation_policy}
                ),
            },
            "model": {
                "id": self.model.model_id,
                "revision": self.model.revision,
                "tokenizer_revision": self.model.tokenizer_revision,
                "dtype": self.model.dtype,
                "device": self.model.device,
                "enable_thinking": self.model.enable_thinking,
                "expected_model_type": self.model.expected_model_type,
                "expected_num_hidden_layers": self.model.expected_num_hidden_layers,
                "expected_hidden_size": self.model.expected_hidden_size,
            },
            "runtime": {
                "require_cuda": self.runtime.require_cuda,
                "require_bfloat16": self.runtime.require_bfloat16,
                "expected_python": self.runtime.expected_python,
                "expected_torch": self.runtime.expected_torch,
                "expected_transformers": self.runtime.expected_transformers,
                "expected_datasets": self.runtime.expected_datasets,
                "expected_math_verify": self.runtime.expected_math_verify,
                "expected_sklearn": self.runtime.expected_sklearn,
                "expected_accelerate": self.runtime.expected_accelerate,
                "expected_huggingface_hub": self.runtime.expected_huggingface_hub,
                "expected_safetensors": self.runtime.expected_safetensors,
                "minimum_free_disk_bytes": self.runtime.minimum_free_disk_bytes,
                "local_max_projected_wall_hours": self.runtime.local_max_projected_wall_hours,
                "minimum_activation_headroom_bytes": self.runtime.minimum_activation_headroom_bytes,
            },
            "prompt": {"template": self.prompt.template},
            "generation": {
                "reasoning_max_new_tokens": self.generation.reasoning_max_new_tokens,
                "do_sample": self.generation.do_sample,
                "temperature": self.generation.temperature,
                "top_p": self.generation.top_p,
                "top_k": self.generation.top_k,
                "min_p": self.generation.min_p,
                "rollout_seed_policy": self.generation.rollout_seed_policy,
                "rollout_seed_root": self.generation.rollout_seed_root,
                "rollout_indices": list(self.generation.rollout_indices),
                "checkpoint_token_positions": list(self.generation.checkpoint_token_positions),
                "close_think_marker_text": self.generation.close_think_marker_text,
                "coordinate_policy": self.generation.coordinate_policy,
                "early_no_close_policy": self.generation.early_no_close_policy,
                "terminal_cap_policy": self.generation.terminal_cap_policy,
            },
            "readout": {
                "protocol_id": self.readout.protocol_id,
                "cue_text": self.readout.cue_text,
                "max_new_tokens": self.readout.max_new_tokens,
                "do_sample": self.readout.do_sample,
                "required_termination_status": self.readout.required_termination_status,
                "answer_extraction_policy": self.readout.answer_extraction_policy,
                "observable_feature_policy": self.readout.observable_feature_policy,
                "previous_readout_policy": self.readout.previous_readout_policy,
            },
            "activations": {
                "protocol_id": self.activations.protocol_id,
                "layer_indexing": self.activations.layer_indexing,
                "storage_dtype": self.activations.storage_dtype,
                "storage_format": self.activations.storage_format,
                "include_embedding_state": self.activations.include_embedding_state,
                "include_final_norm_state": self.activations.include_final_norm_state,
                "input_policy": self.activations.input_policy,
                "use_cache": self.activations.use_cache,
                "output_attentions": self.activations.output_attentions,
                "retain_full_sequence_layer_tensors": self.activations.retain_full_sequence_layer_tensors,
            },
            "splits": {
                "policy": self.splits.policy,
                "split_seed": self.splits.split_seed,
                "train_per_difficulty": self.splits.train_per_difficulty,
                "validation_per_difficulty": self.splits.validation_per_difficulty,
                "test_per_difficulty": self.splits.test_per_difficulty,
                "duplicate_clusters_must_not_cross_splits": self.splits.duplicate_clusters_must_not_cross_splits,
                "test_unblinding_policy": self.splits.test_unblinding_policy,
            },
            "features": {
                "structural_feature_policy": self.features.structural_feature_policy,
                "observable_feature_policy": self.features.observable_feature_policy,
                "normalized_checkpoint_denominator": self.features.normalized_checkpoint_denominator,
                "topic_encoding": self.features.topic_encoding,
                "numeric_missing_policy": self.features.numeric_missing_policy,
                "previous_agreement_missing_encoding": self.features.previous_agreement_missing_encoding,
                "forbid_future_terminal_gold_evaluator_features": self.features.forbid_future_terminal_gold_evaluator_features,
            },
            "probe": {
                "family": self.probe.family,
                "solver": self.probe.solver,
                "max_iter": self.probe.max_iter,
                "class_weight": self.probe.class_weight,
                "regularization_c_grid": list(self.probe.regularization_c_grid),
                "hyperparameter_selection_metric": self.probe.hyperparameter_selection_metric,
                "hyperparameter_tiebreak": self.probe.hyperparameter_tiebreak,
                "primary_layer_selection_metric": self.probe.primary_layer_selection_metric,
                "training_problem_weighting": self.probe.training_problem_weighting,
                "activation_comparison_cohort": self.probe.activation_comparison_cohort,
            },
            "analysis": {
                "positive_label": self.analysis.positive_label,
                "negative_label": self.analysis.negative_label,
                "primary_metrics": list(self.analysis.primary_metrics),
                "secondary_metrics": list(self.analysis.secondary_metrics),
                "bootstrap_policy": self.analysis.bootstrap_policy,
                "bootstrap_resamples": self.analysis.bootstrap_resamples,
                "bootstrap_seed": self.analysis.bootstrap_seed,
                "bootstrap_min_valid_fraction": self.analysis.bootstrap_min_valid_fraction,
                "calibration_bins": self.analysis.calibration_bins,
                "report_unweighted_row_sensitivity": self.analysis.report_unweighted_row_sensitivity,
                "min_train_rows_per_class": self.analysis.min_train_rows_per_class,
                "min_train_problems_per_class": self.analysis.min_train_problems_per_class,
                "min_validation_rows_per_class": self.analysis.min_validation_rows_per_class,
                "min_validation_problems_per_class": self.analysis.min_validation_problems_per_class,
                "min_test_rows_per_class": self.analysis.min_test_rows_per_class,
                "min_test_problems_per_class": self.analysis.min_test_problems_per_class,
                "min_matched_problem_checkpoint_groups": self.analysis.min_matched_problem_checkpoint_groups,
                "min_matched_unique_problems": self.analysis.min_matched_unique_problems,
                "primary_result_estimand": self.analysis.primary_result_estimand,
            },
            "artifacts": {
                "root": self.artifacts.root,
                "public_summary_path": self.artifacts.public_summary_path,
                "raw_artifacts_publication": self.artifacts.raw_artifacts_publication,
            },
            "hardware_preflight": {
                "protocol_id": self.hardware_preflight.protocol_id,
                "synthetic_generation_max_new_tokens": self.hardware_preflight.synthetic_generation_max_new_tokens,
                "synthetic_readout_max_new_tokens": self.hardware_preflight.synthetic_readout_max_new_tokens,
                "activation_input_token_lengths": list(self.hardware_preflight.activation_input_token_lengths),
                "warmup_iterations": self.hardware_preflight.warmup_iterations,
                "measured_iterations": self.hardware_preflight.measured_iterations,
            },
            "exploratory_extension": {
                "status": self.exploratory_extension.status,
                "protocol_requirement": self.exploratory_extension.protocol_requirement,
            },
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "V2Config":
        """Validate every field before checking the exact frozen contract."""

        _reject_unknown_keys(
            raw,
            {
                "schema_version", "study", "dataset", "model", "runtime", "prompt",
                "generation", "readout", "activations", "splits", "features", "probe",
                "analysis", "artifacts", "hardware_preflight", "exploratory_extension",
            },
            "top-level configuration",
        )
        schema_version = _required_int(raw, "schema_version")

        study_raw = _mapping(raw, "study")
        _reject_unknown_keys(
            study_raw,
            {"name", "protocol_id", "phase", "p0_artifact_reuse", "source_identity_policy"},
            "study",
        )
        study = V2StudyConfig(
            name=_required_string(study_raw, "name"),
            protocol_id=_required_string(study_raw, "protocol_id"),
            phase=_required_string(study_raw, "phase"),
            p0_artifact_reuse=_required_string(study_raw, "p0_artifact_reuse"),
            source_identity_policy=_required_string(study_raw, "source_identity_policy"),
        )

        dataset_raw = _mapping(raw, "dataset")
        _reject_unknown_keys(
            {
                key: value
                for key, value in dataset_raw.items()
                if key != "answer_validation_policy"
            },
            {
                "id", "revision", "split", "problem_field", "answer_field",
                "difficulty_field", "topic_field", "stable_id_policy",
                "duplicate_cluster_policy", "selection_method", "selection_seed", "strata",
            },
            "dataset",
        )
        strata: list[V2DifficultyStratum] = []
        for index, item in enumerate(_list(dataset_raw, "strata")):
            if not isinstance(item, Mapping):
                raise V2ConfigurationError(f"dataset.strata[{index}] must be a mapping")
            _reject_unknown_keys(item, {"difficulty", "count"}, f"dataset.strata[{index}]")
            strata.append(
                V2DifficultyStratum(
                    difficulty=_required_float(item, "difficulty"),
                    count=_required_int(item, "count"),
                )
            )
        dataset = V2DatasetConfig(
            dataset_id=_required_string(dataset_raw, "id"),
            revision=_required_string(dataset_raw, "revision"),
            split=_required_string(dataset_raw, "split"),
            problem_field=_required_string(dataset_raw, "problem_field"),
            answer_field=_required_string(dataset_raw, "answer_field"),
            difficulty_field=_required_string(dataset_raw, "difficulty_field"),
            topic_field=_required_string(dataset_raw, "topic_field"),
            stable_id_policy=_required_string(dataset_raw, "stable_id_policy"),
            duplicate_cluster_policy=_required_string(dataset_raw, "duplicate_cluster_policy"),
            selection_method=_required_string(dataset_raw, "selection_method"),
            selection_seed=_required_int(dataset_raw, "selection_seed"),
            strata=tuple(strata),
            answer_validation_policy=(
                _required_string(dataset_raw, "answer_validation_policy")
                if "answer_validation_policy" in dataset_raw
                else "all-source-final-answer-v1"
            ),
        )

        model_raw = _mapping(raw, "model")
        _reject_unknown_keys(
            model_raw,
            {
                "id", "revision", "tokenizer_revision", "dtype", "device",
                "enable_thinking", "expected_model_type", "expected_num_hidden_layers",
                "expected_hidden_size",
            },
            "model",
        )
        model = V2ModelConfig(
            model_id=_required_string(model_raw, "id"),
            revision=_required_string(model_raw, "revision"),
            tokenizer_revision=_required_string(model_raw, "tokenizer_revision"),
            dtype=_required_string(model_raw, "dtype"),
            device=_required_string(model_raw, "device"),
            enable_thinking=_required_bool(model_raw, "enable_thinking"),
            expected_model_type=_required_string(model_raw, "expected_model_type"),
            expected_num_hidden_layers=_required_int(model_raw, "expected_num_hidden_layers"),
            expected_hidden_size=_required_int(model_raw, "expected_hidden_size"),
        )

        runtime_raw = _mapping(raw, "runtime")
        _reject_unknown_keys(
            runtime_raw,
            {
                "require_cuda", "require_bfloat16", "expected_python", "expected_torch",
                "expected_transformers", "expected_datasets", "expected_math_verify",
                "expected_sklearn", "expected_accelerate", "expected_huggingface_hub",
                "expected_safetensors", "minimum_free_disk_bytes",
                "local_max_projected_wall_hours", "minimum_activation_headroom_bytes",
            },
            "runtime",
        )
        runtime = V2RuntimeConfig(
            require_cuda=_required_bool(runtime_raw, "require_cuda"),
            require_bfloat16=_required_bool(runtime_raw, "require_bfloat16"),
            expected_python=_required_string(runtime_raw, "expected_python"),
            expected_torch=_required_string(runtime_raw, "expected_torch"),
            expected_transformers=_required_string(runtime_raw, "expected_transformers"),
            expected_datasets=_required_string(runtime_raw, "expected_datasets"),
            expected_math_verify=_required_string(runtime_raw, "expected_math_verify"),
            expected_sklearn=_required_string(runtime_raw, "expected_sklearn"),
            expected_accelerate=_required_string(runtime_raw, "expected_accelerate"),
            expected_huggingface_hub=_required_string(runtime_raw, "expected_huggingface_hub"),
            expected_safetensors=_required_string(runtime_raw, "expected_safetensors"),
            minimum_free_disk_bytes=_required_int(runtime_raw, "minimum_free_disk_bytes"),
            local_max_projected_wall_hours=_required_float(runtime_raw, "local_max_projected_wall_hours"),
            minimum_activation_headroom_bytes=_required_int(runtime_raw, "minimum_activation_headroom_bytes"),
        )

        prompt_raw = _mapping(raw, "prompt")
        _reject_unknown_keys(prompt_raw, {"template"}, "prompt")
        prompt = V2PromptConfig(template=_required_string(prompt_raw, "template"))
        if prompt.template.count("{problem}") != 1:
            raise V2ConfigurationError("prompt.template must contain exactly one {problem} placeholder")

        generation_raw = _mapping(raw, "generation")
        _reject_unknown_keys(
            generation_raw,
            {
                "reasoning_max_new_tokens", "do_sample", "temperature", "top_p", "top_k",
                "min_p", "rollout_seed_policy", "rollout_seed_root", "rollout_indices",
                "checkpoint_token_positions", "close_think_marker_text", "coordinate_policy",
                "early_no_close_policy", "terminal_cap_policy",
            },
            "generation",
        )
        generation = V2GenerationConfig(
            reasoning_max_new_tokens=_required_int(generation_raw, "reasoning_max_new_tokens"),
            do_sample=_required_bool(generation_raw, "do_sample"),
            temperature=_required_float(generation_raw, "temperature"),
            top_p=_required_float(generation_raw, "top_p"),
            top_k=_required_int(generation_raw, "top_k"),
            min_p=_required_float(generation_raw, "min_p"),
            rollout_seed_policy=_required_string(generation_raw, "rollout_seed_policy"),
            rollout_seed_root=_required_int(generation_raw, "rollout_seed_root"),
            rollout_indices=_int_tuple(generation_raw, "rollout_indices"),
            checkpoint_token_positions=_int_tuple(generation_raw, "checkpoint_token_positions"),
            close_think_marker_text=_required_string(generation_raw, "close_think_marker_text"),
            coordinate_policy=_required_string(generation_raw, "coordinate_policy"),
            early_no_close_policy=_required_string(generation_raw, "early_no_close_policy"),
            terminal_cap_policy=_required_string(generation_raw, "terminal_cap_policy"),
        )

        readout_raw = _mapping(raw, "readout")
        _reject_unknown_keys(
            readout_raw,
            {
                "protocol_id", "cue_text", "max_new_tokens", "do_sample",
                "required_termination_status", "answer_extraction_policy",
                "observable_feature_policy", "previous_readout_policy",
            },
            "readout",
        )
        readout = V2ReadoutConfig(
            protocol_id=_required_string(readout_raw, "protocol_id"),
            cue_text=_required_string(readout_raw, "cue_text"),
            max_new_tokens=_required_int(readout_raw, "max_new_tokens"),
            do_sample=_required_bool(readout_raw, "do_sample"),
            required_termination_status=_required_string(readout_raw, "required_termination_status"),
            answer_extraction_policy=_required_string(readout_raw, "answer_extraction_policy"),
            observable_feature_policy=_required_string(readout_raw, "observable_feature_policy"),
            previous_readout_policy=_required_string(readout_raw, "previous_readout_policy"),
        )

        activations_raw = _mapping(raw, "activations")
        _reject_unknown_keys(
            activations_raw,
            {
                "protocol_id", "layer_indexing", "storage_dtype", "storage_format",
                "include_embedding_state", "include_final_norm_state", "input_policy",
                "use_cache", "output_attentions", "retain_full_sequence_layer_tensors",
            },
            "activations",
        )
        activations = V2ActivationConfig(
            protocol_id=_required_string(activations_raw, "protocol_id"),
            layer_indexing=_required_string(activations_raw, "layer_indexing"),
            storage_dtype=_required_string(activations_raw, "storage_dtype"),
            storage_format=_required_string(activations_raw, "storage_format"),
            include_embedding_state=_required_bool(activations_raw, "include_embedding_state"),
            include_final_norm_state=_required_bool(activations_raw, "include_final_norm_state"),
            input_policy=_required_string(activations_raw, "input_policy"),
            use_cache=_required_bool(activations_raw, "use_cache"),
            output_attentions=_required_bool(activations_raw, "output_attentions"),
            retain_full_sequence_layer_tensors=_required_bool(activations_raw, "retain_full_sequence_layer_tensors"),
        )

        splits_raw = _mapping(raw, "splits")
        _reject_unknown_keys(
            splits_raw,
            {
                "policy", "split_seed", "train_per_difficulty", "validation_per_difficulty",
                "test_per_difficulty", "duplicate_clusters_must_not_cross_splits",
                "test_unblinding_policy",
            },
            "splits",
        )
        splits = V2SplitsConfig(
            policy=_required_string(splits_raw, "policy"),
            split_seed=_required_int(splits_raw, "split_seed"),
            train_per_difficulty=_required_int(splits_raw, "train_per_difficulty"),
            validation_per_difficulty=_required_int(splits_raw, "validation_per_difficulty"),
            test_per_difficulty=_required_int(splits_raw, "test_per_difficulty"),
            duplicate_clusters_must_not_cross_splits=_required_bool(splits_raw, "duplicate_clusters_must_not_cross_splits"),
            test_unblinding_policy=_required_string(splits_raw, "test_unblinding_policy"),
        )

        features_raw = _mapping(raw, "features")
        _reject_unknown_keys(
            features_raw,
            {
                "structural_feature_policy", "observable_feature_policy",
                "normalized_checkpoint_denominator", "topic_encoding",
                "numeric_missing_policy", "previous_agreement_missing_encoding",
                "forbid_future_terminal_gold_evaluator_features",
            },
            "features",
        )
        features = V2FeaturesConfig(
            structural_feature_policy=_required_string(features_raw, "structural_feature_policy"),
            observable_feature_policy=_required_string(features_raw, "observable_feature_policy"),
            normalized_checkpoint_denominator=_required_int(features_raw, "normalized_checkpoint_denominator"),
            topic_encoding=_required_string(features_raw, "topic_encoding"),
            numeric_missing_policy=_required_string(features_raw, "numeric_missing_policy"),
            previous_agreement_missing_encoding=_required_string(features_raw, "previous_agreement_missing_encoding"),
            forbid_future_terminal_gold_evaluator_features=_required_bool(features_raw, "forbid_future_terminal_gold_evaluator_features"),
        )

        probe_raw = _mapping(raw, "probe")
        _reject_unknown_keys(
            probe_raw,
            {
                "family", "solver", "max_iter", "class_weight", "regularization_c_grid",
                "hyperparameter_selection_metric", "hyperparameter_tiebreak",
                "primary_layer_selection_metric", "training_problem_weighting",
                "activation_comparison_cohort",
            },
            "probe",
        )
        if probe_raw.get("class_weight") is not None:
            raise V2ConfigurationError("probe.class_weight must be null")
        probe = V2ProbeConfig(
            family=_required_string(probe_raw, "family"),
            solver=_required_string(probe_raw, "solver"),
            max_iter=_required_int(probe_raw, "max_iter"),
            class_weight=None,
            regularization_c_grid=_float_tuple(probe_raw, "regularization_c_grid"),
            hyperparameter_selection_metric=_required_string(probe_raw, "hyperparameter_selection_metric"),
            hyperparameter_tiebreak=_required_string(probe_raw, "hyperparameter_tiebreak"),
            primary_layer_selection_metric=_required_string(probe_raw, "primary_layer_selection_metric"),
            training_problem_weighting=_required_string(probe_raw, "training_problem_weighting"),
            activation_comparison_cohort=_required_string(probe_raw, "activation_comparison_cohort"),
        )

        analysis_raw = _mapping(raw, "analysis")
        _reject_unknown_keys(
            analysis_raw,
            {
                "positive_label", "negative_label", "primary_metrics", "secondary_metrics",
                "bootstrap_policy", "bootstrap_resamples", "bootstrap_seed",
                "bootstrap_min_valid_fraction", "calibration_bins",
                "report_unweighted_row_sensitivity", "min_train_rows_per_class",
                "min_train_problems_per_class", "min_validation_rows_per_class",
                "min_validation_problems_per_class", "min_test_rows_per_class",
                "min_test_problems_per_class", "min_matched_problem_checkpoint_groups",
                "min_matched_unique_problems", "primary_result_estimand",
            },
            "analysis",
        )
        analysis = V2AnalysisConfig(
            positive_label=_required_string(analysis_raw, "positive_label"),
            negative_label=_required_string(analysis_raw, "negative_label"),
            primary_metrics=_string_tuple(analysis_raw, "primary_metrics"),
            secondary_metrics=_string_tuple(analysis_raw, "secondary_metrics"),
            bootstrap_policy=_required_string(analysis_raw, "bootstrap_policy"),
            bootstrap_resamples=_required_int(analysis_raw, "bootstrap_resamples"),
            bootstrap_seed=_required_int(analysis_raw, "bootstrap_seed"),
            bootstrap_min_valid_fraction=_required_float(analysis_raw, "bootstrap_min_valid_fraction"),
            calibration_bins=_required_int(analysis_raw, "calibration_bins"),
            report_unweighted_row_sensitivity=_required_bool(analysis_raw, "report_unweighted_row_sensitivity"),
            min_train_rows_per_class=_required_int(analysis_raw, "min_train_rows_per_class"),
            min_train_problems_per_class=_required_int(analysis_raw, "min_train_problems_per_class"),
            min_validation_rows_per_class=_required_int(analysis_raw, "min_validation_rows_per_class"),
            min_validation_problems_per_class=_required_int(analysis_raw, "min_validation_problems_per_class"),
            min_test_rows_per_class=_required_int(analysis_raw, "min_test_rows_per_class"),
            min_test_problems_per_class=_required_int(analysis_raw, "min_test_problems_per_class"),
            min_matched_problem_checkpoint_groups=_required_int(analysis_raw, "min_matched_problem_checkpoint_groups"),
            min_matched_unique_problems=_required_int(analysis_raw, "min_matched_unique_problems"),
            primary_result_estimand=_required_string(analysis_raw, "primary_result_estimand"),
        )

        artifacts_raw = _mapping(raw, "artifacts")
        _reject_unknown_keys(
            artifacts_raw, {"root", "public_summary_path", "raw_artifacts_publication"}, "artifacts"
        )
        artifacts = V2ArtifactsConfig(
            root=_required_string(artifacts_raw, "root"),
            public_summary_path=_required_string(artifacts_raw, "public_summary_path"),
            raw_artifacts_publication=_required_string(artifacts_raw, "raw_artifacts_publication"),
        )

        hardware_raw = _mapping(raw, "hardware_preflight")
        _reject_unknown_keys(
            hardware_raw,
            {
                "protocol_id", "synthetic_generation_max_new_tokens",
                "synthetic_readout_max_new_tokens", "activation_input_token_lengths",
                "warmup_iterations", "measured_iterations",
            },
            "hardware_preflight",
        )
        hardware_preflight = V2HardwarePreflightConfig(
            protocol_id=_required_string(hardware_raw, "protocol_id"),
            synthetic_generation_max_new_tokens=_required_int(hardware_raw, "synthetic_generation_max_new_tokens"),
            synthetic_readout_max_new_tokens=_required_int(hardware_raw, "synthetic_readout_max_new_tokens"),
            activation_input_token_lengths=_int_tuple(hardware_raw, "activation_input_token_lengths"),
            warmup_iterations=_required_int(hardware_raw, "warmup_iterations"),
            measured_iterations=_required_int(hardware_raw, "measured_iterations"),
        )

        exploratory_raw = _mapping(raw, "exploratory_extension")
        _reject_unknown_keys(exploratory_raw, {"status", "protocol_requirement"}, "exploratory_extension")
        exploratory_extension = V2ExploratoryExtensionConfig(
            status=_required_string(exploratory_raw, "status"),
            protocol_requirement=_required_string(exploratory_raw, "protocol_requirement"),
        )

        config = cls(
            schema_version=schema_version,
            study=study,
            dataset=dataset,
            model=model,
            runtime=runtime,
            prompt=prompt,
            generation=generation,
            readout=readout,
            activations=activations,
            splits=splits,
            features=features,
            probe=probe,
            analysis=analysis,
            artifacts=artifacts,
            hardware_preflight=hardware_preflight,
            exploratory_extension=exploratory_extension,
        )
        _validate_frozen_v2_contract(config)
        return config


def load_v2_config(path: Path) -> V2Config:
    """Load one V2 YAML file without importing a model or dataset runtime."""

    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyYAML is required to load V2 configuration files; install .[runtime]"
        ) from error

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise V2ConfigurationError(f"cannot read configuration {path}: {error}") from error
    except yaml.YAMLError as error:
        raise V2ConfigurationError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise V2ConfigurationError("top-level configuration must be a mapping")
    return V2Config.from_mapping(raw)


def _validate_frozen_v2_contract(config: V2Config) -> None:
    """Make policy-bearing fields immutable, not merely hashed.

    A changed protocol belongs in a new explicitly authorized configuration,
    rather than being accepted as a loose variant of the frozen study.
    """

    raw = config.to_dict()
    expected = (
        _FROZEN_V2A1_CONFIG
        if config.study.protocol_id == "v2-recovery-activation-probe-20261002-a1"
        else _FROZEN_V2_CONFIG
    )
    if raw != expected:
        differences = _frozen_differences(expected, raw)
        preview = ", ".join(differences[:6])
        suffix = "" if len(differences) <= 6 else ", ..."
        raise V2ConfigurationError(
            f"configuration differs from frozen V2 contract at {preview}{suffix}"
        )


def _frozen_differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Return concise paths for an immutable-contract error message."""

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected or key not in actual:
                differences.append(child)
            else:
                differences.extend(_frozen_differences(expected[key], actual[key], child))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [path]
        differences = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            differences.extend(
                _frozen_differences(expected_item, actual_item, f"{path}[{index}]")
            )
        return differences
    return [] if expected == actual else [path]


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise V2ConfigurationError(f"{key} must be a mapping")
    return value


def _list(raw: Mapping[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise V2ConfigurationError(f"{key} must be a list")
    return value


def _reject_unknown_keys(raw: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise V2ConfigurationError(
            f"{label} contains unsupported field(s): {', '.join(unexpected)}"
        )
    missing = sorted(allowed - set(raw))
    if missing:
        raise V2ConfigurationError(
            f"{label} is missing required field(s): {', '.join(missing)}"
        )


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise V2ConfigurationError(f"{key} must be a non-empty string")
    return value


def _required_bool(raw: Mapping[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise V2ConfigurationError(f"{key} must be boolean")
    return value


def _required_int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise V2ConfigurationError(f"{key} must be an integer")
    return value


def _required_float(raw: Mapping[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2ConfigurationError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise V2ConfigurationError(f"{key} must be finite")
    return result


def _int_tuple(raw: Mapping[str, Any], key: str) -> tuple[int, ...]:
    items = _list(raw, key)
    values: list[int] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int):
            raise V2ConfigurationError(f"{key}[{index}] must be an integer")
        values.append(item)
    return tuple(values)


def _float_tuple(raw: Mapping[str, Any], key: str) -> tuple[float, ...]:
    items = _list(raw, key)
    values: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise V2ConfigurationError(f"{key}[{index}] must be numeric")
        value = float(item)
        if not math.isfinite(value):
            raise V2ConfigurationError(f"{key}[{index}] must be finite")
        values.append(value)
    return tuple(values)


def _string_tuple(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    items = _list(raw, key)
    values: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise V2ConfigurationError(f"{key}[{index}] must be a non-empty string")
        values.append(item)
    return tuple(values)


_FROZEN_V2_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "study": {
        "name": "reasoning-recovery-probes",
        "protocol_id": "v2-recovery-activation-probe-20260929",
        "phase": "primary",
        "p0_artifact_reuse": "forbidden",
        "source_identity_policy": "clean-git-commit-and-tree-v1",
    },
    "dataset": {
        "id": "zwhe99/DeepMath-103K",
        "revision": "5cf055d1fe3d7a2eb19719ac020211469736ae44",
        "split": "train",
        "problem_field": "question",
        "answer_field": "final_answer",
        "difficulty_field": "difficulty",
        "topic_field": "topic",
        "stable_id_policy": "source-index-question-sha256-v1",
        "duplicate_cluster_policy": "normalized-question-sha256-v1",
        "selection_method": "sha256-stratified-difficulty-problem-id-source-index-v1",
        "selection_seed": 20260929,
        "strata": [
            {"difficulty": 7.0, "count": 80},
            {"difficulty": 8.0, "count": 80},
        ],
    },
    "model": {
        "id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "tokenizer_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "dtype": "bfloat16",
        "device": "cuda:0",
        "enable_thinking": True,
        "expected_model_type": "qwen3",
        "expected_num_hidden_layers": 28,
        "expected_hidden_size": 2048,
    },
    "runtime": {
        "require_cuda": True,
        "require_bfloat16": True,
        "expected_python": "3.12.3",
        "expected_torch": "2.11.0+cu128",
        "expected_transformers": "4.52.4",
        "expected_datasets": "4.3.0",
        "expected_math_verify": "0.9.0",
        "expected_sklearn": "1.5.1",
        "expected_accelerate": "1.12.0",
        "expected_huggingface_hub": "0.36.2",
        "expected_safetensors": "0.4.3",
        "minimum_free_disk_bytes": 21474836480,
        "local_max_projected_wall_hours": 48.0,
        "minimum_activation_headroom_bytes": 1073741824,
    },
    "prompt": {
        "template": (
            "Solve the following mathematics problem. Work through the reasoning carefully.\n"
            "After the thinking section, give only a concise final answer in exactly one\n"
            "\\boxed{...} expression, with no explanation after the box.\n\n"
            "Problem:\n"
            "{problem}"
        )
    },
    "generation": {
        "reasoning_max_new_tokens": 4096,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "rollout_seed_policy": "sha256-root-problem-id-rollout-index-v1",
        "rollout_seed_root": 20260929,
        "rollout_indices": [0, 1, 2, 3, 4, 5],
        "checkpoint_token_positions": [128, 256, 512, 1024, 2048],
        "close_think_marker_text": "</think>",
        "coordinate_policy": "raw-generated-pre-close-marker-v1",
        "early_no_close_policy": "non-evaluable-reasoning-boundary-v1",
        "terminal_cap_policy": "cap-prefix-is-terminal-v1",
    },
    "readout": {
        "protocol_id": "symmetric-close-think-greedy-readout-v1",
        "cue_text": "</think>\n\n",
        "max_new_tokens": 64,
        "do_sample": False,
        "required_termination_status": "EOS",
        "answer_extraction_policy": "last-balanced-boxed-only-v1",
        "observable_feature_policy": "candidate-token-logprob-and-first-step-v1",
        "previous_readout_policy": "immediate-prior-available-checkpoint-v1",
    },
    "activations": {
        "protocol_id": "post-block-last-reasoning-token-v1",
        "layer_indexing": "zero-based-transformer-block-output-before-final-norm",
        "storage_dtype": "bfloat16",
        "storage_format": "torch-save-bfloat16-matrix-v1",
        "include_embedding_state": False,
        "include_final_norm_state": False,
        "input_policy": "prompt-plus-exact-prefix-no-readout-cue-v1",
        "use_cache": False,
        "output_attentions": False,
        "retain_full_sequence_layer_tensors": False,
    },
    "splits": {
        "policy": "sha256-stratified-problem-cluster-v1",
        "split_seed": 20260930,
        "train_per_difficulty": 48,
        "validation_per_difficulty": 16,
        "test_per_difficulty": 16,
        "duplicate_clusters_must_not_cross_splits": True,
        "test_unblinding_policy": "validation-selection-record-required-v1",
    },
    "features": {
        "structural_feature_policy": "structural-problem-only-v1",
        "observable_feature_policy": "current-readout-only-v1",
        "normalized_checkpoint_denominator": 4096,
        "topic_encoding": "one-hot-handle-unknown-ignore-v1",
        "numeric_missing_policy": "train-median-plus-indicator-v1",
        "previous_agreement_missing_encoding": "value-zero-plus-availability-indicator-v1",
        "forbid_future_terminal_gold_evaluator_features": True,
    },
    "probe": {
        "family": "l2-logistic-regression-v1",
        "solver": "lbfgs",
        "max_iter": 10000,
        "class_weight": None,
        "regularization_c_grid": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "hyperparameter_selection_metric": "validation-log-loss",
        "hyperparameter_tiebreak": "smaller-c-then-lower-layer-index",
        "primary_layer_selection_metric": "validation-log-loss",
        "training_problem_weighting": "equal-total-weight-per-problem-v1",
        "activation_comparison_cohort": "activation-available-primary-label-rows-v1",
    },
    "analysis": {
        "positive_label": "W_TO_C",
        "negative_label": "W_TO_W",
        "primary_metrics": ["pr_auc", "log_loss", "brier"],
        "secondary_metrics": ["auroc", "calibration_deciles"],
        "bootstrap_policy": "paired-problem-cluster-percentile-v1",
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261001,
        "bootstrap_min_valid_fraction": 0.95,
        "calibration_bins": 10,
        "report_unweighted_row_sensitivity": True,
        "min_train_rows_per_class": 50,
        "min_train_problems_per_class": 15,
        "min_validation_rows_per_class": 20,
        "min_validation_problems_per_class": 8,
        "min_test_rows_per_class": 20,
        "min_test_problems_per_class": 8,
        "min_matched_problem_checkpoint_groups": 15,
        "min_matched_unique_problems": 8,
        "primary_result_estimand": "paired-test-log-loss-reduction-b-minus-c-at-validation-selected-layer",
    },
    "artifacts": {
        "root": "artifacts/v2",
        "public_summary_path": "reports/v2_summary.json",
        "raw_artifacts_publication": "forbidden",
    },
    "hardware_preflight": {
        "protocol_id": "v2-synthetic-runtime-qualification-v1",
        "synthetic_generation_max_new_tokens": 128,
        "synthetic_readout_max_new_tokens": 64,
        "activation_input_token_lengths": [512, 1024, 2048, 4096],
        "warmup_iterations": 1,
        "measured_iterations": 3,
    },
    "exploratory_extension": {
        "status": "deferred-until-primary-terminal-decision",
        "protocol_requirement": "separate-v2-e1-protocol-and-nonoverlapping-manifest",
    },
}

# A1 is a separately approved exact profile, not a configurable science override.
# Derivation keeps the original contract untouched and makes its narrow delta
# explicit: validation scope, protocol identity, and two output namespaces only.
_FROZEN_V2A1_CONFIG = deepcopy(_FROZEN_V2_CONFIG)
_FROZEN_V2A1_CONFIG["study"]["protocol_id"] = "v2-recovery-activation-probe-20261002-a1"
_FROZEN_V2A1_CONFIG["dataset"]["answer_validation_policy"] = "selected-cohort-final-answer-v1"
_FROZEN_V2A1_CONFIG["artifacts"]["root"] = "artifacts/v2a1"
_FROZEN_V2A1_CONFIG["artifacts"]["public_summary_path"] = "reports/v2a1_summary.json"
