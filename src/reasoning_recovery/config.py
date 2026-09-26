"""Validated P0 configuration.

Optional runtime dependencies are imported only when a YAML file is loaded, so
offline unit tests can exercise the data contract without model packages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .provenance import sha256_json


class ConfigurationError(ValueError):
    """Raised when a P0 configuration cannot be interpreted safely."""


@dataclass(frozen=True)
class StudyConfig:
    name: str
    protocol_id: str
    feasibility_only: bool


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    revision: str | None
    split: str
    problem_field: str
    answer_field: str
    id_field: str
    selection_method: str
    selection_seed: int
    pilot_size: int


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str | None
    tokenizer_revision: str | None
    dtype: str
    device: str
    enable_thinking: bool


@dataclass(frozen=True)
class PromptConfig:
    template: str

    def render(self, problem: str) -> str:
        """Substitute only the documented placeholder.

        String-formatting would treat LaTex braces in the prompt as fields, so
        yahan exact placeholder replacement hi safer hai.
        """

        return self.template.replace("{problem}", problem)


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    rollout_seeds: tuple[int, ...]
    checkpoint_token_positions: tuple[int, ...]


@dataclass(frozen=True)
class ForcedAnswerConfig:
    protocol_id: str
    close_think_text: str
    max_new_tokens: int
    do_sample: bool


@dataclass(frozen=True)
class AnswerExtractionConfig:
    policy: str


@dataclass(frozen=True)
class ArtifactConfig:
    root: str


@dataclass(frozen=True)
class PilotConfig:
    schema_version: int
    study: StudyConfig
    dataset: DatasetConfig
    model: ModelConfig
    prompt: PromptConfig
    generation: GenerationConfig
    forced_answer: ForcedAnswerConfig
    answer_extraction: AnswerExtractionConfig
    artifacts: ArtifactConfig

    @property
    def config_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PilotConfig":
        schema_version = _required_int(raw, "schema_version")
        if schema_version != 1:
            raise ConfigurationError(f"unsupported schema_version: {schema_version}")

        study_raw = _required_mapping(raw, "study")
        study = StudyConfig(
            name=_required_string(study_raw, "name"),
            protocol_id=_required_string(study_raw, "protocol_id"),
            feasibility_only=_required_bool(study_raw, "feasibility_only"),
        )
        if not study.feasibility_only:
            raise ConfigurationError("P0 configuration must set feasibility_only=true")

        dataset_raw = _required_mapping(raw, "dataset")
        dataset = DatasetConfig(
            dataset_id=_required_string(dataset_raw, "id"),
            revision=_optional_string(dataset_raw.get("revision"), "dataset.revision"),
            split=_required_string(dataset_raw, "split"),
            problem_field=_required_string(dataset_raw, "problem_field"),
            answer_field=_required_string(dataset_raw, "answer_field"),
            id_field=_required_string(dataset_raw, "id_field"),
            selection_method=_required_string(dataset_raw, "selection_method"),
            selection_seed=_required_int(dataset_raw, "selection_seed"),
            pilot_size=_required_int(dataset_raw, "pilot_size"),
        )
        if dataset.selection_method != "sha256-problem-id-source-index-v1":
            raise ConfigurationError(
                "only sha256-problem-id-source-index-v1 is implemented for P0 selection"
            )
        if dataset.pilot_size <= 0:
            raise ConfigurationError("dataset.pilot_size must be positive")

        model_raw = _required_mapping(raw, "model")
        model = ModelConfig(
            model_id=_required_string(model_raw, "id"),
            revision=_optional_string(model_raw.get("revision"), "model.revision"),
            tokenizer_revision=_optional_string(
                model_raw.get("tokenizer_revision"), "model.tokenizer_revision"
            ),
            dtype=_required_string(model_raw, "dtype"),
            device=_required_string(model_raw, "device"),
            enable_thinking=_required_bool(model_raw, "enable_thinking"),
        )
        if model.dtype not in {"bfloat16", "float16"}:
            raise ConfigurationError("model.dtype must be bfloat16 or float16")
        if not model.enable_thinking:
            raise ConfigurationError("P0 requires thinking mode to be enabled")

        prompt_raw = _required_mapping(raw, "prompt")
        prompt = PromptConfig(template=_required_string(prompt_raw, "template"))
        if prompt.template.count("{problem}") != 1:
            raise ConfigurationError("prompt.template must contain exactly one {problem} placeholder")

        generation_raw = _required_mapping(raw, "generation")
        generation = GenerationConfig(
            max_new_tokens=_required_int(generation_raw, "max_new_tokens"),
            do_sample=_required_bool(generation_raw, "do_sample"),
            temperature=_required_float(generation_raw, "temperature"),
            top_p=_required_float(generation_raw, "top_p"),
            top_k=_required_int(generation_raw, "top_k"),
            min_p=_required_float(generation_raw, "min_p"),
            rollout_seeds=_int_tuple(generation_raw, "rollout_seeds"),
            checkpoint_token_positions=_int_tuple(
                generation_raw, "checkpoint_token_positions"
            ),
        )
        if generation.max_new_tokens <= 0:
            raise ConfigurationError("generation.max_new_tokens must be positive")
        if not generation.do_sample:
            raise ConfigurationError("base trajectory generation must be stochastic")
        if generation.temperature <= 0:
            raise ConfigurationError("generation.temperature must be positive")
        if not 0 < generation.top_p <= 1:
            raise ConfigurationError("generation.top_p must be in (0, 1]")
        if generation.top_k <= 0 or generation.min_p < 0:
            raise ConfigurationError("generation.top_k/min_p are invalid")
        _strictly_increasing_positive(
            generation.checkpoint_token_positions, "checkpoint_token_positions"
        )
        if len(set(generation.rollout_seeds)) != len(generation.rollout_seeds):
            raise ConfigurationError("generation.rollout_seeds must be unique")
        if not generation.rollout_seeds:
            raise ConfigurationError("generation.rollout_seeds cannot be empty")

        forced_raw = _required_mapping(raw, "forced_answer")
        forced_answer = ForcedAnswerConfig(
            protocol_id=_required_string(forced_raw, "protocol_id"),
            close_think_text=_required_string(forced_raw, "close_think_text"),
            max_new_tokens=_required_int(forced_raw, "max_new_tokens"),
            do_sample=_required_bool(forced_raw, "do_sample"),
        )
        if forced_answer.do_sample:
            raise ConfigurationError("forced-answer decoding must be deterministic")
        if forced_answer.max_new_tokens <= 0:
            raise ConfigurationError("forced_answer.max_new_tokens must be positive")

        extraction_raw = _required_mapping(raw, "answer_extraction")
        answer_extraction = AnswerExtractionConfig(
            policy=_required_string(extraction_raw, "policy")
        )
        if answer_extraction.policy != "last-balanced-boxed-only-v1":
            raise ConfigurationError(
                "only last-balanced-boxed-only-v1 is implemented for P0"
            )

        artifacts_raw = _required_mapping(raw, "artifacts")
        artifacts = ArtifactConfig(root=_required_string(artifacts_raw, "root"))

        return cls(
            schema_version=schema_version,
            study=study,
            dataset=dataset,
            model=model,
            prompt=prompt,
            generation=generation,
            forced_answer=forced_answer,
            answer_extraction=answer_extraction,
            artifacts=artifacts,
        )


def load_config(path: Path) -> PilotConfig:
    """Load and validate a YAML P0 configuration."""

    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyYAML is required to load configuration files; install .[runtime]"
        ) from error

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigurationError(f"cannot read configuration {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {path}: {error}") from error

    if not isinstance(raw, Mapping):
        raise ConfigurationError("top-level configuration must be a mapping")
    return PilotConfig.from_mapping(raw)


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty string")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a non-empty string or null")
    return value


def _required_int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{key} must be an integer")
    return value


def _required_float(raw: Mapping[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{key} must be numeric")
    return float(value)


def _required_bool(raw: Mapping[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{key} must be boolean")
    return value


def _int_tuple(raw: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ConfigurationError(f"{key} must be a list of integers")
    values: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigurationError(f"{key} must contain only integers")
        values.append(item)
    return tuple(values)


def _strictly_increasing_positive(values: tuple[int, ...], label: str) -> None:
    if not values or any(value <= 0 for value in values):
        raise ConfigurationError(f"{label} must contain positive values")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ConfigurationError(f"{label} must be strictly increasing")
