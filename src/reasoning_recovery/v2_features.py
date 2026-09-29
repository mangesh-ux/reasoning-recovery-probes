"""Leakage-guarded V2 feature enrollment and train-only preprocessing.

V2 is intentionally stricter than a generic tabular modelling helper. A row
first exists as an unlabelled checkpoint observation assembled from data that
was available at that checkpoint. The recovery label is joined later, and the
only rows eligible for the primary probe are W_TO_C and W_TO_W. This makes it
harder for a caller to accidentally put a terminal answer, gold answer, or
evaluator result into a fitted feature matrix.

The module is pure and offline: it never loads a model, dataset, evaluator, or
artifact. It accepts compact scalar observations that the V2 runtime records
from the current deterministic readout.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


class V2FeatureError(ValueError):
    """Raised when a V2 feature, label join, or preprocessing contract fails."""


class FeatureSet(str, Enum):
    """The frozen non-activation feature representations for V2."""

    STRUCTURAL = "A"
    OBSERVABLE = "B"


PRIMARY_POSITIVE_LABEL = "W_TO_C"
PRIMARY_NEGATIVE_LABEL = "W_TO_W"
PRIMARY_LABELS = frozenset({PRIMARY_POSITIVE_LABEL, PRIMARY_NEGATIVE_LABEL})
ALL_TRANSITION_LABELS = frozenset({"W_TO_C", "W_TO_W", "C_TO_C", "C_TO_W"})

# These names form the frozen A and B representations. Keep the list explicit
# rather than accepting arbitrary numeric columns from a receipt. In
# particular, terminal fields and evaluator fields never have a route into the
# matrix builder.
STRUCTURAL_NUMERIC_FEATURES = (
    "checkpoint_token_position",
    "normalized_checkpoint_position",
    "problem_token_count",
    "difficulty",
)
OBSERVABLE_NUMERIC_FEATURES = (
    "mean_answer_token_logprob",
    "min_answer_token_logprob",
    "answer_token_count",
    "first_readout_token_entropy_nats",
    "first_readout_token_probability_margin",
    "previous_readout_agreement",
    "previous_readout_available",
)
TOPIC_FEATURE_NAME = "topic"

# A caller may use this audit helper before adapting a receipt schema. The
# static V2 feature lists above are the actual enforcement point; this list is
# intentionally conservative and documents the major future and label routes.
FORBIDDEN_FEATURE_NAME_FRAGMENTS = (
    "terminal",
    "future",
    "gold",
    "reference_answer",
    "evaluator",
    "evaluation",
    "transition_label",
    "recovery_label",
    "hidden_state",
    "activation",
    "natural_final",
)


@dataclass(frozen=True)
class CheckpointReadoutFeatureInput:
    """One current-checkpoint, unlabelled readout observation.

    normalized_extracted_answer is used only to construct the permitted
    immediate-prior agreement feature, then discarded from V2FeatureRow. It is
    not a fitted feature. It must be a normalization of the current short
    readout rather than a gold or evaluator-derived value.
    """

    row_id: str
    problem_id: str
    rollout_id: str
    checkpoint_token: int
    problem_token_count: int
    difficulty: float
    topic: str
    mean_answer_token_logprob: float | None
    min_answer_token_logprob: float | None
    answer_token_count: int | None
    first_readout_token_entropy_nats: float | None
    first_readout_token_probability_margin: float | None
    normalized_extracted_answer: str | None
    readout_available: bool = True

    def __post_init__(self) -> None:
        _require_nonempty_string(self.row_id, "row_id")
        _require_nonempty_string(self.problem_id, "problem_id")
        _require_nonempty_string(self.rollout_id, "rollout_id")
        _require_positive_int(self.checkpoint_token, "checkpoint_token")
        _require_nonnegative_int(self.problem_token_count, "problem_token_count")
        _require_finite_number(self.difficulty, "difficulty")
        _require_nonempty_string(self.topic, "topic")
        if not isinstance(self.readout_available, bool):
            raise V2FeatureError("readout_available must be Boolean")
        _require_optional_finite_number(
            self.mean_answer_token_logprob, "mean_answer_token_logprob"
        )
        _require_optional_finite_number(
            self.min_answer_token_logprob, "min_answer_token_logprob"
        )
        if self.answer_token_count is not None:
            _require_nonnegative_int(self.answer_token_count, "answer_token_count")
        _require_optional_finite_number(
            self.first_readout_token_entropy_nats,
            "first_readout_token_entropy_nats",
        )
        _require_optional_finite_number(
            self.first_readout_token_probability_margin,
            "first_readout_token_probability_margin",
        )
        if self.normalized_extracted_answer is not None:
            _require_nonempty_string(
                self.normalized_extracted_answer,
                "normalized_extracted_answer",
            )
        if not self.readout_available and self.normalized_extracted_answer is not None:
            raise V2FeatureError(
                "an unavailable readout cannot carry a normalized extracted answer"
            )


@dataclass(frozen=True)
class V2FeatureRow:
    """A leakage-safe A/B feature row with no label or terminal information."""

    row_id: str
    problem_id: str
    rollout_id: str
    checkpoint_token: int
    checkpoint_token_position: float
    normalized_checkpoint_position: float
    problem_token_count: float
    difficulty: float
    topic: str
    mean_answer_token_logprob: float | None
    min_answer_token_logprob: float | None
    answer_token_count: float | None
    first_readout_token_entropy_nats: float | None
    first_readout_token_probability_margin: float | None
    previous_readout_agreement: float
    previous_readout_available: float

    def __post_init__(self) -> None:
        _require_nonempty_string(self.row_id, "row_id")
        _require_nonempty_string(self.problem_id, "problem_id")
        _require_nonempty_string(self.rollout_id, "rollout_id")
        _require_positive_int(self.checkpoint_token, "checkpoint_token")
        _require_finite_number(
            self.checkpoint_token_position, "checkpoint_token_position"
        )
        _require_finite_number(
            self.normalized_checkpoint_position, "normalized_checkpoint_position"
        )
        _require_finite_number(self.problem_token_count, "problem_token_count")
        _require_finite_number(self.difficulty, "difficulty")
        _require_nonempty_string(self.topic, "topic")
        for name in (
            "mean_answer_token_logprob",
            "min_answer_token_logprob",
            "answer_token_count",
            "first_readout_token_entropy_nats",
            "first_readout_token_probability_margin",
        ):
            _require_optional_finite_number(getattr(self, name), name)
        if self.previous_readout_agreement not in (0.0, 1.0):
            raise V2FeatureError(
                "previous_readout_agreement must be 0.0 or 1.0 after encoding"
            )
        if self.previous_readout_available not in (0.0, 1.0):
            raise V2FeatureError(
                "previous_readout_available must be 0.0 or 1.0 after encoding"
            )
        if self.previous_readout_available == 0.0 and self.previous_readout_agreement:
            raise V2FeatureError(
                "previous_readout_agreement must be zero when unavailable"
            )

    def numeric_values(self, feature_set: FeatureSet | str) -> dict[str, float | None]:
        """Return only frozen A or B numeric values, never arbitrary row fields."""

        selected = _coerce_feature_set(feature_set)
        values: dict[str, float | None] = {
            "checkpoint_token_position": self.checkpoint_token_position,
            "normalized_checkpoint_position": self.normalized_checkpoint_position,
            "problem_token_count": self.problem_token_count,
            "difficulty": self.difficulty,
        }
        if selected is FeatureSet.OBSERVABLE:
            values.update(
                {
                    "mean_answer_token_logprob": self.mean_answer_token_logprob,
                    "min_answer_token_logprob": self.min_answer_token_logprob,
                    "answer_token_count": self.answer_token_count,
                    "first_readout_token_entropy_nats": (
                        self.first_readout_token_entropy_nats
                    ),
                    "first_readout_token_probability_margin": (
                        self.first_readout_token_probability_margin
                    ),
                    "previous_readout_agreement": self.previous_readout_agreement,
                    "previous_readout_available": self.previous_readout_available,
                }
            )
        return values


@dataclass(frozen=True)
class V2PrimaryRow:
    """A currently-wrong primary-probe row after a separate label join."""

    feature_row: V2FeatureRow
    transition_label: str

    def __post_init__(self) -> None:
        if self.transition_label not in PRIMARY_LABELS:
            raise V2FeatureError(
                "V2 primary rows must have exactly W_TO_C or W_TO_W labels"
            )

    @property
    def row_id(self) -> str:
        return self.feature_row.row_id

    @property
    def problem_id(self) -> str:
        return self.feature_row.problem_id

    @property
    def rollout_id(self) -> str:
        return self.feature_row.rollout_id

    @property
    def checkpoint_token(self) -> int:
        return self.feature_row.checkpoint_token

    @property
    def target(self) -> int:
        """Return the frozen positive-class encoding: W_TO_C == 1."""

        return int(self.transition_label == PRIMARY_POSITIVE_LABEL)


@dataclass(frozen=True)
class PrimaryEnrollment:
    """The result of joining labels after feature construction."""

    rows: tuple[V2PrimaryRow, ...]
    exclusion_counts: Mapping[str, int]

    @property
    def label_counts(self) -> dict[str, int]:
        return dict(
            sorted(Counter(row.transition_label for row in self.rows).items())
        )


@dataclass(frozen=True)
class FittedFeaturePreprocessor:
    """A train-fitted encoder/scaler that can only transform later rows.

    fit_train_preprocessor receives no labels and never touches validation or
    test rows. The object stores all imputation, standardization, topic
    vocabulary, and zero-variance decisions made on its train rows.
    """

    feature_set: FeatureSet
    numeric_feature_names: tuple[str, ...]
    numeric_medians: tuple[float, ...]
    numeric_means: tuple[float, ...]
    numeric_scales: tuple[float, ...]
    topic_categories: tuple[str, ...]
    additional_feature_names: tuple[str, ...]
    kept_column_indices: tuple[int, ...]
    output_feature_names: tuple[str, ...]

    def transform(
        self,
        rows: Sequence[V2FeatureRow | V2PrimaryRow],
        *,
        additional_numeric: Mapping[str, Sequence[float]] | None = None,
    ) -> Any:
        """Transform a fixed row set without refitting preprocessing state."""

        np = _numpy()
        matrix, names = _raw_feature_matrix(
            rows,
            feature_set=self.feature_set,
            topic_categories=self.topic_categories,
            additional_feature_names=self.additional_feature_names,
            additional_numeric=additional_numeric,
        )
        expected_names = _unpruned_feature_names(
            self.numeric_feature_names,
            self.topic_categories,
        )
        if tuple(names) != expected_names:
            raise V2FeatureError(
                "feature schema differs from the train-fitted preprocessing schema"
            )
        numeric_count = len(self.numeric_feature_names)
        transformed = matrix.astype(float, copy=True)
        for index in range(numeric_count):
            raw = transformed[:, index]
            missing = ~np.isfinite(raw)
            raw[missing] = self.numeric_medians[index]
            scale = self.numeric_scales[index]
            if scale > 0:
                raw[:] = (raw - self.numeric_means[index]) / scale
            else:
                raw[:] = 0.0
            transformed[:, index] = raw
        return transformed[:, self.kept_column_indices]


def build_v2_feature_rows(
    observations: Iterable[CheckpointReadoutFeatureInput],
    *,
    normalized_checkpoint_denominator: int = 4096,
) -> tuple[V2FeatureRow, ...]:
    """Build A/B rows and compute only the permitted immediate-prior feature.

    The denominator is deliberately the known frozen reasoning budget, never a
    terminal reasoning length. Observations should contain checkpoint readouts
    only; terminal readout records have no place in this constructor.
    """

    _require_positive_int(
        normalized_checkpoint_denominator, "normalized_checkpoint_denominator"
    )
    supplied = tuple(observations)
    row_ids = [item.row_id for item in supplied]
    if len(set(row_ids)) != len(row_ids):
        raise V2FeatureError("checkpoint feature inputs must have unique row_id values")

    by_rollout: dict[str, list[CheckpointReadoutFeatureInput]] = defaultdict(list)
    rollout_problem: dict[str, str] = {}
    for item in supplied:
        if not isinstance(item, CheckpointReadoutFeatureInput):
            raise V2FeatureError(
                "build_v2_feature_rows accepts CheckpointReadoutFeatureInput objects"
            )
        existing_problem = rollout_problem.setdefault(item.rollout_id, item.problem_id)
        if existing_problem != item.problem_id:
            raise V2FeatureError(
                "one rollout_id cannot belong to more than one problem_id"
            )
        by_rollout[item.rollout_id].append(item)

    output: list[V2FeatureRow] = []
    for rollout_id in sorted(by_rollout):
        ordered = sorted(
            by_rollout[rollout_id],
            key=lambda value: (value.checkpoint_token, value.row_id),
        )
        positions = [item.checkpoint_token for item in ordered]
        if len(set(positions)) != len(positions):
            raise V2FeatureError(
                f"rollout {rollout_id!r} has duplicate checkpoint token coordinates"
            )
        previous_available: CheckpointReadoutFeatureInput | None = None
        for item in ordered:
            comparison_available = (
                previous_available is not None
                and previous_available.normalized_extracted_answer is not None
                and item.normalized_extracted_answer is not None
            )
            previous_agreement = (
                float(
                    previous_available.normalized_extracted_answer
                    == item.normalized_extracted_answer
                )
                if comparison_available and previous_available is not None
                else 0.0
            )
            output.append(
                V2FeatureRow(
                    row_id=item.row_id,
                    problem_id=item.problem_id,
                    rollout_id=item.rollout_id,
                    checkpoint_token=item.checkpoint_token,
                    checkpoint_token_position=float(item.checkpoint_token),
                    normalized_checkpoint_position=(
                        float(item.checkpoint_token)
                        / float(normalized_checkpoint_denominator)
                    ),
                    problem_token_count=float(item.problem_token_count),
                    difficulty=float(item.difficulty),
                    topic=item.topic,
                    mean_answer_token_logprob=item.mean_answer_token_logprob,
                    min_answer_token_logprob=item.min_answer_token_logprob,
                    answer_token_count=(
                        float(item.answer_token_count)
                        if item.answer_token_count is not None
                        else None
                    ),
                    first_readout_token_entropy_nats=(
                        item.first_readout_token_entropy_nats
                    ),
                    first_readout_token_probability_margin=(
                        item.first_readout_token_probability_margin
                    ),
                    previous_readout_agreement=previous_agreement,
                    previous_readout_available=float(comparison_available),
                )
            )
            # Immediate previous available checkpoint readout skips an
            # unavailable checkpoint, but it never looks ahead.
            if item.readout_available:
                previous_available = item
    return tuple(sorted(output, key=lambda value: value.row_id))


def enroll_v2_primary_rows(
    feature_rows: Iterable[V2FeatureRow],
    labels_by_row_id: Mapping[str, str | None],
    *,
    reject_unknown_label_rows: bool = True,
) -> PrimaryEnrollment:
    """Join labels after feature construction and retain only W_TO_C/W_TO_W.

    labels_by_row_id is intentionally a separate input. A caller cannot use a
    transition label as an ordinary numeric feature through this API.
    """

    rows = tuple(feature_rows)
    if any(not isinstance(row, V2FeatureRow) for row in rows):
        raise V2FeatureError("feature_rows must contain V2FeatureRow values")
    row_ids = [row.row_id for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise V2FeatureError("feature row IDs must be unique before label joining")
    unknown = set(labels_by_row_id) - set(row_ids)
    if reject_unknown_label_rows and unknown:
        raise V2FeatureError(
            "labels refer to unknown feature rows: " + ", ".join(sorted(unknown))
        )

    primary_rows: list[V2PrimaryRow] = []
    excluded = Counter[str]()
    for row in rows:
        label = labels_by_row_id.get(row.row_id)
        if label is None:
            excluded["MISSING_TRANSITION_LABEL"] += 1
            continue
        if not isinstance(label, str) or label not in ALL_TRANSITION_LABELS:
            excluded["INVALID_TRANSITION_LABEL"] += 1
            continue
        if label not in PRIMARY_LABELS:
            excluded[f"NON_PRIMARY_{label}"] += 1
            continue
        primary_rows.append(V2PrimaryRow(feature_row=row, transition_label=label))
    return PrimaryEnrollment(
        rows=tuple(sorted(primary_rows, key=lambda value: value.row_id)),
        exclusion_counts=dict(sorted(excluded.items())),
    )


def equal_total_problem_weights(
    rows: Sequence[V2FeatureRow | V2PrimaryRow],
) -> tuple[float, ...]:
    """Return row weights with equal total contribution per problem.

    The common per-problem total is normalized so the average row weight is
    one. Multiplying all sample weights by a common constant would preserve the
    equal-problem estimand, but this normalization keeps L2-C comparisons
    numerically interpretable across splits.
    """

    if not rows:
        raise V2FeatureError("cannot compute problem weights for an empty row set")
    problem_ids = [_problem_id(row) for row in rows]
    counts = Counter(problem_ids)
    if any(not problem_id for problem_id in problem_ids):
        raise V2FeatureError("every row must have a non-empty problem_id")
    target_total = len(rows) / len(counts)
    weights = tuple(target_total / counts[problem_id] for problem_id in problem_ids)
    totals: dict[str, float] = defaultdict(float)
    for problem_id, weight in zip(problem_ids, weights):
        totals[problem_id] += weight
    if any(abs(total - target_total) > 1e-12 for total in totals.values()):
        raise AssertionError("equal-total-problem weight construction failed")
    return weights


def assert_no_future_feature_names(feature_names: Iterable[str]) -> None:
    """Fail closed when an adapted schema proposes a forbidden feature route."""

    for name in feature_names:
        if not isinstance(name, str) or not name.strip():
            raise V2FeatureError("feature names must be non-empty strings")
        lowered = name.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_FEATURE_NAME_FRAGMENTS):
            raise V2FeatureError(
                f"forbidden future/label/activation feature name in A/B schema: {name}"
            )


def fit_train_preprocessor(
    train_rows: Sequence[V2FeatureRow | V2PrimaryRow],
    *,
    feature_set: FeatureSet | str,
    additional_numeric: Mapping[str, Sequence[float]] | None = None,
    additional_feature_prefix: str = "activation_layer",
) -> FittedFeaturePreprocessor:
    """Fit imputation, scaling, topic encoding, and variance filtering on train.

    additional_numeric is for one C-layer activation vector. It is accepted
    only after the caller has restricted rows to activation-available primary
    rows. Those values are standardized using train rows only.
    """

    np = _numpy()
    selected = _coerce_feature_set(feature_set)
    rows = tuple(train_rows)
    _validate_rows(rows)
    if not isinstance(additional_feature_prefix, str) or not additional_feature_prefix:
        raise V2FeatureError("additional_feature_prefix must be a non-empty string")

    additional_names = _additional_feature_names(
        rows,
        additional_numeric,
        prefix=additional_feature_prefix,
    )
    topic_categories = tuple(sorted({_feature_row(row).topic for row in rows}))
    matrix, _ = _raw_feature_matrix(
        rows,
        feature_set=selected,
        topic_categories=topic_categories,
        additional_feature_names=additional_names,
        additional_numeric=additional_numeric,
    )
    numeric_names = tuple(_numeric_feature_names(selected)) + additional_names
    numeric_count = len(numeric_names)
    medians: list[float] = []
    means: list[float] = []
    scales: list[float] = []
    transformed = matrix.astype(float, copy=True)
    for index, name in enumerate(numeric_names):
        values = transformed[:, index]
        observed = values[np.isfinite(values)]
        if observed.size == 0:
            raise V2FeatureError(
                f"train-only median is undefined because {name!r} is missing in every train row"
            )
        value_median = float(median(float(value) for value in observed))
        medians.append(value_median)
        values[~np.isfinite(values)] = value_median
        value_mean = float(np.mean(values))
        value_scale = float(np.std(values))
        means.append(value_mean)
        scales.append(value_scale)
        transformed[:, index] = (
            (values - value_mean) / value_scale
            if value_scale > 0
            else np.zeros_like(values, dtype=float)
        )

    # Numeric missingness indicators follow all numeric values. They are
    # already part of matrix and deliberately remain 0/1 rather than being
    # standardized. Topics are one-hot encoded from the train vocabulary.
    variances = np.var(transformed, axis=0)
    kept = tuple(
        int(index)
        for index, value in enumerate(variances)
        if isfinite(float(value)) and float(value) > 0.0
    )
    if not kept:
        raise V2FeatureError("all train preprocessing columns are zero-variance")
    unpruned_names = _unpruned_feature_names(numeric_names, topic_categories)
    if len(unpruned_names) != transformed.shape[1]:
        raise AssertionError("feature-name and matrix widths disagree")
    return FittedFeaturePreprocessor(
        feature_set=selected,
        numeric_feature_names=numeric_names,
        numeric_medians=tuple(medians),
        numeric_means=tuple(means),
        numeric_scales=tuple(scales),
        topic_categories=topic_categories,
        additional_feature_names=additional_names,
        kept_column_indices=kept,
        output_feature_names=tuple(unpruned_names[index] for index in kept),
    )


def feature_matrix_fit_transform(
    train_rows: Sequence[V2FeatureRow | V2PrimaryRow],
    *,
    feature_set: FeatureSet | str,
    additional_numeric: Mapping[str, Sequence[float]] | None = None,
    additional_feature_prefix: str = "activation_layer",
) -> tuple[FittedFeaturePreprocessor, Any]:
    """Convenience wrapper used by offline probe fitting."""

    preprocessor = fit_train_preprocessor(
        train_rows,
        feature_set=feature_set,
        additional_numeric=additional_numeric,
        additional_feature_prefix=additional_feature_prefix,
    )
    return preprocessor, preprocessor.transform(
        train_rows, additional_numeric=additional_numeric
    )


def _raw_feature_matrix(
    rows: Sequence[V2FeatureRow | V2PrimaryRow],
    *,
    feature_set: FeatureSet,
    topic_categories: tuple[str, ...],
    additional_feature_names: tuple[str, ...],
    additional_numeric: Mapping[str, Sequence[float]] | None,
) -> tuple[Any, tuple[str, ...]]:
    np = _numpy()
    _validate_rows(rows)
    numeric_names = tuple(_numeric_feature_names(feature_set)) + additional_feature_names
    names = _unpruned_feature_names(numeric_names, topic_categories)
    values: list[list[float]] = []
    for row in rows:
        feature_row = _feature_row(row)
        numeric = feature_row.numeric_values(feature_set)
        vector: list[float] = [
            _number_or_nan(numeric[name]) for name in _numeric_feature_names(feature_set)
        ]
        if additional_feature_names:
            if additional_numeric is None:
                raise V2FeatureError("activation/additional vectors are required")
            vector.extend(
                _validated_additional_vector(
                    additional_numeric.get(feature_row.row_id),
                    expected_width=len(additional_feature_names),
                    row_id=feature_row.row_id,
                )
            )
        missing_indicators = [float(not isfinite(value)) for value in vector]
        topic_one_hot = [
            float(feature_row.topic == category) for category in topic_categories
        ]
        values.append(vector + missing_indicators + topic_one_hot)
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise AssertionError("raw feature matrix has an unexpected shape")
    return matrix, names


def _additional_feature_names(
    rows: Sequence[V2FeatureRow | V2PrimaryRow],
    additional_numeric: Mapping[str, Sequence[float]] | None,
    *,
    prefix: str,
) -> tuple[str, ...]:
    if additional_numeric is None:
        return ()
    _validate_rows(rows)
    widths: set[int] = set()
    for row in rows:
        vector = additional_numeric.get(_feature_row(row).row_id)
        if vector is None:
            raise V2FeatureError(
                f"missing additional activation vector for row {_feature_row(row).row_id!r}"
            )
        widths.add(len(tuple(vector)))
    if len(widths) != 1:
        raise V2FeatureError("additional activation vectors must have one shared width")
    width = next(iter(widths))
    if width <= 0:
        raise V2FeatureError("additional activation vectors must be non-empty")
    # Additional names are deliberately allowed only for C. They are never
    # passed through the A/B feature-lineage auditing helper.
    return tuple(f"{prefix}_{index:04d}" for index in range(width))


def _unpruned_feature_names(
    numeric_feature_names: Sequence[str],
    topic_categories: Sequence[str],
) -> tuple[str, ...]:
    return tuple(numeric_feature_names) + tuple(
        f"{name}__missing" for name in numeric_feature_names
    ) + tuple(f"topic={category}" for category in topic_categories)


def _numeric_feature_names(feature_set: FeatureSet) -> tuple[str, ...]:
    assert_no_future_feature_names(STRUCTURAL_NUMERIC_FEATURES)
    if feature_set is FeatureSet.STRUCTURAL:
        return STRUCTURAL_NUMERIC_FEATURES
    assert_no_future_feature_names(OBSERVABLE_NUMERIC_FEATURES)
    return STRUCTURAL_NUMERIC_FEATURES + OBSERVABLE_NUMERIC_FEATURES


def _coerce_feature_set(value: FeatureSet | str) -> FeatureSet:
    if isinstance(value, FeatureSet):
        return value
    try:
        return FeatureSet(value)
    except (TypeError, ValueError) as error:
        raise V2FeatureError("feature_set must be A/STRUCTURAL or B/OBSERVABLE") from error


def _validate_rows(rows: Sequence[V2FeatureRow | V2PrimaryRow]) -> None:
    if not rows:
        raise V2FeatureError("at least one row is required")
    row_ids = []
    for row in rows:
        if not isinstance(row, (V2FeatureRow, V2PrimaryRow)):
            raise V2FeatureError("rows must be V2FeatureRow or V2PrimaryRow")
        row_ids.append(_feature_row(row).row_id)
    if len(set(row_ids)) != len(row_ids):
        raise V2FeatureError("row IDs must be unique within one fit/transform call")


def _feature_row(row: V2FeatureRow | V2PrimaryRow) -> V2FeatureRow:
    return row.feature_row if isinstance(row, V2PrimaryRow) else row


def _problem_id(row: V2FeatureRow | V2PrimaryRow) -> str:
    return _feature_row(row).problem_id


def _number_or_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _validated_additional_vector(
    vector: Sequence[float] | None,
    *,
    expected_width: int,
    row_id: str,
) -> list[float]:
    if vector is None:
        raise V2FeatureError(f"missing additional activation vector for row {row_id!r}")
    values = list(vector)
    if len(values) != expected_width:
        raise V2FeatureError(
            f"additional activation vector for row {row_id!r} has unexpected width"
        )
    for index, value in enumerate(values):
        _require_finite_number(value, f"additional activation value {index}")
    return [float(value) for value in values]


def _require_nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise V2FeatureError(f"{name} must be a non-empty string")


def _require_positive_int(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise V2FeatureError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise V2FeatureError(f"{name} must be a non-negative integer")


def _require_finite_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2FeatureError(f"{name} must be a finite number")
    if not isfinite(float(value)):
        raise V2FeatureError(f"{name} must be finite")


def _require_optional_finite_number(value: object, name: str) -> None:
    if value is not None:
        _require_finite_number(value, name)


def _numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "numpy is required for V2 offline feature preprocessing"
        ) from error
    return np
