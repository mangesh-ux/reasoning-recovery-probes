"""Leakage-safe V2 A/B/C probe fitting and aggregate-only evaluation.

The module deliberately keeps model fitting separate from trajectory collection.
It accepts only already-enrolled primary rows and private activation matrices;
the terminal answer, gold answer, and evaluator records never enter a feature
matrix.  Validation selects regularization and a primary layer.  Test fitting
uses that immutable selection but never changes it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .v2_config import V2Config
from .v2_features import (
    FeatureSet,
    V2FeatureError,
    V2PrimaryRow,
    equal_total_problem_weights,
    feature_matrix_fit_transform,
)


class V2ProbeError(RuntimeError):
    """Raised when a frozen V2 modelling contract cannot be satisfied."""


class V2ScientificBlock(V2ProbeError):
    """A predeclared event gate failed; callers must preserve the block."""

    def __init__(self, message: str, *, payload: Mapping[str, object]) -> None:
        super().__init__(message)
        self.payload = dict(payload)


@dataclass(frozen=True)
class V2EventGate:
    """Aggregate class-support evidence for one frozen split."""

    split: str
    row_counts: Mapping[str, int]
    problem_counts: Mapping[str, int]
    required_rows_per_class: int
    required_problems_per_class: int
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "row_counts": dict(sorted(self.row_counts.items())),
            "problem_counts": dict(sorted(self.problem_counts.items())),
            "required_rows_per_class": self.required_rows_per_class,
            "required_problems_per_class": self.required_problems_per_class,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class V2MatchedGate:
    """Aggregate availability of the fixed within-problem control."""

    group_count: int
    unique_problem_count: int
    required_group_count: int
    required_problem_count: int
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "matched_problem_checkpoint_groups": self.group_count,
            "matched_unique_problems": self.unique_problem_count,
            "required_groups": self.required_group_count,
            "required_unique_problems": self.required_problem_count,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class V2ActivationCohort:
    """Rows retained for the paired B-versus-C estimand plus attrition."""

    rows: tuple[V2PrimaryRow, ...]
    matrices: Mapping[str, Any]
    total_primary_rows: int
    available_primary_rows: int
    missing_activation_rows: int
    invalid_activation_rows: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_primary_rows": self.total_primary_rows,
            "activation_available_primary_rows": self.available_primary_rows,
            "activation_missing_primary_rows": self.missing_activation_rows,
            "activation_invalid_primary_rows": self.invalid_activation_rows,
        }


@dataclass(frozen=True)
class V2SelectedProbe:
    """Validation-selected fixed model specification, without a fitted object."""

    model_name: str
    feature_set: str
    regularization_c: float
    layer_index: int | None
    validation_metrics: Mapping[str, float]
    train_row_count: int
    validation_row_count: int
    output_feature_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "feature_set": self.feature_set,
            "regularization_c": self.regularization_c,
            "layer_index": self.layer_index,
            "validation_metrics": _metric_dict(self.validation_metrics),
            "train_row_count": self.train_row_count,
            "validation_row_count": self.validation_row_count,
            "output_feature_count": self.output_feature_count,
        }


@dataclass(frozen=True)
class V2ValidationSelection:
    """Immutable validation outcome required before test trajectory collection."""

    status: str
    paired_cohort: Mapping[str, V2ActivationCohort]
    event_gates: Mapping[str, V2EventGate]
    selected_a: V2SelectedProbe | None
    selected_b: V2SelectedProbe | None
    selected_c_by_layer: Mapping[int, V2SelectedProbe]
    primary_layer: int | None
    block_reasons: tuple[str, ...]

    @property
    def ready_for_test(self) -> bool:
        return self.status == "READY_FOR_TEST" and self.primary_layer is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "activation_comparison_cohort": {
                split: cohort.to_dict()
                for split, cohort in sorted(self.paired_cohort.items())
            },
            "event_gates": {
                split: gate.to_dict() for split, gate in sorted(self.event_gates.items())
            },
            "selected_a": self.selected_a.to_dict() if self.selected_a else None,
            "selected_b": self.selected_b.to_dict() if self.selected_b else None,
            "selected_c_by_layer": {
                str(layer): selected.to_dict()
                for layer, selected in sorted(self.selected_c_by_layer.items())
            },
            "validation_selected_primary_layer": self.primary_layer,
            "block_reasons": list(self.block_reasons),
        }

    def to_private_dict(self) -> dict[str, object]:
        """Add private stable row identities needed to prove a later refit matches.

        This selection receipt is stored only under the ignored V2 artifact
        directory.  Public summaries must use :meth:`to_dict` instead.
        """

        payload = self.to_dict()
        payload["private_activation_cohort_row_ids"] = {
            split: [row.row_id for row in cohort.rows]
            for split, cohort in sorted(self.paired_cohort.items())
        }
        return payload


@dataclass(frozen=True)
class V2FittedProbe:
    """Private refit model bundle.  It is serializable with joblib, not JSON."""

    selected: V2SelectedProbe
    preprocessor: Any
    classifier: Any

    def predict_proba(
        self,
        rows: Sequence[V2PrimaryRow],
        *,
        activation_vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> Any:
        matrix = self.preprocessor.transform(rows, additional_numeric=activation_vectors)
        probabilities = self.classifier.predict_proba(matrix)
        np = _numpy()
        values = np.asarray(probabilities[:, 1], dtype=float)
        if values.ndim != 1 or values.size != len(rows):
            raise V2ProbeError("classifier produced an invalid probability shape")
        if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0.0)) or bool(
            np.any(values > 1.0)
        ):
            raise V2ProbeError("classifier produced invalid recovery probabilities")
        return values


@dataclass(frozen=True)
class V2FinalModels:
    """Private train-plus-validation refits under an immutable selection."""

    selected_a: V2FittedProbe
    selected_b: V2FittedProbe
    selected_c_by_layer: Mapping[int, V2FittedProbe]
    primary_layer: int

    def joblib_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "record_type": "V2_FINAL_MODEL_BUNDLE",
            "primary_layer": self.primary_layer,
            "selected_a": self.selected_a,
            "selected_b": self.selected_b,
            "selected_c_by_layer": dict(self.selected_c_by_layer),
        }


def event_gate(
    rows: Sequence[V2PrimaryRow], *, split: str, config: V2Config
) -> V2EventGate:
    """Evaluate the predeclared class/independent-problem floor."""

    if split == "train":
        required_rows = config.analysis.min_train_rows_per_class
        required_problems = config.analysis.min_train_problems_per_class
    elif split == "validation":
        required_rows = config.analysis.min_validation_rows_per_class
        required_problems = config.analysis.min_validation_problems_per_class
    elif split == "test":
        required_rows = config.analysis.min_test_rows_per_class
        required_problems = config.analysis.min_test_problems_per_class
    else:
        raise V2ProbeError(f"unsupported V2 split for event gate: {split!r}")
    _validate_primary_rows(rows)
    row_counts = Counter(row.transition_label for row in rows)
    problem_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        problem_sets[row.transition_label].add(row.problem_id)
    problem_counts = {label: len(problem_sets[label]) for label in _LABELS}
    reasons: list[str] = []
    for label in _LABELS:
        count = int(row_counts[label])
        problems = int(problem_counts[label])
        if count < required_rows:
            reasons.append(
                f"{split}:{label} rows {count} below frozen floor {required_rows}"
            )
        if problems < required_problems:
            reasons.append(
                f"{split}:{label} unique problems {problems} below frozen floor {required_problems}"
            )
    return V2EventGate(
        split=split,
        row_counts={label: int(row_counts[label]) for label in _LABELS},
        problem_counts=problem_counts,
        required_rows_per_class=required_rows,
        required_problems_per_class=required_problems,
        passed=not reasons,
        failure_reasons=tuple(reasons),
    )


def matched_gate(rows: Sequence[V2PrimaryRow], *, config: V2Config) -> V2MatchedGate:
    """Count same-problem/same-coordinate label-contrast groups without scores."""

    _validate_primary_rows(rows)
    labels_by_group: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        labels_by_group[(row.problem_id, row.checkpoint_token)].add(row.transition_label)
    qualifying = [
        group for group, labels in labels_by_group.items() if set(_LABELS).issubset(labels)
    ]
    problems = {problem_id for problem_id, _position in qualifying}
    reasons: list[str] = []
    if len(qualifying) < config.analysis.min_matched_problem_checkpoint_groups:
        reasons.append(
            "matched problem/checkpoint groups "
            f"{len(qualifying)} below frozen floor "
            f"{config.analysis.min_matched_problem_checkpoint_groups}"
        )
    if len(problems) < config.analysis.min_matched_unique_problems:
        reasons.append(
            "matched unique problems "
            f"{len(problems)} below frozen floor {config.analysis.min_matched_unique_problems}"
        )
    return V2MatchedGate(
        group_count=len(qualifying),
        unique_problem_count=len(problems),
        required_group_count=config.analysis.min_matched_problem_checkpoint_groups,
        required_problem_count=config.analysis.min_matched_unique_problems,
        passed=not reasons,
        failure_reasons=tuple(reasons),
    )


def activation_available_cohort(
    rows: Sequence[V2PrimaryRow],
    activation_matrices: Mapping[str, Any],
    *,
    config: V2Config,
) -> V2ActivationCohort:
    """Retain only valid private [layer, width] matrices; never impute them."""

    _validate_primary_rows(rows)
    np = _numpy()
    kept_rows: list[V2PrimaryRow] = []
    kept: dict[str, Any] = {}
    missing = 0
    invalid = 0
    expected_shape = (
        config.model.expected_num_hidden_layers,
        config.model.expected_hidden_size,
    )
    for row in rows:
        matrix = activation_matrices.get(row.row_id)
        if matrix is None:
            missing += 1
            continue
        try:
            array = np.asarray(matrix, dtype=float)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if array.shape != expected_shape or not bool(np.all(np.isfinite(array))):
            invalid += 1
            continue
        kept_rows.append(row)
        kept[row.row_id] = array
    return V2ActivationCohort(
        rows=tuple(sorted(kept_rows, key=lambda row: row.row_id)),
        matrices=kept,
        total_primary_rows=len(rows),
        available_primary_rows=len(kept_rows),
        missing_activation_rows=missing,
        invalid_activation_rows=invalid,
    )


def select_validation_models(
    *,
    train_rows: Sequence[V2PrimaryRow],
    validation_rows: Sequence[V2PrimaryRow],
    activation_matrices: Mapping[str, Any],
    config: V2Config,
) -> V2ValidationSelection:
    """Tune A/B/C only on train/validation paired activation cohorts.

    The returned value contains no fitted estimators.  That makes it suitable
    for the immutable test-unblinding selection receipt.  Fitting after this
    point occurs only via :func:`refit_models_for_test` with its fixed choices.
    """

    train_cohort = activation_available_cohort(
        train_rows, activation_matrices, config=config
    )
    validation_cohort = activation_available_cohort(
        validation_rows, activation_matrices, config=config
    )
    gates = {
        "train": event_gate(train_cohort.rows, split="train", config=config),
        "validation": event_gate(
            validation_cohort.rows, split="validation", config=config
        ),
    }
    reasons = tuple(
        reason
        for split in ("train", "validation")
        for reason in gates[split].failure_reasons
    )
    if reasons:
        return V2ValidationSelection(
            status="SCIENTIFIC_BLOCK_EVENT_GATE",
            paired_cohort={"train": train_cohort, "validation": validation_cohort},
            event_gates=gates,
            selected_a=None,
            selected_b=None,
            selected_c_by_layer={},
            primary_layer=None,
            block_reasons=reasons,
        )

    selected_a = _select_probe(
        model_name="A",
        feature_set=FeatureSet.STRUCTURAL,
        train_rows=train_cohort.rows,
        validation_rows=validation_cohort.rows,
        config=config,
        layer_index=None,
        train_activation_vectors=None,
        validation_activation_vectors=None,
    )
    selected_b = _select_probe(
        model_name="B",
        feature_set=FeatureSet.OBSERVABLE,
        train_rows=train_cohort.rows,
        validation_rows=validation_cohort.rows,
        config=config,
        layer_index=None,
        train_activation_vectors=None,
        validation_activation_vectors=None,
    )
    selected_c: dict[int, V2SelectedProbe] = {}
    for layer_index in range(config.model.expected_num_hidden_layers):
        selected_c[layer_index] = _select_probe(
            model_name=f"C_layer_{layer_index}",
            feature_set=FeatureSet.OBSERVABLE,
            train_rows=train_cohort.rows,
            validation_rows=validation_cohort.rows,
            config=config,
            layer_index=layer_index,
            train_activation_vectors=_layer_vectors(train_cohort, layer_index),
            validation_activation_vectors=_layer_vectors(validation_cohort, layer_index),
        )
    primary_layer = min(
        selected_c,
        key=lambda layer: (
            selected_c[layer].validation_metrics["log_loss"],
            layer,
        ),
    )
    return V2ValidationSelection(
        status="READY_FOR_TEST",
        paired_cohort={"train": train_cohort, "validation": validation_cohort},
        event_gates=gates,
        selected_a=selected_a,
        selected_b=selected_b,
        selected_c_by_layer=selected_c,
        primary_layer=primary_layer,
        block_reasons=(),
    )


def refit_models_for_test(
    *,
    train_rows: Sequence[V2PrimaryRow],
    validation_rows: Sequence[V2PrimaryRow],
    activation_matrices: Mapping[str, Any],
    selection: V2ValidationSelection,
    config: V2Config,
) -> V2FinalModels:
    """Refit fixed A/B/all-C choices exactly once on train plus validation."""

    if not selection.ready_for_test or selection.selected_a is None or selection.selected_b is None:
        raise V2ProbeError("cannot refit V2 models without a ready validation selection")
    cohort = activation_available_cohort(
        tuple(train_rows) + tuple(validation_rows), activation_matrices, config=config
    )
    # A clean, source-bound resume must reproduce exactly the cohort seen at
    # validation time.  Missing activations after selection are evidence of a
    # contract break, not a reason to silently refit a different model.
    expected_ids = {
        row.row_id
        for split in ("train", "validation")
        for row in selection.paired_cohort[split].rows
    }
    actual_ids = {row.row_id for row in cohort.rows}
    if actual_ids != expected_ids:
        raise V2ProbeError("activation-available train/validation cohort changed after selection")
    selected_a = _fit_selected(
        selection.selected_a, cohort.rows, activation_vectors=None, config=config
    )
    selected_b = _fit_selected(
        selection.selected_b, cohort.rows, activation_vectors=None, config=config
    )
    selected_c = {
        layer: _fit_selected(
            selected,
            cohort.rows,
            activation_vectors=_layer_vectors(cohort, layer),
            config=config,
        )
        for layer, selected in selection.selected_c_by_layer.items()
    }
    assert selection.primary_layer is not None
    if selection.primary_layer not in selected_c:
        raise V2ProbeError("validation-selected V2 layer is absent from final refits")
    return V2FinalModels(
        selected_a=selected_a,
        selected_b=selected_b,
        selected_c_by_layer=selected_c,
        primary_layer=selection.primary_layer,
    )


def evaluate_test_models(
    *,
    models: V2FinalModels,
    test_rows: Sequence[V2PrimaryRow],
    activation_matrices: Mapping[str, Any],
    config: V2Config,
) -> "V2TestEvaluation":
    """Run the one terminal aggregate test evaluation under frozen selections."""

    cohort = activation_available_cohort(test_rows, activation_matrices, config=config)
    gate = event_gate(cohort.rows, split="test", config=config)
    contrast_gate = matched_gate(cohort.rows, config=config)
    reasons = tuple(gate.failure_reasons) + tuple(contrast_gate.failure_reasons)
    if reasons:
        raise V2ScientificBlock(
            "test event gate failed; no model result is evaluable",
            payload={
                "test_activation_comparison_cohort": cohort.to_dict(),
                "test_event_gate": gate.to_dict(),
                "test_matched_gate": contrast_gate.to_dict(),
                "block_reasons": list(reasons),
            },
        )
    rows = cohort.rows
    targets = _targets(rows)
    weights = _numpy().asarray(equal_total_problem_weights(rows), dtype=float)
    scores_a = models.selected_a.predict_proba(rows)
    scores_b = models.selected_b.predict_proba(rows)
    c_scores: dict[int, Any] = {}
    for layer, model in models.selected_c_by_layer.items():
        c_scores[layer] = model.predict_proba(
            rows, activation_vectors=_layer_vectors(cohort, layer)
        )
    primary_scores = c_scores[models.primary_layer]
    metrics_a = probability_metrics(targets, scores_a, sample_weight=weights)
    metrics_b = probability_metrics(targets, scores_b, sample_weight=weights)
    metrics_c = probability_metrics(targets, primary_scores, sample_weight=weights)
    layer_curve = {
        layer: _layer_curve_entry(
            layer=layer,
            metrics_b=metrics_b,
            metrics_c=probability_metrics(targets, score, sample_weight=weights),
        )
        for layer, score in sorted(c_scores.items())
    }
    bootstrap = clustered_bootstrap_b_vs_c(
        rows=rows,
        targets=targets,
        scores_b=scores_b,
        scores_c=primary_scores,
        config=config,
    )
    matched = matched_concordance_b_vs_c(
        rows=rows,
        scores_b=scores_b,
        scores_c=primary_scores,
        config=config,
    )
    sensitivity = {
        "unweighted_row_metrics": {
            "A": probability_metrics(targets, scores_a, sample_weight=None),
            "B": probability_metrics(targets, scores_b, sample_weight=None),
            "C_selected_layer": probability_metrics(
                targets, primary_scores, sample_weight=None
            ),
        }
    }
    calibration = {
        "A": calibration_deciles(
            targets, scores_a, bins=config.analysis.calibration_bins
        ),
        "B": calibration_deciles(
            targets, scores_b, bins=config.analysis.calibration_bins
        ),
        "C_selected_layer": calibration_deciles(
            targets, primary_scores, bins=config.analysis.calibration_bins
        ),
    }
    return V2TestEvaluation(
        test_cohort=cohort,
        event_gate=gate,
        matched_gate=contrast_gate,
        primary_layer=models.primary_layer,
        metrics={"A": metrics_a, "B": metrics_b, "C_selected_layer": metrics_c},
        layer_curve=layer_curve,
        bootstrap=bootstrap,
        matched=matched,
        calibration=calibration,
        sensitivity=sensitivity,
    )


@dataclass(frozen=True)
class V2TestEvaluation:
    """Aggregate-safe terminal test result; no row IDs, answers, or tensors."""

    test_cohort: V2ActivationCohort
    event_gate: V2EventGate
    matched_gate: V2MatchedGate
    primary_layer: int
    metrics: Mapping[str, Mapping[str, float]]
    layer_curve: Mapping[int, Mapping[str, object]]
    bootstrap: Mapping[str, object]
    matched: Mapping[str, object]
    calibration: Mapping[str, object]
    sensitivity: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "test_activation_comparison_cohort": self.test_cohort.to_dict(),
            "test_event_gate": self.event_gate.to_dict(),
            "test_matched_gate": self.matched_gate.to_dict(),
            "validation_selected_primary_layer": self.primary_layer,
            "held_out_metrics": {
                name: _metric_dict(values) for name, values in sorted(self.metrics.items())
            },
            "complete_test_layer_curve": {
                str(layer): dict(entry) for layer, entry in sorted(self.layer_curve.items())
            },
            "clustered_bootstrap": dict(self.bootstrap),
            "within_problem_matched_control": dict(self.matched),
            "calibration_deciles": dict(self.calibration),
            "sensitivity": dict(self.sensitivity),
        }


def probability_metrics(
    targets: Sequence[int], probabilities: Any, *, sample_weight: Any | None
) -> dict[str, float]:
    """Compute frozen scalar metrics with an explicit positive label of one."""

    np = _numpy()
    metrics = _metrics()
    y = np.asarray(targets, dtype=int)
    score = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or score.ndim != 1 or y.size != score.size or y.size == 0:
        raise V2ProbeError("metric inputs must be equally sized non-empty vectors")
    if set(int(item) for item in y) != {0, 1}:
        raise V2ProbeError("primary recovery metrics require both W_TO_C and W_TO_W")
    if not bool(np.all(np.isfinite(score))) or bool(np.any(score < 0.0)) or bool(
        np.any(score > 1.0)
    ):
        raise V2ProbeError("recovery probabilities must be finite values in [0, 1]")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights is not None and (weights.shape != y.shape or not bool(np.all(weights > 0.0))):
        raise V2ProbeError("metric sample weights must be positive and row-aligned")
    return {
        "pr_auc": float(metrics["average_precision_score"](y, score, sample_weight=weights)),
        "log_loss": float(
            metrics["log_loss"](y, score, labels=[0, 1], sample_weight=weights)
        ),
        "brier": float(metrics["brier_score_loss"](y, score, sample_weight=weights)),
        "auroc": float(metrics["roc_auc_score"](y, score, sample_weight=weights)),
    }


def clustered_bootstrap_b_vs_c(
    *,
    rows: Sequence[V2PrimaryRow],
    targets: Sequence[int],
    scores_b: Any,
    scores_c: Any,
    config: V2Config,
) -> dict[str, object]:
    """Problem-cluster percentile intervals for the predeclared B-to-C deltas."""

    np = _numpy()
    y = np.asarray(targets, dtype=int)
    b = np.asarray(scores_b, dtype=float)
    c = np.asarray(scores_c, dtype=float)
    if len(rows) != y.size or b.shape != y.shape or c.shape != y.shape:
        raise V2ProbeError("bootstrap arrays must align with test primary rows")
    index_by_problem: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        index_by_problem[row.problem_id].append(index)
    problems = tuple(sorted(index_by_problem))
    if not problems:
        raise V2ProbeError("bootstrap requires at least one held-out problem")
    rng = np.random.default_rng(config.analysis.bootstrap_seed)
    deltas: dict[str, list[float]] = {
        "pr_auc_c_minus_b": [],
        "log_loss_b_minus_c": [],
        "brier_b_minus_c": [],
        "auroc_c_minus_b": [],
    }
    invalid = 0
    for _iteration in range(config.analysis.bootstrap_resamples):
        sampled = rng.choice(problems, size=len(problems), replace=True)
        indices: list[int] = []
        for problem_id in sampled:
            indices.extend(index_by_problem[str(problem_id)])
        selected = np.asarray(indices, dtype=int)
        try:
            weights = np.asarray(
                equal_total_problem_weights([rows[int(index)] for index in selected]),
                dtype=float,
            )
            metrics_b = probability_metrics(y[selected], b[selected], sample_weight=weights)
            metrics_c = probability_metrics(y[selected], c[selected], sample_weight=weights)
        except V2ProbeError:
            invalid += 1
            continue
        deltas["pr_auc_c_minus_b"].append(metrics_c["pr_auc"] - metrics_b["pr_auc"])
        deltas["log_loss_b_minus_c"].append(metrics_b["log_loss"] - metrics_c["log_loss"])
        deltas["brier_b_minus_c"].append(metrics_b["brier"] - metrics_c["brier"])
        deltas["auroc_c_minus_b"].append(metrics_c["auroc"] - metrics_b["auroc"])
    valid = config.analysis.bootstrap_resamples - invalid
    valid_fraction = valid / config.analysis.bootstrap_resamples
    intervals: dict[str, object] = {}
    for name, values in deltas.items():
        if valid_fraction < config.analysis.bootstrap_min_valid_fraction or not values:
            intervals[name] = {
                "evaluable": False,
                "percentile_ci_95": None,
                "reason": "valid bootstrap fraction below frozen minimum",
            }
        else:
            intervals[name] = {
                "evaluable": True,
                "percentile_ci_95": [
                    float(np.percentile(values, 2.5)),
                    float(np.percentile(values, 97.5)),
                ],
            }
    return {
        "policy": config.analysis.bootstrap_policy,
        "seed": config.analysis.bootstrap_seed,
        "requested_resamples": config.analysis.bootstrap_resamples,
        "valid_resamples": valid,
        "invalid_resamples": invalid,
        "valid_fraction": valid_fraction,
        "minimum_valid_fraction": config.analysis.bootstrap_min_valid_fraction,
        "deltas": intervals,
    }


def matched_concordance_b_vs_c(
    *,
    rows: Sequence[V2PrimaryRow],
    scores_b: Any,
    scores_c: Any,
    config: V2Config,
) -> dict[str, object]:
    """Compute the specified matched control and its problem bootstrap interval."""

    np = _numpy()
    b = np.asarray(scores_b, dtype=float)
    c = np.asarray(scores_c, dtype=float)
    if b.shape != (len(rows),) or c.shape != (len(rows),):
        raise V2ProbeError("matched score arrays must align with primary rows")
    group_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        group_indices[(row.problem_id, row.checkpoint_token)].append(index)
    per_problem_b: dict[str, list[float]] = defaultdict(list)
    per_problem_c: dict[str, list[float]] = defaultdict(list)
    qualifying_groups = 0
    for (problem_id, _checkpoint), indices in group_indices.items():
        positives = [index for index in indices if rows[index].target == 1]
        negatives = [index for index in indices if rows[index].target == 0]
        if not positives or not negatives:
            continue
        qualifying_groups += 1
        per_problem_b[problem_id].append(_pairwise_concordance(b, positives, negatives))
        per_problem_c[problem_id].append(_pairwise_concordance(c, positives, negatives))
    problems = tuple(sorted(set(per_problem_b) & set(per_problem_c)))
    if not problems:
        raise V2ProbeError("matched concordance requires at least one contrast group")
    problem_b = np.asarray([np.mean(per_problem_b[item]) for item in problems], dtype=float)
    problem_c = np.asarray([np.mean(per_problem_c[item]) for item in problems], dtype=float)
    point_b = float(np.mean(problem_b))
    point_c = float(np.mean(problem_c))
    rng = np.random.default_rng(config.analysis.bootstrap_seed)
    deltas: list[float] = []
    for _iteration in range(config.analysis.bootstrap_resamples):
        indices = rng.integers(0, len(problems), size=len(problems))
        deltas.append(float(np.mean(problem_c[indices] - problem_b[indices])))
    return {
        "matched_problem_checkpoint_groups": qualifying_groups,
        "matched_unique_problems": len(problems),
        "B_concordance": point_b,
        "C_concordance": point_c,
        "c_minus_b": point_c - point_b,
        "bootstrap_policy": config.analysis.bootstrap_policy,
        "bootstrap_seed": config.analysis.bootstrap_seed,
        "bootstrap_valid_resamples": config.analysis.bootstrap_resamples,
        "bootstrap_invalid_resamples": 0,
        "c_minus_b_percentile_ci_95": [
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ],
    }


def calibration_deciles(
    targets: Sequence[int], probabilities: Any, *, bins: int
) -> list[dict[str, object]]:
    """Fixed deciles for aggregate reporting; no post-hoc calibration fit."""

    np = _numpy()
    y = np.asarray(targets, dtype=int)
    score = np.asarray(probabilities, dtype=float)
    if bins <= 1:
        raise V2ProbeError("calibration requires at least two fixed bins")
    output: list[dict[str, object]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (score >= lower) & ((score < upper) if index < bins - 1 else (score <= upper))
        count = int(np.sum(mask))
        output.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_predicted_probability": float(np.mean(score[mask])) if count else None,
                "observed_positive_rate": float(np.mean(y[mask])) if count else None,
            }
        )
    return output


def _select_probe(
    *,
    model_name: str,
    feature_set: FeatureSet,
    train_rows: Sequence[V2PrimaryRow],
    validation_rows: Sequence[V2PrimaryRow],
    config: V2Config,
    layer_index: int | None,
    train_activation_vectors: Mapping[str, Sequence[float]] | None,
    validation_activation_vectors: Mapping[str, Sequence[float]] | None,
) -> V2SelectedProbe:
    candidates: list[tuple[float, float, int, Any]] = []
    for grid_index, regularization_c in enumerate(config.probe.regularization_c_grid):
        fitted = _fit_probe(
            model_name=model_name,
            feature_set=feature_set,
            regularization_c=regularization_c,
            layer_index=layer_index,
            rows=train_rows,
            activation_vectors=train_activation_vectors,
            config=config,
        )
        validation_scores = fitted.predict_proba(
            validation_rows, activation_vectors=validation_activation_vectors
        )
        validation_metrics = probability_metrics(
            _targets(validation_rows),
            validation_scores,
            sample_weight=_numpy().asarray(
                equal_total_problem_weights(validation_rows), dtype=float
            ),
        )
        candidates.append(
            (
                validation_metrics["log_loss"],
                float(regularization_c),
                grid_index,
                (fitted, validation_metrics),
            )
        )
    if not candidates:
        raise V2ProbeError("frozen V2 regularization grid is empty")
    _loss, _c, _grid, selected_pair = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    fitted, validation_metrics = selected_pair
    return V2SelectedProbe(
        model_name=model_name,
        feature_set=feature_set.value,
        regularization_c=fitted.selected.regularization_c,
        layer_index=layer_index,
        validation_metrics=validation_metrics,
        train_row_count=len(train_rows),
        validation_row_count=len(validation_rows),
        output_feature_count=len(fitted.preprocessor.output_feature_names),
    )


def _fit_selected(
    selected: V2SelectedProbe,
    rows: Sequence[V2PrimaryRow],
    *,
    activation_vectors: Mapping[str, Sequence[float]] | None,
    config: V2Config,
) -> V2FittedProbe:
    return _fit_probe(
        model_name=selected.model_name,
        feature_set=FeatureSet(selected.feature_set),
        regularization_c=selected.regularization_c,
        layer_index=selected.layer_index,
        rows=rows,
        activation_vectors=activation_vectors,
        config=config,
    )


def _fit_probe(
    *,
    model_name: str,
    feature_set: FeatureSet,
    regularization_c: float,
    layer_index: int | None,
    rows: Sequence[V2PrimaryRow],
    activation_vectors: Mapping[str, Sequence[float]] | None,
    config: V2Config,
) -> V2FittedProbe:
    _validate_primary_rows(rows)
    if regularization_c <= 0.0 or not isfinite(float(regularization_c)):
        raise V2ProbeError("L2 regularization C must be finite and positive")
    preprocessor, matrix = feature_matrix_fit_transform(
        rows,
        feature_set=feature_set,
        additional_numeric=activation_vectors,
        additional_feature_prefix=(f"activation_layer_{layer_index}" if layer_index is not None else "activation"),
    )
    LogisticRegression = _logistic_regression()
    classifier = LogisticRegression(
        penalty="l2",
        C=float(regularization_c),
        solver=config.probe.solver,
        max_iter=config.probe.max_iter,
        class_weight=None,
        random_state=0,
    )
    targets = _targets(rows)
    classifier.fit(
        matrix,
        targets,
        sample_weight=_numpy().asarray(equal_total_problem_weights(rows), dtype=float),
    )
    selected = V2SelectedProbe(
        model_name=model_name,
        feature_set=feature_set.value,
        regularization_c=float(regularization_c),
        layer_index=layer_index,
        validation_metrics={},
        train_row_count=len(rows),
        validation_row_count=0,
        output_feature_count=len(preprocessor.output_feature_names),
    )
    return V2FittedProbe(selected=selected, preprocessor=preprocessor, classifier=classifier)


def _layer_vectors(
    cohort: V2ActivationCohort, layer_index: int
) -> dict[str, Sequence[float]]:
    vectors: dict[str, Sequence[float]] = {}
    for row in cohort.rows:
        matrix = cohort.matrices.get(row.row_id)
        if matrix is None:
            raise V2ProbeError("activation cohort has a missing retained matrix")
        try:
            vector = matrix[layer_index]
        except (IndexError, TypeError) as error:
            raise V2ProbeError(f"activation layer {layer_index} is unavailable") from error
        vectors[row.row_id] = tuple(float(value) for value in vector)
    return vectors


def _layer_curve_entry(
    *, layer: int, metrics_b: Mapping[str, float], metrics_c: Mapping[str, float]
) -> dict[str, object]:
    return {
        "layer_index": layer,
        "C_metrics": _metric_dict(metrics_c),
        "pr_auc_c_minus_b": metrics_c["pr_auc"] - metrics_b["pr_auc"],
        "log_loss_b_minus_c": metrics_b["log_loss"] - metrics_c["log_loss"],
        "brier_b_minus_c": metrics_b["brier"] - metrics_c["brier"],
        "auroc_c_minus_b": metrics_c["auroc"] - metrics_b["auroc"],
    }


def _pairwise_concordance(scores: Any, positives: Sequence[int], negatives: Sequence[int]) -> float:
    np = _numpy()
    positive = np.asarray([scores[index] for index in positives], dtype=float)
    negative = np.asarray([scores[index] for index in negatives], dtype=float)
    differences = positive[:, None] - negative[None, :]
    return float(np.mean((differences > 0.0).astype(float) + 0.5 * (differences == 0.0)))


def _targets(rows: Sequence[V2PrimaryRow]) -> Any:
    _validate_primary_rows(rows)
    return _numpy().asarray([row.target for row in rows], dtype=int)


def _validate_primary_rows(rows: Sequence[V2PrimaryRow]) -> None:
    if not rows:
        raise V2ProbeError("V2 probe operation requires at least one primary row")
    row_ids = [row.row_id for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise V2ProbeError("V2 primary rows must have unique row IDs")
    if any(not isinstance(row, V2PrimaryRow) for row in rows):
        raise V2ProbeError("V2 probe operation requires V2PrimaryRow values")


def _metric_dict(values: Mapping[str, float]) -> dict[str, float]:
    return {str(name): float(value) for name, value in sorted(values.items())}


def _numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise V2ProbeError("V2 analysis requires NumPy") from error
    return np


def _metrics() -> Mapping[str, Any]:
    try:
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise V2ProbeError("V2 analysis requires scikit-learn") from error
    return {
        "average_precision_score": average_precision_score,
        "brier_score_loss": brier_score_loss,
        "log_loss": log_loss,
        "roc_auc_score": roc_auc_score,
    }


def _logistic_regression() -> Any:
    try:
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise V2ProbeError("V2 probes require scikit-learn") from error
    return LogisticRegression


_LABELS = ("W_TO_C", "W_TO_W")
