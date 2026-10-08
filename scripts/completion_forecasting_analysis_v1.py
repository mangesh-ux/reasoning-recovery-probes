"""CPU-only, write-once grouped selection and held-out completion prognosis.

Consumes derivative features, never original benchmark text, test data or gold.
The new target is NOT recovery/correctness. This does not invoke a language model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit

GRID = (0.001, 0.01, 0.1, 1.0)
MODELS = ("A", "B", "Blex", "C", "Clex")
RUNTIME = {"numpy": "2.2.6", "scipy": "1.14.0", "scikit-learn": "1.5.1"}
REPO = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_once(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def weights(groups):
    counts = Counter(groups)
    return np.array([(len(groups) / len(counts)) / counts[g] for g in groups])


def binary_losses(y, p):
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid prediction probabilities")
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def problem_means(values, groups):
    unique = sorted(set(groups))
    return unique, np.array([np.mean(values[groups == g]) for g in unique])


def score(y, p, groups):
    return float(problem_means(binary_losses(y, p), groups)[1].mean())


def fold_assignments(groups, difficulties):
    by_problem = {}
    for group, difficulty in zip(groups, difficulties):
        require(group not in by_problem or by_problem[group] == difficulty, "Problem difficulty changed")
        by_problem[group] = difficulty
    assignments = {}
    for difficulty in sorted(set(by_problem.values())):
        ranked = sorted((g for g, d in by_problem.items() if d == difficulty), key=lambda g: (hashlib.sha256(("completion-forecasting-v1|fold|" + g).encode()).hexdigest(), g))
        for i, group in enumerate(ranked):
            assignments[group] = i % 5
    return assignments


class Preprocessor:
    """Learn imputation, category vocabulary and scaling from a fit partition."""
    def __init__(self, model, layer=None):
        self.model = model
        self.layer = layer

    def numeric(self, data):
        return data["numeric"][:, :4] if self.model == "A" else data["numeric"]

    def fit(self, data):
        raw = self.numeric(data)
        self.medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in raw.T])
        self.topics = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        self.topics.fit(data["topics"].reshape(-1, 1))
        design = self.unscaled(data)
        w = weights(data["groups"])
        self.mean = np.average(design, axis=0, weights=w)
        var = np.average((design - self.mean) ** 2, axis=0, weights=w)
        self.keep = var > 1e-12
        self.scale = np.sqrt(var[self.keep])
        require(self.keep.any(), "All training features constant")
        return self

    def unscaled(self, data):
        raw = self.numeric(data)
        missing = ~np.isfinite(raw)
        blocks = [np.where(missing, self.medians, raw), missing.astype(float), self.topics.transform(data["topics"].reshape(-1, 1))]
        if self.model in ("Blex", "Clex"):
            blocks.append(data["lexical"])
        if self.model in ("C", "Clex"):
            require(self.layer is not None, "Activation layer absent")
            blocks.append(data["activations"][:, self.layer, :])
        return np.concatenate(blocks, axis=1)

    def transform(self, data):
        return (self.unscaled(data)[:, self.keep] - self.mean[self.keep]) / self.scale


def subset(data, mask):
    return {k: v[mask] for k, v in data.items()}


def predict(train, evaluation, model, strength, layer=None):
    require(set(train["labels"]) == {0, 1}, "Fit partition is single class")
    preprocessing = Preprocessor(model, layer).fit(train)
    estimator = LogisticRegression(C=strength, solver="lbfgs", penalty="l2", max_iter=10000, tol=1e-6, random_state=0)
    w = weights(train["groups"])
    w *= len(w) / w.sum()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(preprocessing.transform(train), train["labels"], sample_weight=w)
    probabilities = estimator.predict_proba(preprocessing.transform(evaluation))[:, 1]
    require(np.isfinite(probabilities).all(), "Model emitted nonfinite probabilities")
    state = {"model": model, "layer": layer, "C": strength, "numeric_medians": preprocessing.medians.tolist(), "topic_vocabulary": preprocessing.topics.categories_[0].tolist(), "kept_columns": np.flatnonzero(preprocessing.keep).tolist(), "means": preprocessing.mean[preprocessing.keep].tolist(), "scales": preprocessing.scale.tolist(), "coefficients": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0])}
    return probabilities, state


def predict_frozen(data, state):
    """Reconstruct saved TRAIN-only linear model; never fit on held-out data."""
    raw = data["numeric"][:, :4] if state["model"] == "A" else data["numeric"]
    missing = ~np.isfinite(raw)
    vocabulary = state["topic_vocabulary"]
    topics = np.array([[float(t == category) for category in vocabulary] for t in data["topics"]])
    blocks = [np.where(missing, np.array(state["numeric_medians"]), raw), missing.astype(float), topics]
    if state["model"] in ("Blex", "Clex"):
        blocks.append(data["lexical"])
    if state["model"] in ("C", "Clex"):
        blocks.append(data["activations"][:, state["layer"], :])
    design = np.concatenate(blocks, axis=1)[:, state["kept_columns"]]
    scaled = (design - np.array(state["means"])) / np.array(state["scales"])
    probabilities = expit(scaled @ np.array(state["coefficients"]) + state["intercept"])
    require(np.isfinite(probabilities).all(), "Saved model emitted nonfinite probabilities")
    return probabilities


def load(path, expected_hash, split):
    require(file_sha(path) == expected_hash, "Derivative data changed")
    with np.load(path, allow_pickle=False) as archive:
        # No arbitrary metadata route into a design matrix.
        data = {name: archive[name] for name in ("numeric", "topics", "groups", "lexical", "activations", "labels", "row_ids", "fold_ids")}
    n = len(data["labels"])
    require(all(len(v) == n for v in data.values()), "Derivative row counts differ")
    require(data["numeric"].shape == (n, 11) and data["lexical"].shape == (n, 512) and data["activations"].shape == (n, 28, 2048), "Derivative feature schema differs")
    require(set(data["labels"]) == {0, 1} and np.isfinite(data["activations"]).all() and np.isfinite(data["lexical"]).all(), "Derivative target/tensor invalid")
    require(not np.isinf(data["numeric"]).any(), "Derivative numeric features contain infinity")
    require(len(set(data["row_ids"])) == n, "Duplicate rows")
    require(len(set(data["groups"])) <= (96 if split == "train" else 32), "Derivative split scope exceeded")
    return data


def bootstrap_interval(values, seed=20261008):
    rng = np.random.default_rng(seed)
    replicates = np.mean(values[rng.integers(0, len(values), size=(2000, len(values)))], axis=1)
    return [float(x) for x in np.quantile(replicates, [0.025, 0.975])]


def paired_bootstrap(values, positive_support, negative_support, seed=20261008):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(2000, len(values)))
    valid = positive_support[indices].any(axis=1) & negative_support[indices].any(axis=1)
    count = int(valid.sum())
    interval = [float(x) for x in np.quantile(values[indices[valid]].mean(axis=1), [0.025, 0.975])] if count >= 1900 else None
    return {"problem_cluster_bootstrap_95_percentile_ci": interval, "bootstrap_resamples": 2000, "valid_resamples": count, "invalid_fraction": 1 - count / 2000, "bootstrap_seed": seed}


def metrics(data, probabilities):
    y, groups = data["labels"], data["groups"]
    w = weights(groups)
    return {"problem_weighted_log_loss_nats": score(y, probabilities, groups), "problem_weighted_brier": float(np.average((probabilities - y) ** 2, weights=w)), "problem_weighted_pr_auc_average_precision": float(average_precision_score(y, probabilities, sample_weight=w)), "problem_weighted_roc_auc": float(roc_auc_score(y, probabilities, sample_weight=w))}


def matched(data, p):
    values = []
    groups = []
    for group in sorted(set(data["groups"])):
        mask = data["groups"] == group
        y, scores = data["labels"][mask], p[mask]
        if set(y) != {0, 1}:
            continue
        delta = scores[y == 1, None] - scores[y == 0][None, :]
        values.append(float(np.mean((delta > 0) + 0.5 * (delta == 0))))
        groups.append(group)
    return groups, np.array(values)


def main(args):
    require({x: importlib.metadata.version(x) for x in RUNTIME} == RUNTIME, "CPU analysis runtime differs")
    output = args.output.resolve()
    require(output.is_relative_to((REPO / "artifacts/completion-forecasting/v1").resolve()), "Analysis output must use new private namespace")
    output.mkdir(parents=True, exist_ok=True)
    protocol_hash = file_sha(args.protocol)
    require(protocol_hash == args.protocol_sha256, "Frozen pivot protocol changed")
    source_hash = file_sha(Path(__file__))
    train = load(args.train, args.train_sha256, "train")
    # Any availability mismatch must be resolved by the loader, never row replacement.
    for label in (0, 1):
        require(sum(train["labels"] == label) >= 50 and len(set(train["groups"][train["labels"] == label])) >= 15, "TRAIN target diversity gate failed")
    if args.mode == "select":
        require(not (output / "selection-intent.json").exists(), "Selection already attempted; no retry")
        folds = train["fold_ids"]
        require(set(folds) == set(range(5)), "Frozen fold identity differs")
        assignments = {g: int(folds[np.flatnonzero(train["groups"] == g)[0]]) for g in set(train["groups"])}
        require(all(len(set(folds[train["groups"] == g])) == 1 for g in assignments), "Problem crosses CV folds")
        for fold in range(5):
            require(set(train["labels"][folds != fold]) == {0, 1}, "Grouped fit partition is single class")
        write_once(output / "selection-intent.json", {"created_at_utc": utc(), "protocol_sha256": protocol_hash, "analysis_source_sha256": source_hash, "train_sha256": args.train_sha256, "runtime": RUNTIME, "fold_assignment": assignments, "grid": GRID, "layers_zero_based": list(range(28)), "paid_gpu_calls": 0})
        scores = []
        failures = []
        for model in ("A", "B", "Blex", "Clex"):
            layers = range(28) if model == "Clex" else (None,)
            for layer in layers:
                for strength in GRID:
                    predictions = np.zeros(len(folds))
                    try:
                        for fold in range(5):
                            predictions[folds == fold], _ = predict(subset(train, folds != fold), subset(train, folds == fold), model, strength, layer)
                        loss = score(train["labels"], predictions, train["groups"])
                        scores.append({"model": model, "layer": layer, "C": strength, "cv_problem_weighted_log_loss_nats": loss})
                        write_once(output / f"cv-config-{len(scores) + len(failures):03d}.json", scores[-1])
                    except Exception as error:
                        failures.append({"model": model, "layer": layer, "C": strength, "error_type": type(error).__name__})
                        write_once(output / f"cv-config-{len(scores) + len(failures):03d}.json", failures[-1])
                    print(json.dumps({"phase": "train_grouped_selection", "model": model, "layer": layer, "C": strength, "configs_finished": len(scores) + len(failures)}), flush=True)
        require(not failures, "A predeclared configuration failed; preserve failure and stop")
        chosen = {}
        for model in ("A", "B", "Blex", "Clex"):
            eligible = [r for r in scores if r["model"] == model]
            # Exact ties prefer stronger regularization, then shallower layer.
            chosen[model] = min(eligible, key=lambda r: (r["cv_problem_weighted_log_loss_nats"], r["C"], -1 if r["layer"] is None else r["layer"]))
        layer = chosen["Clex"]["layer"]
        bare_c = []
        for strength in GRID:
            probabilities = np.zeros(len(folds))
            for fold in range(5):
                probabilities[folds == fold], _ = predict(subset(train, folds != fold), subset(train, folds == fold), "C", strength, layer)
            bare_c.append({"model": "C", "layer": layer, "C": strength, "cv_problem_weighted_log_loss_nats": score(train["labels"], probabilities, train["groups"])})
            write_once(output / f"cv-secondary-C-{len(bare_c):02d}.json", bare_c[-1])
        chosen["C"] = min(bare_c, key=lambda r: (r["cv_problem_weighted_log_loss_nats"], r["C"]))
        fitted = {}
        for model in MODELS:
            config = chosen[model]
            expected, fitted[model] = predict(train, train, model, config["C"], config["layer"])
            require(np.allclose(expected, predict_frozen(train, fitted[model]), rtol=1e-10, atol=1e-12), "Saved-model prediction mismatch")
        prevalence = float(problem_means(train["labels"], train["groups"])[1].mean())
        write_once(output / "fitted-models.json", {"models": fitted, "train_problem_prevalence": prevalence})
        selection = {"created_at_utc": utc(), "record_type": "COMPLETION_FORECAST_TRAIN_SELECTION_V1", "protocol_sha256": protocol_hash, "analysis_source_sha256": source_hash, "train_sha256": args.train_sha256, "fitted_models_sha256": file_sha(output / "fitted-models.json"), "runtime": RUNTIME, "selected": chosen, "cv_scores": scores + bare_c, "failures": failures, "fold_assignment": assignments, "validation_loaded": False, "original_test_loaded": False, "selection_cv_is_not_unbiased_performance": True}
        write_once(output / "selection.json", selection)
        print(json.dumps({"selection_sha256": file_sha(output / "selection.json"), "chosen_layer_zero_based": layer, "heldout_evaluated": False}), flush=True)
    else:
        require(args.validation is not None and args.validation_sha256 is not None, "Held-out inputs absent")
        require(args.selection_sha256 is not None and file_sha(output / "selection.json") == args.selection_sha256, "Selection identity mismatch")
        selection = json.loads((output / "selection.json").read_text())
        require(selection["protocol_sha256"] == protocol_hash and selection["analysis_source_sha256"] == source_hash and selection["train_sha256"] == args.train_sha256, "Selection source/input changed")
        require(file_sha(output / "fitted-models.json") == selection["fitted_models_sha256"], "Frozen model file changed")
        fitted = json.loads((output / "fitted-models.json").read_text())
        require(not (output / "evaluation-intent.json").exists(), "Held-out evaluation consumed; no retry")
        write_once(output / "evaluation-intent.json", {"created_at_utc": utc(), "selection_sha256": args.selection_sha256, "validation_sha256": args.validation_sha256, "no_refit_on_validation": True, "original_test_loaded": False})
        validation = load(args.validation, args.validation_sha256, "validation")
        require(not (set(validation["groups"]) & set(train["groups"])), "Problem leakage across partitions")
        for label in (0, 1):
            require(sum(validation["labels"] == label) >= 20 and len(set(validation["groups"][validation["labels"] == label])) >= 8, "Held-out target diversity failed")
        predictions = {}
        for model in MODELS:
            predictions[model] = predict_frozen(validation, fitted["models"][model])
        results = {model: metrics(validation, p) for model, p in predictions.items()}
        results["TRAIN_prevalence"] = metrics(validation, np.full(len(validation["labels"]), fitted["train_problem_prevalence"]))
        paired = {}
        for baseline, activation in (("Blex", "Clex"), ("B", "C"), ("B", "Blex")):
            groups, deltas = problem_means(binary_losses(validation["labels"], predictions[baseline]) - binary_losses(validation["labels"], predictions[activation]), validation["groups"])
            positive = np.array([np.any(validation["labels"][validation["groups"] == g] == 1) for g in groups])
            negative = np.array([np.any(validation["labels"][validation["groups"] == g] == 0) for g in groups])
            paired[activation + "_vs_" + baseline] = {"mean_problem_log_loss_improvement_nats": float(deltas.mean()), **paired_bootstrap(deltas, positive, negative), "problem_count": len(deltas), "positive_favors_added_predictors": True}
        mg, concordance = matched(validation, predictions["Clex"])
        _, baseline_concordance = matched(validation, predictions["Blex"])
        matched_result = {"matched_problem_count": len(mg), "adequate_support": len(mg) >= 8, "Clex_concordance": float(concordance.mean()) if len(mg) else None, "Blex_concordance": float(baseline_concordance.mean()) if len(mg) else None, "paired_concordance_difference": float((concordance - baseline_concordance).mean()) if len(mg) else None, "difference_bootstrap_95_percentile_ci": bootstrap_interval(concordance - baseline_concordance) if len(mg) >= 8 else None, "bootstrap_resamples": 2000 if len(mg) >= 8 else 0, "bootstrap_valid_resamples": 2000 if len(mg) >= 8 else 0}
        record = {"record_type": "COMPLETION_FORECAST_VALIDATION_HELDOUT_RESULT_V1", "completed_at_utc": utc(), "question": "Future natural thinking completion before the frozen token cap, not correctness or recovery", "protocol_sha256": protocol_hash, "selection_sha256": args.selection_sha256, "fitted_models_sha256": selection["fitted_models_sha256"], "train_sha256": args.train_sha256, "validation_sha256": args.validation_sha256, "metrics": results, "paired_comparisons": paired, "matched_control": matched_result, "rows": len(validation["labels"]), "problems": len(set(validation["groups"])), "positive_rows": int(validation["labels"].sum()), "problem_weighted_prevalence": float(problem_means(validation["labels"], validation["groups"])[1].mean()), "original_test_loaded": False, "gpu_calls": 0, "no_validation_refit": True}
        write_once(output / "heldout-result.json", record)
        with (output / "heldout-predictions.npz").open("xb") as f:
            np.savez_compressed(f, labels=validation["labels"], groups=validation["groups"], row_ids=validation["row_ids"], **predictions)
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("select", "evaluate"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--validation-sha256")
    parser.add_argument("--selection-sha256")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        output = args.output.resolve()
        if output.is_relative_to((REPO / "artifacts/completion-forecasting/v1").resolve()) and output.exists():
            failure = output / (args.mode + "-failure.json")
            if not failure.exists():
                write_once(failure, {"failed_at_utc": utc(), "error_type": type(error).__name__, "safe_message": str(error) if type(error) is RuntimeError else None, "automatic_retry_allowed": False})
        print(json.dumps({"status": "CPU_ANALYSIS_FAILED", "mode": args.mode, "error_type": type(error).__name__}), flush=True)
        raise SystemExit(2)
