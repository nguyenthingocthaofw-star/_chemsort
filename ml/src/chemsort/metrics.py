"""Classification metrics shared by every model family.

Every evaluation goes through ``evaluate_probabilities``: it takes the
9-class softmax of any model (Keras, PyTorch YOLO or TFLite), scores it at
the 9-class RealWaste level and at the 6-group app level (probabilities are
summed per group before argmax, the same rule as ``mapping_service.dart``),
and returns plain JSON-serialisable dictionaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .contracts import CHEMSORT_CONTRACT

CALIBRATION_BINS = 15


def normalize_outputs(outputs: np.ndarray) -> np.ndarray:
    """Return probabilities; apply softmax only if rows are not already ones."""
    values = np.asarray(outputs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected [batch, classes] outputs, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Model outputs contain NaN or Inf")
    looks_like_probabilities = (
        np.all(values >= -1e-6)
        and np.all(values <= 1 + 1e-6)
        and np.allclose(values.sum(axis=1), 1.0, atol=1e-3)
    )
    if not looks_like_probabilities:
        shifted = values - values.max(axis=1, keepdims=True)
        values = np.exp(shifted)
    values = np.clip(values, 0.0, None)
    return values / values.sum(axis=1, keepdims=True)


def aggregate_to_app(probabilities_9: np.ndarray) -> np.ndarray:
    return np.column_stack(
        CHEMSORT_CONTRACT.aggregate_probability_columns(np.asarray(probabilities_9))
    )


def labels_to_app(labels_9: Sequence[int]) -> np.ndarray:
    to_app = np.asarray(CHEMSORT_CONTRACT.model_to_app_ids(), dtype=np.int64)
    return to_app[np.asarray(labels_9, dtype=np.int64)]


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        members = bins == index
        if members.any():
            gap = abs(correct[members].mean() - confidence[members].mean())
            error += members.mean() * gap
    return float(error)


def calibration_curve(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> dict[str, list[float]]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    curve: dict[str, list[float]] = {"bin_center": [], "accuracy": [], "confidence": [], "count": []}
    for index in range(n_bins):
        members = bins == index
        if not members.any():
            continue
        curve["bin_center"].append(float((edges[index] + edges[index + 1]) / 2))
        curve["accuracy"].append(float(correct[members].mean()))
        curve["confidence"].append(float(confidence[members].mean()))
        curve["count"].append(int(members.sum()))
    return curve


def _top_k_accuracy(probabilities: np.ndarray, y_true: np.ndarray, k: int) -> float:
    top_k = np.argsort(-probabilities, axis=1)[:, :k]
    return float(np.mean(np.any(top_k == y_true[:, np.newaxis], axis=1)))


def _macro_roc_auc(probabilities: np.ndarray, y_true: np.ndarray) -> float | None:
    present = np.unique(y_true)
    if len(present) < 2:
        return None
    scores = []
    for class_id in present:
        positives = y_true == class_id
        if positives.all():
            continue
        scores.append(roc_auc_score(positives, probabilities[:, class_id]))
    return float(np.mean(scores)) if scores else None


def classification_metrics(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Full metric block for one label space.

    Macro averages run over *all* classes, so a class with no test images
    (Vegetation in ``test_ext``) scores 0 and lowers the macro value; the
    ``*_present`` variants average only the classes that occur.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    class_count = len(class_names)
    if probabilities.shape != (len(y_true), class_count):
        raise ValueError(
            f"Probabilities shape {probabilities.shape} does not match "
            f"{len(y_true)} samples x {class_count} classes"
        )
    labels = np.arange(class_count)
    y_pred = probabilities.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    predicted_counts = matrix.sum(axis=0)
    true_negative = matrix.sum() - matrix.sum(axis=0) - matrix.sum(axis=1) + np.diag(matrix)
    false_positive = predicted_counts - np.diag(matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        specificity = np.where(
            true_negative + false_positive > 0,
            true_negative / (true_negative + false_positive),
            0.0,
        )
    present = support > 0
    weights = support / support.sum()

    block: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_f1_present": float(f1[present].mean()),
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_recall": float(np.sum(recall * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(
            log_loss(y_true, np.clip(probabilities, 1e-12, 1.0), labels=labels)
        ),
        "brier_score": float(
            np.mean(np.sum((probabilities - np.eye(class_count)[y_true]) ** 2, axis=1))
        ),
        "ece": expected_calibration_error(probabilities, y_true),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "roc_auc_ovr_macro": _macro_roc_auc(probabilities, y_true),
        "classes_present": int(present.sum()),
        "class_names": list(class_names),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "specificity": float(specificity[index]),
                "support": int(support[index]),
                "predicted": int(predicted_counts[index]),
            }
            for index, name in enumerate(class_names)
        },
    }
    for k in (2, 3):
        if class_count > k:
            block[f"top{k}_accuracy"] = _top_k_accuracy(probabilities, y_true, k)
    return block


def headline_metrics(block_9: dict[str, Any], block_6: dict[str, Any]) -> dict[str, Any]:
    """Flat dictionary used for comparison tables (legacy key names kept)."""
    summary: dict[str, Any] = {}
    for suffix, block in (("9class", block_9), ("6class", block_6)):
        for key in (
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "macro_f1_present",
            "weighted_f1",
            "cohen_kappa",
            "mcc",
            "log_loss",
            "ece",
            "roc_auc_ovr_macro",
            "top3_accuracy",
        ):
            if key in block:
                summary[f"{key}_{suffix}"] = block[key]
        for name, values in block["per_class"].items():
            slug = name.lower().replace("/", "_").replace(" ", "_")
            summary[f"f1_{suffix}_{slug}"] = values["f1"]
    per_class_6 = block_6["per_class"]
    summary["organic_precision_6class"] = per_class_6["Organic"]["precision"]
    summary["organic_recall_6class"] = per_class_6["Organic"]["recall"]
    summary["plastic_precision_6class"] = per_class_6["Plastic"]["precision"]
    summary["plastic_recall_6class"] = per_class_6["Plastic"]["recall"]
    return summary


def evaluate_probabilities(
    y_true_9: Sequence[int],
    probabilities_9: np.ndarray,
) -> dict[str, Any]:
    probabilities_9 = normalize_outputs(probabilities_9)
    y_true_9 = np.asarray(y_true_9, dtype=np.int64)
    probabilities_6 = aggregate_to_app(probabilities_9)
    y_true_6 = labels_to_app(y_true_9)
    block_9 = classification_metrics(
        y_true_9, probabilities_9, CHEMSORT_CONTRACT.model_class_names
    )
    block_6 = classification_metrics(
        y_true_6, probabilities_6, CHEMSORT_CONTRACT.app_class_names
    )
    return {
        "summary": headline_metrics(block_9, block_6),
        "9class": block_9,
        "6class": block_6,
    }


def _macro_f1_from_confusion(matrices: np.ndarray) -> np.ndarray:
    true_positive = np.diagonal(matrices, axis1=1, axis2=2).astype(np.float64)
    predicted = matrices.sum(axis=1)
    actual = matrices.sum(axis=2)
    denominator = predicted + actual
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denominator > 0, 2 * true_positive / denominator, 0.0)
    return f1.mean(axis=1)


def bootstrap_intervals(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_count: int,
    samples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap intervals for accuracy and macro-F1."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(y_true), size=(samples, len(y_true)))
    flat = y_true[indices] * class_count + y_pred[indices]
    offsets = np.arange(samples)[:, np.newaxis] * class_count * class_count
    counts = np.bincount(
        (flat + offsets).ravel(), minlength=samples * class_count * class_count
    ).reshape(samples, class_count, class_count)
    accuracy = np.trace(counts, axis1=1, axis2=2) / len(y_true)
    macro_f1 = _macro_f1_from_confusion(counts)
    tail = (1 - confidence) / 2 * 100
    return {
        name: {
            "lower": float(np.percentile(values, tail)),
            "upper": float(np.percentile(values, 100 - tail)),
            "std": float(values.std(ddof=1)),
            "samples": samples,
            "seed": seed,
        }
        for name, values in (("accuracy", accuracy), ("macro_f1", macro_f1))
    }


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, Any]:
    """Paired test of whether two models differ in error rate on the same images.

    Uses the exact binomial test when fewer than 25 discordant pairs exist,
    otherwise the chi-squared statistic with continuity correction.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError("Both models must be scored on the same images")
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    discordant = only_a + only_b
    if discordant == 0:
        return {"only_a_correct": 0, "only_b_correct": 0, "statistic": 0.0,
                "p_value": 1.0, "method": "none"}
    if discordant < 25:
        p_value = stats.binomtest(only_a, discordant, 0.5).pvalue
        return {"only_a_correct": only_a, "only_b_correct": only_b,
                "statistic": float(min(only_a, only_b)),
                "p_value": float(p_value), "method": "exact_binomial"}
    statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
    return {"only_a_correct": only_a, "only_b_correct": only_b,
            "statistic": float(statistic),
            "p_value": float(stats.chi2.sf(statistic, df=1)),
            "method": "chi2_continuity"}


def format_classification_report(block: dict[str, Any], title: str = "") -> str:
    """Text report in the style of sklearn, plus specificity and extras."""
    names = block["class_names"]
    width = max(12, max(len(name) for name in names) + 2)
    lines = []
    if title:
        lines += [title, "=" * len(title), ""]
    header = f"{'':<{width}}{'precision':>10}{'recall':>10}{'f1-score':>10}{'specif.':>10}{'support':>10}"
    lines.append(header)
    lines.append("")
    for name in names:
        row = block["per_class"][name]
        lines.append(
            f"{name:<{width}}{row['precision']:>10.4f}{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}{row['specificity']:>10.4f}{row['support']:>10d}"
        )
    total = block["n_samples"]
    lines += [
        "",
        f"{'accuracy':<{width}}{'':>30}{block['accuracy']:>10.4f}{'':>10}{total:>10d}",
        f"{'macro avg':<{width}}{block['macro_precision']:>10.4f}{block['macro_recall']:>10.4f}"
        f"{block['macro_f1']:>10.4f}{'':>10}{total:>10d}",
        f"{'weighted avg':<{width}}{block['weighted_precision']:>10.4f}"
        f"{block['weighted_recall']:>10.4f}{block['weighted_f1']:>10.4f}{'':>10}{total:>10d}",
        "",
        f"balanced accuracy : {block['balanced_accuracy']:.4f}",
        f"Cohen's kappa     : {block['cohen_kappa']:.4f}",
        f"MCC               : {block['mcc']:.4f}",
        f"log loss          : {block['log_loss']:.4f}",
        f"ECE ({CALIBRATION_BINS} bins)     : {block['ece']:.4f}",
    ]
    if block.get("roc_auc_ovr_macro") is not None:
        lines.append(f"ROC-AUC (OvR)     : {block['roc_auc_ovr_macro']:.4f}")
    if "top3_accuracy" in block:
        lines.append(f"top-3 accuracy    : {block['top3_accuracy']:.4f}")
    if block["classes_present"] < len(names):
        lines.append(
            f"note: only {block['classes_present']}/{len(names)} classes occur; "
            f"macro F1 over present classes = {block['macro_f1_present']:.4f}"
        )
    return "\n".join(lines) + "\n"
