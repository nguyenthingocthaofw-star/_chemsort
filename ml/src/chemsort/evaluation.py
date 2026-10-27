"""Write a complete, comparable evaluation report for one model on one split.

Layout of ``reports/experiments/<run>/<artifact>/<split>/``::

    metrics.json                  all metrics, CIs, model hash, timing
    predictions.csv               one row per image with 9- and 6-class probs
    report_9class.txt             per-class precision/recall/F1/specificity
    report_6class.txt
    confusion_matrix_9class.png   counts + row-normalised
    confusion_matrix_6class.png
    per_class_9class.png          precision/recall/F1 bars
    per_class_6class.png
    reliability_6class.png        calibration curve
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import CHEMSORT_CONTRACT
from .metrics import (
    aggregate_to_app,
    bootstrap_intervals,
    calibration_curve,
    evaluate_probabilities,
    format_classification_report,
    labels_to_app,
    normalize_outputs,
)
from .plots import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_reliability_diagram,
    save_json_ready,
)
from .predictors import Predictor, predict_paths

METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "predictions.csv"


def predictions_frame(frame: pd.DataFrame, probabilities_9: np.ndarray) -> pd.DataFrame:
    model_names = CHEMSORT_CONTRACT.model_class_names
    app_names = CHEMSORT_CONTRACT.app_class_names
    probabilities_6 = aggregate_to_app(probabilities_9)
    true_9 = frame["label_id"].to_numpy(dtype=np.int64)
    true_6 = labels_to_app(true_9)
    predicted_9 = probabilities_9.argmax(axis=1)
    predicted_6 = probabilities_6.argmax(axis=1)

    columns = [column for column in ("rel_path", "source", "split") if column in frame.columns]
    output = frame[columns].copy().reset_index(drop=True)
    output["true_9"] = [model_names[index] for index in true_9]
    output["predicted_9"] = [model_names[index] for index in predicted_9]
    output["confidence_9"] = probabilities_9.max(axis=1)
    output["correct_9"] = true_9 == predicted_9
    output["true_6"] = [app_names[index] for index in true_6]
    output["predicted_6"] = [app_names[index] for index in predicted_6]
    output["confidence_6"] = probabilities_6.max(axis=1)
    output["correct_6"] = true_6 == predicted_6
    for index, name in enumerate(model_names):
        output[f"prob_9_{name}"] = probabilities_9[:, index]
    for index, name in enumerate(app_names):
        output[f"prob_6_{name}"] = probabilities_6[:, index]
    return output


def per_source_breakdown(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    if "source" not in predictions.columns or predictions["source"].nunique() < 2:
        return {}
    return {
        source: {
            "n_samples": int(len(group)),
            "accuracy_9class": float(group["correct_9"].mean()),
            "accuracy_6class": float(group["correct_6"].mean()),
        }
        for source, group in predictions.groupby("source")
    }


def write_evaluation_report(
    output_dir: Path,
    title: str,
    frame: pd.DataFrame,
    probabilities_9: np.ndarray,
    metadata: dict[str, Any],
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Score probabilities against ``frame`` labels and write every artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities_9 = normalize_outputs(probabilities_9)
    if len(frame) != len(probabilities_9):
        raise ValueError("Frame and probabilities have different lengths")

    result = evaluate_probabilities(frame["label_id"].to_numpy(), probabilities_9)
    predictions = predictions_frame(frame, probabilities_9)
    predictions.to_csv(output_dir / PREDICTIONS_FILE, index=False, encoding="utf-8")

    true_9 = frame["label_id"].to_numpy(dtype=np.int64)
    true_6 = labels_to_app(true_9)
    probabilities_6 = aggregate_to_app(probabilities_9)
    intervals = {
        "9class": bootstrap_intervals(
            true_9, probabilities_9.argmax(axis=1), 9, samples=bootstrap_samples
        ),
        "6class": bootstrap_intervals(
            true_6, probabilities_6.argmax(axis=1), 6, samples=bootstrap_samples
        ),
    }
    calibration = calibration_curve(probabilities_6, true_6)

    for suffix, block in (("9class", result["9class"]), ("6class", result["6class"])):
        level = "9 RealWaste classes" if suffix == "9class" else "6 app groups"
        (output_dir / f"report_{suffix}.txt").write_text(
            format_classification_report(block, f"{title} - {level}"),
            encoding="utf-8",
        )
        plot_confusion_matrix(
            block["confusion_matrix"],
            block["class_names"],
            f"{title} - {level} (n={block['n_samples']})",
            output_dir / f"confusion_matrix_{suffix}.png",
        )
        plot_per_class_metrics(
            block["per_class"],
            f"{title} - per-class metrics, {level}",
            output_dir / f"per_class_{suffix}.png",
        )
    plot_reliability_diagram(
        calibration,
        result["6class"]["ece"],
        f"{title} - calibration (6 groups)",
        output_dir / "reliability_6class.png",
    )

    document = {
        **metadata,
        "title": title,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_images": int(len(frame)),
        "class_counts_9class": frame["label"].value_counts().sort_index().to_dict(),
        "summary": result["summary"],
        "confidence_intervals_95": intervals,
        "per_source": per_source_breakdown(predictions),
        "calibration_6class": calibration,
        "metrics_9class": result["9class"],
        "metrics_6class": result["6class"],
    }
    with (output_dir / METRICS_FILE).open("w", encoding="utf-8") as handle:
        json.dump(save_json_ready(document), handle, indent=2, ensure_ascii=False)
    return document


def evaluate_predictor(
    predictor: Predictor,
    frame: pd.DataFrame,
    output_dir: Path,
    title: str,
    metadata: dict[str, Any],
    batch_size: int = 32,
) -> dict[str, Any]:
    probabilities, timing = predict_paths(predictor, frame["abs_path"].tolist(), batch_size)
    return write_evaluation_report(
        output_dir,
        title,
        frame,
        probabilities,
        {
            **metadata,
            "model": predictor.description,
            "input_spec": predictor.input_spec.to_dict(),
            "timing": timing,
        },
    )


def print_summary(document: dict[str, Any]) -> None:
    summary = document["summary"]
    intervals = document["confidence_intervals_95"]
    print(f"\n{document['title']}  ({document['n_images']} images)")
    print(f"{'metric':<26}{'9 classes':>12}{'6 groups':>12}")
    for key, label in (
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro precision"),
        ("macro_recall", "Macro recall"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("cohen_kappa", "Cohen's kappa"),
        ("mcc", "MCC"),
    ):
        print(f"{label:<26}{summary[f'{key}_9class']:>12.4f}{summary[f'{key}_6class']:>12.4f}")
    for level in ("9class", "6class"):
        bounds = intervals[level]["macro_f1"]
        print(f"Macro F1 {level} 95% CI: [{bounds['lower']:.4f}, {bounds['upper']:.4f}]")
    print("Per-group F1 (6):", ", ".join(
        f"{name} {values['f1']:.3f}" for name, values in document["metrics_6class"]["per_class"].items()
    ))
