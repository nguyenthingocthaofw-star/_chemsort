"""Final, one-time test evaluation for the selected ChemSort model.

Run this module once from the project root after locking the model:

    python -m pipelines.legacy.evaluate_mobilenetv2_test

This script never trains or changes the model. It evaluates only the locked
``test`` split, reports both the original 9 classes and the 6 ChemSort app
classes, and writes reproducible reports to
``reports/final_test/mobilenetv2_no_aug_finetuned_test/``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from pipelines._bootstrap import configure_tensorflow_logging, ensure_local_package

ensure_local_package()
configure_tensorflow_logging(tf)

from chemsort.contracts import APP_CLASS_NAMES as CONTRACT_APP_CLASS_NAMES
from chemsort.contracts import (
    MODEL_CLASS_NAMES,
    aggregate_probability_columns,
    model_to_app_ids,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/raw/RealWaste")
MANIFEST_PATH = Path("data/manifest.csv")
MODEL_PATH = Path("models/mobilenetv2_no_aug_finetuned.keras")
VALIDATION_COMPARISON_PATH = Path(
    "reports/validation/mobilenetv2_no_aug_finetuned/"
    "mobilenetv2_no_aug_finetuned_comparison.json"
)

RUN_NAME = "mobilenetv2_no_aug_finetuned_test"
REPORT_DIR = Path("reports/final_test") / RUN_NAME

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
BOOTSTRAP_SAMPLES = 2000
SEED = 42

CLASS_NAMES = list(MODEL_CLASS_NAMES)
APP_CLASS_NAMES = list(CONTRACT_APP_CLASS_NAMES)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise SystemExit(f"{description} not found: {path.resolve()}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


require_path(DATA_ROOT, "Dataset root")
require_path(MANIFEST_PATH, "Manifest")
require_path(MODEL_PATH, "Selected fine-tuned model")

manifest = pd.read_csv(MANIFEST_PATH)
required_columns = {"path", "original_label", "split"}
missing_columns = required_columns - set(manifest.columns)
if missing_columns:
    raise SystemExit(
        "manifest.csv is missing columns: " + ", ".join(sorted(missing_columns))
    )

found_classes = sorted(manifest["original_label"].dropna().unique().tolist())
if found_classes != CLASS_NAMES:
    raise SystemExit(
        "Unexpected class names.\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found:    {found_classes}"
    )

test_df = (
    manifest.loc[manifest["split"].astype(str).str.lower() == "test"]
    .copy()
    .reset_index(drop=True)
)
if test_df.empty:
    raise SystemExit("The manifest contains no rows with split='test'.")

label_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
test_df["label_id"] = test_df["original_label"].map(label_to_id)
if test_df["label_id"].isna().any():
    raise SystemExit("Some test labels could not be converted to label IDs.")

test_df["absolute_path"] = test_df["path"].map(
    lambda relative_path: str((DATA_ROOT / str(relative_path)).resolve())
)
missing_files = [path for path in test_df["absolute_path"] if not Path(path).is_file()]
if missing_files:
    preview = "\n".join(missing_files[:10])
    raise SystemExit(
        f"The test split references {len(missing_files)} missing image(s).\n{preview}"
    )


# ---------------------------------------------------------------------------
# Test data pipeline: deterministic and without augmentation
# ---------------------------------------------------------------------------


def load_image(path: tf.Tensor, label: tf.Tensor):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


test_paths = test_df["absolute_path"].to_numpy(dtype=str)
true_9 = test_df["label_id"].to_numpy(dtype=np.int32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, true_9))
test_dataset = test_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Nine-class to six-class mapping
# ---------------------------------------------------------------------------

original_name_to_app_id = dict(zip(CLASS_NAMES, model_to_app_ids()))

true_6 = test_df["original_label"].map(original_name_to_app_id).to_numpy(
    dtype=np.int32
)


def aggregate_probabilities_to_app6(probabilities: np.ndarray) -> np.ndarray:
    """Aggregate probabilities, then choose the 6-class prediction."""

    return np.column_stack(aggregate_probability_columns(probabilities))


# ---------------------------------------------------------------------------
# Metrics, confidence intervals and report writers
# ---------------------------------------------------------------------------


def bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
) -> dict[str, float | int]:
    """Return a deterministic non-parametric 95% bootstrap interval."""

    rng = np.random.default_rng(SEED)
    sample_count = len(y_true)
    scores = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)

    if metric_name == "accuracy":
        metric = accuracy_score
    elif metric_name == "macro_f1":
        labels = np.arange(len(np.unique(y_true)))

        def metric(actual, predicted):
            return f1_score(
                actual,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )

    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    for index in range(BOOTSTRAP_SAMPLES):
        sampled_indices = rng.integers(0, sample_count, size=sample_count)
        scores[index] = metric(y_true[sampled_indices], y_pred[sampled_indices])

    return {
        "lower_95": float(np.percentile(scores, 2.5)),
        "upper_95": float(np.percentile(scores, 97.5)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    display_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    width = 12 if len(display_names) == 9 else 10
    figure, axis = plt.subplots(figsize=(width, width - 1))
    display = ConfusionMatrixDisplay(matrix, display_labels=display_names)
    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format="d",
        xticks_rotation=45,
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# Locked-model inference
# ---------------------------------------------------------------------------

print("Final test evaluation (model is not trained or modified):")
print(f"  Model:       {MODEL_PATH}")
print(f"  Test images: {len(test_df)}")
print(f"  Test split:  {sorted(test_df['split'].astype(str).unique().tolist())}")
print("  Augmentation: OFF")

model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
if model.output_shape[-1] != len(CLASS_NAMES):
    raise SystemExit(
        f"Model output has {model.output_shape[-1]} classes; "
        f"expected {len(CLASS_NAMES)}."
    )

raw_outputs = np.asarray(model.predict(test_dataset, verbose=1))
if raw_outputs.shape != (len(test_df), len(CLASS_NAMES)):
    raise SystemExit(
        f"Unexpected prediction shape {raw_outputs.shape}; "
        f"expected {(len(test_df), len(CLASS_NAMES))}."
    )
if not np.isfinite(raw_outputs).all():
    raise SystemExit("The model produced NaN or infinite values.")

row_sums = raw_outputs.sum(axis=1)
looks_like_probabilities = (
    np.all(raw_outputs >= -1e-6)
    and np.all(raw_outputs <= 1.0 + 1e-6)
    and np.allclose(row_sums, 1.0, atol=1e-4)
)
probabilities_9 = (
    raw_outputs
    if looks_like_probabilities
    else tf.nn.softmax(raw_outputs, axis=1).numpy()
)

predicted_9 = np.argmax(probabilities_9, axis=1).astype(np.int32)
probabilities_6 = aggregate_probabilities_to_app6(probabilities_9)
predicted_6 = np.argmax(probabilities_6, axis=1).astype(np.int32)

labels_9 = np.arange(len(CLASS_NAMES))
labels_6 = np.arange(len(APP_CLASS_NAMES))

report_9_dict = classification_report(
    true_9,
    predicted_9,
    labels=labels_9,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)
report_9_text = classification_report(
    true_9,
    predicted_9,
    labels=labels_9,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

report_6_dict = classification_report(
    true_6,
    predicted_6,
    labels=labels_6,
    target_names=APP_CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)
report_6_text = classification_report(
    true_6,
    predicted_6,
    labels=labels_6,
    target_names=APP_CLASS_NAMES,
    digits=4,
    zero_division=0,
)

metrics = {
    "accuracy_9class": float(accuracy_score(true_9, predicted_9)),
    "macro_f1_9class": float(
        f1_score(
            true_9,
            predicted_9,
            labels=labels_9,
            average="macro",
            zero_division=0,
        )
    ),
    "accuracy_6class": float(accuracy_score(true_6, predicted_6)),
    "macro_f1_6class": float(
        f1_score(
            true_6,
            predicted_6,
            labels=labels_6,
            average="macro",
            zero_division=0,
        )
    ),
    "organic_precision_6class": float(report_6_dict["Organic"]["precision"]),
    "organic_recall_6class": float(report_6_dict["Organic"]["recall"]),
    "plastic_precision_6class": float(report_6_dict["Plastic"]["precision"]),
    "plastic_recall_6class": float(report_6_dict["Plastic"]["recall"]),
}

confidence_intervals = {
    "accuracy_9class": bootstrap_interval(true_9, predicted_9, "accuracy"),
    "macro_f1_9class": bootstrap_interval(true_9, predicted_9, "macro_f1"),
    "accuracy_6class": bootstrap_interval(true_6, predicted_6, "accuracy"),
    "macro_f1_6class": bootstrap_interval(true_6, predicted_6, "macro_f1"),
}


# ---------------------------------------------------------------------------
# Save complete, auditable outputs
# ---------------------------------------------------------------------------

predictions = test_df[["path", "original_label", "split"]].copy()
predictions["true_9"] = [CLASS_NAMES[index] for index in true_9]
predictions["predicted_9"] = [CLASS_NAMES[index] for index in predicted_9]
predictions["confidence_9"] = probabilities_9.max(axis=1)
predictions["correct_9"] = true_9 == predicted_9
predictions["true_6"] = [APP_CLASS_NAMES[index] for index in true_6]
predictions["predicted_6"] = [APP_CLASS_NAMES[index] for index in predicted_6]
predictions["confidence_6"] = probabilities_6.max(axis=1)
predictions["correct_6"] = true_6 == predicted_6

for index, name in enumerate(CLASS_NAMES):
    predictions[f"prob_9_{name}"] = probabilities_9[:, index]
for index, name in enumerate(APP_CLASS_NAMES):
    predictions[f"prob_6_{name}"] = probabilities_6[:, index]

predictions_path = REPORT_DIR / f"{RUN_NAME}_predictions.csv"
predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

(REPORT_DIR / f"{RUN_NAME}_9class_report.txt").write_text(
    report_9_text,
    encoding="utf-8",
)
(REPORT_DIR / f"{RUN_NAME}_6class_report.txt").write_text(
    report_6_text,
    encoding="utf-8",
)

save_confusion_matrix(
    true_9,
    predicted_9,
    labels_9,
    CLASS_NAMES,
    "MobileNetV2 fine-tuned - Final test (9 classes)",
    REPORT_DIR / f"{RUN_NAME}_9class_confusion_matrix.png",
)
save_confusion_matrix(
    true_6,
    predicted_6,
    labels_6,
    APP_CLASS_NAMES,
    "MobileNetV2 fine-tuned - Final test (6 app classes)",
    REPORT_DIR / f"{RUN_NAME}_6class_confusion_matrix.png",
)

validation_reference = None
generalization_gap_test_minus_validation = None
if VALIDATION_COMPARISON_PATH.exists():
    with VALIDATION_COMPARISON_PATH.open("r", encoding="utf-8") as file:
        comparison = json.load(file)
    validation_reference = comparison.get("finetuned")
    if validation_reference:
        generalization_gap_test_minus_validation = {
            key: metrics[key] - float(validation_reference[key])
            for key in (
                "accuracy_9class",
                "macro_f1_9class",
                "accuracy_6class",
                "macro_f1_6class",
                "organic_precision_6class",
                "organic_recall_6class",
                "plastic_precision_6class",
                "plastic_recall_6class",
            )
            if key in validation_reference
        }

summary = {
    "status": "final_test_completed",
    "model_path": str(MODEL_PATH),
    "model_sha256": sha256_file(MODEL_PATH),
    "manifest_path": str(MANIFEST_PATH),
    "test_image_count": len(test_df),
    "test_class_counts_9class": test_df["original_label"].value_counts().to_dict(),
    "metrics": metrics,
    "confidence_intervals_95": confidence_intervals,
    "classification_report_9class": report_9_dict,
    "classification_report_6class": report_6_dict,
    "selected_validation_metrics": validation_reference,
    "generalization_gap_test_minus_validation": (
        generalization_gap_test_minus_validation
    ),
    "test_set_used": True,
    "training_performed": False,
    "seed": SEED,
}

summary_path = REPORT_DIR / f"{RUN_NAME}_summary.json"
with summary_path.open("w", encoding="utf-8") as file:
    json.dump(make_json_safe(summary), file, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

print("\nFINAL TEST RESULTS")
print(f"{'Metric':<30} {'Score':>10} {'95% CI':>23}")
print("-" * 67)

display_rows = [
    ("9-class Accuracy", "accuracy_9class"),
    ("9-class Macro-F1", "macro_f1_9class"),
    ("6-class Accuracy", "accuracy_6class"),
    ("6-class Macro-F1", "macro_f1_6class"),
]
for label, key in display_rows:
    interval = confidence_intervals[key]
    interval_text = f"[{interval['lower_95']:.4f}, {interval['upper_95']:.4f}]"
    print(f"{label:<30} {metrics[key]:>10.4f} {interval_text:>23}")

print("\nTarget-class metrics:")
print(f"  Organic precision:  {metrics['organic_precision_6class']:.4f}")
print(f"  Organic recall:     {metrics['organic_recall_6class']:.4f}")
print(f"  Plastic precision:  {metrics['plastic_precision_6class']:.4f}")
print(f"  Plastic recall:     {metrics['plastic_recall_6class']:.4f}")

if validation_reference:
    print("\nValidation versus final test:")
    print(f"{'Metric':<30} {'Validation':>12} {'Test':>10} {'Gap':>10}")
    print("-" * 66)
    for label, key in display_rows:
        validation_score = float(validation_reference[key])
        test_score = metrics[key]
        print(
            f"{label:<30} {validation_score:>12.4f} "
            f"{test_score:>10.4f} {test_score - validation_score:>+10.4f}"
        )

print(f"\nSummary:     {summary_path}")
print(f"Predictions: {predictions_path}")
print("Training performed: False")
print("Test set used: True")
print(
    "Important: this test result is the final unbiased estimate for this model. "
    "Do not tune the model using individual test errors."
)
