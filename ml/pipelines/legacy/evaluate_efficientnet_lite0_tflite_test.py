"""One-time final-test evaluation of the locked ChemSort TFLite artifact.

Run this module once from the project root after locking the TFLite artifact:

    python -m pipelines.legacy.evaluate_efficientnet_lite0_tflite_test

The script evaluates the exact deployment artifact
``models/tflite/efficientnet_lite0_float16.tflite``. It never trains or
changes the model, never uses augmentation, and only reads rows whose split is
``test``. Reports are written to
``reports/final_test/efficientnet_lite0_float16_tflite_test/``.
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
# Locked evaluation contract
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/raw/RealWaste")
MANIFEST_PATH = Path("data/manifest.csv")
MODEL_PATH = Path("models/tflite/efficientnet_lite0_float16.tflite")
EXPORT_REPORT_PATH = Path(
    "reports/tflite/efficientnet_lite0_tflite_export_report.json"
)
VALIDATION_METRICS_PATH = Path(
    "reports/validation/efficientnet_lite0_no_aug/"
    "efficientnet_lite0_no_aug_metrics.json"
)
LABELS_PATH = Path("models/tflite/labels_9.txt")
MAPPING_PATH = Path("models/tflite/chemsort_mapping.json")

BASE_RUN_NAME = "efficientnet_lite0_float16_tflite_test"
RUN_NAME = BASE_RUN_NAME
REPORT_DIR = Path("reports/final_test") / RUN_NAME
SUMMARY_PATH = REPORT_DIR / f"{RUN_NAME}_summary.json"

IMAGE_SIZE = (224, 224)
BOOTSTRAP_SAMPLES = 2000
SEED = 42
INTERPRETER_THREADS = max(1, min(4, os.cpu_count() or 1))

CLASS_NAMES = list(MODEL_CLASS_NAMES)
APP_CLASS_NAMES = list(CONTRACT_APP_CLASS_NAMES)
ORIGINAL_NAME_TO_APP_ID = dict(zip(CLASS_NAMES, model_to_app_ids()))

tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# Validation and integrity checks
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


for required_path, description in (
    (DATA_ROOT, "Dataset root"),
    (MANIFEST_PATH, "Manifest"),
    (MODEL_PATH, "Locked TFLite model"),
    (EXPORT_REPORT_PATH, "TFLite export report"),
    (VALIDATION_METRICS_PATH, "Validation metrics"),
    (LABELS_PATH, "Nine-class labels file"),
    (MAPPING_PATH, "ChemSort mapping file"),
):
    require_path(required_path, description)

actual_model_sha256 = sha256_file(MODEL_PATH)


def read_report_model_sha256(path: Path) -> str:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot read existing final-test report {path}: {error}")
    return str(report.get("model_sha256", "")).lower()


if SUMMARY_PATH.exists():
    existing_model_sha256 = read_report_model_sha256(SUMMARY_PATH)
    if existing_model_sha256 == actual_model_sha256:
        raise SystemExit(
            f"This model artifact was already final-tested: {SUMMARY_PATH}\n"
            "Do not rerun or tune against the test split. Use the existing report."
        )

    RUN_NAME = f"{BASE_RUN_NAME}_{actual_model_sha256[:12]}"
    REPORT_DIR = Path("reports/final_test") / RUN_NAME
    SUMMARY_PATH = REPORT_DIR / f"{RUN_NAME}_summary.json"
    if SUMMARY_PATH.exists():
        versioned_model_sha256 = read_report_model_sha256(SUMMARY_PATH)
        if versioned_model_sha256 == actual_model_sha256:
            raise SystemExit(
                f"This model artifact was already final-tested: {SUMMARY_PATH}\n"
                "Do not rerun or tune against the test split. Use the existing report."
            )
        raise SystemExit(
            f"Final-test path collision for model {actual_model_sha256}: "
            f"{SUMMARY_PATH}"
        )

REPORT_DIR.mkdir(parents=True, exist_ok=True)

labels_from_file = [
    line.strip()
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if labels_from_file != CLASS_NAMES:
    raise SystemExit(
        "labels_9.txt does not match the locked class order.\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found:    {labels_from_file}"
    )

with MAPPING_PATH.open("r", encoding="utf-8") as file:
    mobile_contract = json.load(file)
mapped_app_classes = mobile_contract.get("app_output_6class", {}).get(
    "classes_in_index_order"
)
if mapped_app_classes != APP_CLASS_NAMES:
    raise SystemExit(
        "chemsort_mapping.json does not match the locked 6-class order.\n"
        f"Expected: {APP_CLASS_NAMES}\n"
        f"Found:    {mapped_app_classes}"
    )

with EXPORT_REPORT_PATH.open("r", encoding="utf-8") as file:
    export_report = json.load(file)

float16_export = export_report.get("float16", {})
if not bool(float16_export.get("passed_gate")):
    raise SystemExit(
        "The export report does not mark EfficientNet-Lite0 float16 as PASS."
    )
if export_report.get("recommended_format") != "float16":
    raise SystemExit(
        "The export report does not recommend the float16 artifact."
    )
expected_model_sha256 = str(float16_export.get("sha256", ""))
if not expected_model_sha256 or actual_model_sha256 != expected_model_sha256:
    raise SystemExit(
        "TFLite SHA-256 does not match the passed export report.\n"
        f"Expected: {expected_model_sha256 or '<missing>'}\n"
        f"Actual:   {actual_model_sha256}\n"
        "Do not evaluate an unverified or modified artifact."
    )

with VALIDATION_METRICS_PATH.open("r", encoding="utf-8") as file:
    validation_metrics_file = json.load(file)
validation_reference = validation_metrics_file.get("validation")
if not isinstance(validation_reference, dict):
    raise SystemExit(
        "Validation metrics file has no valid 'validation' metrics block."
    )


# ---------------------------------------------------------------------------
# Locked test split
# ---------------------------------------------------------------------------

manifest = pd.read_csv(MANIFEST_PATH)
required_columns = {"path", "original_label", "split"}
missing_columns = required_columns - set(manifest.columns)
if missing_columns:
    raise SystemExit(
        "manifest.csv is missing columns: "
        + ", ".join(sorted(missing_columns))
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
missing_files = [
    path for path in test_df["absolute_path"] if not Path(path).is_file()
]
if missing_files:
    preview = "\n".join(missing_files[:10])
    raise SystemExit(
        f"The test split references {len(missing_files)} missing image(s).\n"
        f"{preview}"
    )

true_9 = test_df["label_id"].to_numpy(dtype=np.int32)
true_6 = test_df["original_label"].map(ORIGINAL_NAME_TO_APP_ID).to_numpy(
    dtype=np.int32
)
def load_image(path: str) -> np.ndarray:
    """Match the deterministic resize and raw 0..255 input contract."""

    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32).numpy()


def normalize_outputs(outputs: np.ndarray) -> np.ndarray:
    outputs = np.asarray(outputs, dtype=np.float32)
    expected_shape = (len(test_df), len(CLASS_NAMES))
    if outputs.shape != expected_shape:
        raise SystemExit(
            f"Unexpected prediction shape {outputs.shape}; "
            f"expected {expected_shape}."
        )
    if not np.isfinite(outputs).all():
        invalid_count = int((~np.isfinite(outputs)).sum())
        raise SystemExit(
            f"TFLite produced {invalid_count} NaN/Inf output value(s)."
        )

    row_sums = outputs.sum(axis=1)
    looks_like_probabilities = (
        np.all(outputs >= -1e-6)
        and np.all(outputs <= 1.0 + 1e-6)
        and np.allclose(row_sums, 1.0, atol=1e-4)
    )
    probabilities = (
        outputs
        if looks_like_probabilities
        else tf.nn.softmax(outputs, axis=1).numpy()
    )
    if not np.isfinite(probabilities).all():
        raise SystemExit("Normalized TFLite probabilities contain NaN/Inf.")
    return probabilities


def aggregate_probabilities_to_app6(
    probabilities: np.ndarray,
) -> np.ndarray:
    return np.column_stack(aggregate_probability_columns(probabilities))


# ---------------------------------------------------------------------------
# TFLite inference on the exact deployment artifact
# ---------------------------------------------------------------------------

print("Final test evaluation (model is not trained or modified):")
print(f"  Model:        {MODEL_PATH}")
print(f"  SHA-256:      {actual_model_sha256}")
print(f"  Test images:  {len(test_df)}")
print(f"  Test split:   {sorted(test_df['split'].astype(str).unique().tolist())}")
print("  Augmentation: OFF")
print(f"  CPU threads:  {INTERPRETER_THREADS}")

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH),
    num_threads=INTERPRETER_THREADS,
)
input_detail = interpreter.get_input_details()[0]
interpreter.resize_tensor_input(
    input_detail["index"],
    [1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
    strict=False,
)
interpreter.allocate_tensors()
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]

expected_input_shape = [1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
if input_detail["shape"].tolist() != expected_input_shape:
    raise SystemExit(
        f"Unexpected TFLite input shape {input_detail['shape'].tolist()}; "
        f"expected {expected_input_shape}."
    )
if int(np.prod(output_detail["shape"])) != len(CLASS_NAMES):
    raise SystemExit(
        f"Unexpected TFLite output shape {output_detail['shape'].tolist()}; "
        f"expected 9 output values."
    )

input_scale, input_zero = input_detail.get("quantization", (0.0, 0))
output_scale, output_zero = output_detail.get("quantization", (0.0, 0))
raw_outputs = np.empty(
    (len(test_df), len(CLASS_NAMES)),
    dtype=np.float32,
)

print("Running locked TFLite inference...")
for row_index, image_path in enumerate(test_df["absolute_path"]):
    value = load_image(image_path)[np.newaxis, ...]
    if np.issubdtype(input_detail["dtype"], np.integer):
        if input_scale == 0:
            raise SystemExit("Quantized TFLite input has invalid scale=0.")
        value = np.round(value / input_scale + input_zero)
        limits = np.iinfo(input_detail["dtype"])
        value = np.clip(value, limits.min, limits.max)
    value = value.astype(input_detail["dtype"])

    interpreter.set_tensor(input_detail["index"], value)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"]).reshape(-1)
    if np.issubdtype(output_detail["dtype"], np.integer):
        if output_scale == 0:
            raise SystemExit("Quantized TFLite output has invalid scale=0.")
        output = (
            output.astype(np.float32) - output_zero
        ) * output_scale
    raw_outputs[row_index] = np.asarray(output, dtype=np.float32)

    completed = row_index + 1
    if completed % 100 == 0 or completed == len(test_df):
        print(f"  Processed {completed}/{len(test_df)} images")

probabilities_9 = normalize_outputs(raw_outputs)
predicted_9 = np.argmax(probabilities_9, axis=1).astype(np.int32)
probabilities_6 = aggregate_probabilities_to_app6(probabilities_9)
predicted_6 = np.argmax(probabilities_6, axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# Metrics and confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    label_count: int,
) -> dict[str, float | int]:
    """Return a deterministic non-parametric 95% bootstrap interval."""

    rng = np.random.default_rng(SEED)
    sample_count = len(y_true)
    scores = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)

    for sample_index in range(BOOTSTRAP_SAMPLES):
        sampled_indices = rng.integers(0, sample_count, size=sample_count)
        sampled_true = y_true[sampled_indices]
        sampled_predicted = y_pred[sampled_indices]
        if metric_name == "accuracy":
            score = accuracy_score(sampled_true, sampled_predicted)
        elif metric_name == "macro_f1":
            score = f1_score(
                sampled_true,
                sampled_predicted,
                labels=np.arange(label_count),
                average="macro",
                zero_division=0,
            )
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
        scores[sample_index] = score

    return {
        "lower_95": float(np.percentile(scores, 2.5)),
        "upper_95": float(np.percentile(scores, 97.5)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
    }


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
    "accuracy_9class": bootstrap_interval(
        true_9, predicted_9, "accuracy", len(CLASS_NAMES)
    ),
    "macro_f1_9class": bootstrap_interval(
        true_9, predicted_9, "macro_f1", len(CLASS_NAMES)
    ),
    "accuracy_6class": bootstrap_interval(
        true_6, predicted_6, "accuracy", len(APP_CLASS_NAMES)
    ),
    "macro_f1_6class": bootstrap_interval(
        true_6, predicted_6, "macro_f1", len(APP_CLASS_NAMES)
    ),
}


# ---------------------------------------------------------------------------
# Auditable reports
# ---------------------------------------------------------------------------


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


predictions = test_df[["path", "original_label", "split"]].copy()
predictions["true_9"] = [CLASS_NAMES[index] for index in true_9]
predictions["predicted_9"] = [CLASS_NAMES[index] for index in predicted_9]
predictions["confidence_9"] = probabilities_9.max(axis=1)
predictions["correct_9"] = true_9 == predicted_9
predictions["true_6"] = [APP_CLASS_NAMES[index] for index in true_6]
predictions["predicted_6"] = [APP_CLASS_NAMES[index] for index in predicted_6]
predictions["confidence_6"] = probabilities_6.max(axis=1)
predictions["correct_6"] = true_6 == predicted_6

for class_id, class_name in enumerate(CLASS_NAMES):
    predictions[f"prob_9_{class_name}"] = probabilities_9[:, class_id]
for class_id, class_name in enumerate(APP_CLASS_NAMES):
    predictions[f"prob_6_{class_name}"] = probabilities_6[:, class_id]

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
    "EfficientNet-Lite0 float16 TFLite - Final test (9 classes)",
    REPORT_DIR / f"{RUN_NAME}_9class_confusion_matrix.png",
)
save_confusion_matrix(
    true_6,
    predicted_6,
    labels_6,
    APP_CLASS_NAMES,
    "EfficientNet-Lite0 float16 TFLite - Final test (6 app classes)",
    REPORT_DIR / f"{RUN_NAME}_6class_confusion_matrix.png",
)

gap_metric_keys = (
    "accuracy_9class",
    "macro_f1_9class",
    "accuracy_6class",
    "macro_f1_6class",
    "organic_precision_6class",
    "organic_recall_6class",
    "plastic_precision_6class",
    "plastic_recall_6class",
)
generalization_gap_test_minus_validation = {
    key: metrics[key] - float(validation_reference[key])
    for key in gap_metric_keys
    if key in validation_reference
}

summary = {
    "status": "final_test_completed",
    "candidate": "EfficientNet-Lite0 float16 TFLite",
    "model_path": str(MODEL_PATH),
    "model_sha256": actual_model_sha256,
    "model_size_mb": MODEL_PATH.stat().st_size / (1024 * 1024),
    "source_export_report": str(EXPORT_REPORT_PATH),
    "export_parity": float16_export.get("parity"),
    "manifest_path": str(MANIFEST_PATH),
    "test_image_count": len(test_df),
    "test_class_counts_9class": test_df[
        "original_label"
    ].value_counts().to_dict(),
    "tflite_io": {
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_shape": output_detail["shape"].tolist(),
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "interpreter_threads": INTERPRETER_THREADS,
    },
    "metrics": metrics,
    "confidence_intervals_95": confidence_intervals,
    "classification_report_9class": report_9_dict,
    "classification_report_6class": report_6_dict,
    "selected_validation_metrics": validation_reference,
    "generalization_gap_test_minus_validation": (
        generalization_gap_test_minus_validation
    ),
    "input_contract": "RGB float32, 224x224, raw 0..255; preprocessing embedded",
    "test_set_used": True,
    "training_performed": False,
    "augmentation_used": False,
    "seed": SEED,
}

with SUMMARY_PATH.open("w", encoding="utf-8") as file:
    json.dump(make_json_safe(summary), file, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

print("\nFINAL TFLITE TEST RESULTS")
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
    interval_text = (
        f"[{interval['lower_95']:.4f}, {interval['upper_95']:.4f}]"
    )
    print(f"{label:<30} {metrics[key]:>10.4f} {interval_text:>23}")

print("\nTarget-class metrics:")
print(f"  Organic precision:  {metrics['organic_precision_6class']:.4f}")
print(f"  Organic recall:     {metrics['organic_recall_6class']:.4f}")
print(f"  Plastic precision:  {metrics['plastic_precision_6class']:.4f}")
print(f"  Plastic recall:     {metrics['plastic_recall_6class']:.4f}")

print("\nValidation versus final TFLite test:")
print(f"{'Metric':<30} {'Validation':>12} {'Test':>10} {'Gap':>10}")
print("-" * 66)
for label, key in display_rows:
    validation_score = float(validation_reference[key])
    test_score = metrics[key]
    print(
        f"{label:<30} {validation_score:>12.4f} "
        f"{test_score:>10.4f} {test_score - validation_score:>+10.4f}"
    )

print(f"\nSummary:     {SUMMARY_PATH}")
print(f"Predictions: {predictions_path}")
print("Training performed: False")
print("Test set used: True")
print(
    "Important: this is the final estimate for the locked deployment "
    "artifact. Do not tune the model using individual test errors."
)
