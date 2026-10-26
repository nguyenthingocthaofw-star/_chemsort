"""Export the selected ChemSort Keras model to TensorFlow Lite.

Run this module from the project root:

    python -m pipelines.legacy.export_mobilenetv2_tflite

The script creates float32 and float16 TFLite models, then checks numerical
and prediction parity against the locked Keras model on a deterministic,
class-balanced subset of validation images. It never trains the model and
never reads the test split.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from pipelines._bootstrap import configure_tensorflow_logging, ensure_local_package

ensure_local_package()
configure_tensorflow_logging(tf)

from chemsort.contracts import (
    APP_CLASS_NAMES as CONTRACT_APP_CLASS_NAMES,
    APP_GROUPS as CONTRACT_APP_GROUPS,
    MODEL_CLASS_NAMES,
    aggregate_probability_columns,
)
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/raw/RealWaste")
MANIFEST_PATH = Path("data/manifest.csv")
KERAS_MODEL_PATH = Path("models/mobilenetv2_no_aug_finetuned.keras")

OUTPUT_DIR = Path("models/tflite")
REPORT_DIR = Path("reports/tflite")

FLOAT32_MODEL_PATH = OUTPUT_DIR / "chemsort_mobilenetv2_float32.tflite"
FLOAT16_MODEL_PATH = OUTPUT_DIR / "chemsort_mobilenetv2_float16.tflite"
LABELS_PATH = OUTPUT_DIR / "labels_9.txt"
MAPPING_PATH = OUTPUT_DIR / "chemsort_mapping.json"
REPORT_PATH = REPORT_DIR / "mobilenetv2_tflite_export_report.json"

IMAGE_SIZE = (224, 224)
VERIFY_IMAGES_PER_CLASS = 10
SEED = 42

CLASS_NAMES = list(MODEL_CLASS_NAMES)
APP_CLASS_NAMES = list(CONTRACT_APP_CLASS_NAMES)
APP_GROUPS = {
    name: list(labels) for name, labels in CONTRACT_APP_GROUPS.items()
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# Utility functions
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


def size_megabytes(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def normalize_outputs(outputs: np.ndarray) -> np.ndarray:
    """Keep softmax probabilities, or apply softmax if outputs are logits."""

    outputs = np.asarray(outputs, dtype=np.float32)
    sums = outputs.sum(axis=1)
    looks_like_probabilities = (
        np.isfinite(outputs).all()
        and np.all(outputs >= -1e-6)
        and np.all(outputs <= 1.0 + 1e-6)
        and np.allclose(sums, 1.0, atol=1e-4)
    )
    if looks_like_probabilities:
        return outputs
    return tf.nn.softmax(outputs, axis=1).numpy()


class_index = {name: index for index, name in enumerate(CLASS_NAMES)}
app_index = {name: index for index, name in enumerate(APP_CLASS_NAMES)}


def aggregate_probabilities_to_app6(probabilities: np.ndarray) -> np.ndarray:
    return np.column_stack(aggregate_probability_columns(probabilities))


def metrics_from_probabilities(
    probabilities: np.ndarray,
    true_9: np.ndarray,
    true_6: np.ndarray,
) -> dict:
    predicted_9 = np.argmax(probabilities, axis=1).astype(np.int32)
    probabilities_6 = aggregate_probabilities_to_app6(probabilities)
    predicted_6 = np.argmax(probabilities_6, axis=1).astype(np.int32)

    return {
        "predicted_9": predicted_9,
        "predicted_6": predicted_6,
        "accuracy_9class": float(accuracy_score(true_9, predicted_9)),
        "macro_f1_9class": float(
            f1_score(
                true_9,
                predicted_9,
                labels=np.arange(len(CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy_6class": float(accuracy_score(true_6, predicted_6)),
        "macro_f1_6class": float(
            f1_score(
                true_6,
                predicted_6,
                labels=np.arange(len(APP_CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Deterministic validation subset
# ---------------------------------------------------------------------------


require_path(DATA_ROOT, "Dataset root")
require_path(MANIFEST_PATH, "Manifest")
require_path(KERAS_MODEL_PATH, "Selected Keras model")

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

validation_df = manifest.loc[
    manifest["split"].astype(str).str.lower() == "val"
].copy()
if validation_df.empty:
    raise SystemExit("The manifest contains no rows with split='val'.")

validation_subset_parts = []
for _, class_df in validation_df.groupby("original_label", sort=True):
    validation_subset_parts.append(
        class_df.sample(
            n=min(VERIFY_IMAGES_PER_CLASS, len(class_df)),
            random_state=SEED,
        )
    )

# pd.concat preserves ``original_label`` as a normal column across pandas
# versions, unlike GroupBy.apply whose behavior changed in newer releases.
validation_subset = (
    pd.concat(validation_subset_parts, ignore_index=True)
    .sort_values(["original_label", "path"])
    .reset_index(drop=True)
)

if sorted(validation_subset["original_label"].unique().tolist()) != CLASS_NAMES:
    raise SystemExit("The validation subset does not contain all nine classes.")

absolute_paths = [
    (DATA_ROOT / str(relative_path)).resolve()
    for relative_path in validation_subset["path"]
]
missing_images = [str(path) for path in absolute_paths if not path.is_file()]
if missing_images:
    raise SystemExit(
        f"Validation subset contains {len(missing_images)} missing image(s).\n"
        + "\n".join(missing_images[:10])
    )


def load_image(path: Path) -> np.ndarray:
    image_bytes = tf.io.read_file(str(path))
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32).numpy()


print(f"Loading {len(absolute_paths)} validation images for parity checking...")
validation_images = np.stack([load_image(path) for path in absolute_paths])
true_9 = validation_subset["original_label"].map(class_index).to_numpy(
    dtype=np.int32
)

original_to_app_name = {
    original_name: app_name
    for app_name, original_names in APP_GROUPS.items()
    for original_name in original_names
}
true_6 = (
    validation_subset["original_label"]
    .map(original_to_app_name)
    .map(app_index)
    .to_numpy(dtype=np.int32)
)


# ---------------------------------------------------------------------------
# Export float32 and float16 models
# ---------------------------------------------------------------------------


print(f"Loading selected model: {KERAS_MODEL_PATH}")
keras_model = tf.keras.models.load_model(str(KERAS_MODEL_PATH), compile=False)

if keras_model.output_shape[-1] != len(CLASS_NAMES):
    raise SystemExit(
        f"Model output has {keras_model.output_shape[-1]} classes; "
        f"expected {len(CLASS_NAMES)}."
    )

print("Exporting float32 TensorFlow Lite model...")
float32_converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
float32_bytes = float32_converter.convert()
FLOAT32_MODEL_PATH.write_bytes(float32_bytes)

print("Exporting float16-weight TensorFlow Lite model...")
float16_converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
float16_converter.optimizations = [tf.lite.Optimize.DEFAULT]
float16_converter.target_spec.supported_types = [tf.float16]
float16_bytes = float16_converter.convert()
FLOAT16_MODEL_PATH.write_bytes(float16_bytes)


# ---------------------------------------------------------------------------
# TFLite inference and parity checks
# ---------------------------------------------------------------------------


def run_tflite(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, dict]:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=max(1, min(4, os.cpu_count() or 1)),
    )
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.resize_tensor_input(
        input_details["index"],
        [1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
        strict=False,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    outputs = []
    input_scale, input_zero_point = input_details.get("quantization", (0.0, 0))
    output_scale, output_zero_point = output_details.get("quantization", (0.0, 0))

    for image in images:
        input_value = image[np.newaxis, ...]
        if np.issubdtype(input_details["dtype"], np.integer):
            if input_scale == 0:
                raise SystemExit("Quantized TFLite input has invalid scale=0.")
            input_value = np.round(input_value / input_scale + input_zero_point)
            limits = np.iinfo(input_details["dtype"])
            input_value = np.clip(input_value, limits.min, limits.max)
        input_value = input_value.astype(input_details["dtype"])

        interpreter.set_tensor(input_details["index"], input_value)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        if np.issubdtype(output_details["dtype"], np.integer):
            if output_scale == 0:
                raise SystemExit("Quantized TFLite output has invalid scale=0.")
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        outputs.append(np.asarray(output, dtype=np.float32))

    metadata = {
        "input_shape": input_details["shape"].tolist(),
        "input_dtype": np.dtype(input_details["dtype"]).name,
        "input_quantization": [float(input_scale), int(input_zero_point)],
        "output_shape": output_details["shape"].tolist(),
        "output_dtype": np.dtype(output_details["dtype"]).name,
        "output_quantization": [float(output_scale), int(output_zero_point)],
    }
    return normalize_outputs(np.stack(outputs)), metadata


print("Running Keras reference inference...")
keras_probabilities = normalize_outputs(
    keras_model.predict(validation_images, batch_size=32, verbose=1)
)
keras_metrics = metrics_from_probabilities(
    keras_probabilities,
    true_9,
    true_6,
)

print("Checking float32 TFLite parity...")
float32_probabilities, float32_io = run_tflite(
    FLOAT32_MODEL_PATH,
    validation_images,
)
float32_metrics = metrics_from_probabilities(
    float32_probabilities,
    true_9,
    true_6,
)

print("Checking float16 TFLite parity...")
float16_probabilities, float16_io = run_tflite(
    FLOAT16_MODEL_PATH,
    validation_images,
)
float16_metrics = metrics_from_probabilities(
    float16_probabilities,
    true_9,
    true_6,
)


def parity_summary(
    candidate_probabilities: np.ndarray,
    candidate_metrics: dict,
) -> dict:
    keras_predicted_9 = keras_metrics["predicted_9"]
    keras_predicted_6 = keras_metrics["predicted_6"]
    candidate_predicted_9 = candidate_metrics["predicted_9"]
    candidate_predicted_6 = candidate_metrics["predicted_6"]

    return {
        "max_absolute_probability_difference": float(
            np.max(np.abs(candidate_probabilities - keras_probabilities))
        ),
        "mean_absolute_probability_difference": float(
            np.mean(np.abs(candidate_probabilities - keras_probabilities))
        ),
        "prediction_agreement_9class": float(
            np.mean(candidate_predicted_9 == keras_predicted_9)
        ),
        "prediction_agreement_6class": float(
            np.mean(candidate_predicted_6 == keras_predicted_6)
        ),
        "accuracy_delta_9class": (
            candidate_metrics["accuracy_9class"]
            - keras_metrics["accuracy_9class"]
        ),
        "macro_f1_delta_9class": (
            candidate_metrics["macro_f1_9class"]
            - keras_metrics["macro_f1_9class"]
        ),
        "accuracy_delta_6class": (
            candidate_metrics["accuracy_6class"]
            - keras_metrics["accuracy_6class"]
        ),
        "macro_f1_delta_6class": (
            candidate_metrics["macro_f1_6class"]
            - keras_metrics["macro_f1_6class"]
        ),
    }


float32_parity = parity_summary(float32_probabilities, float32_metrics)
float16_parity = parity_summary(float16_probabilities, float16_metrics)

float16_passed = (
    float16_parity["prediction_agreement_9class"] >= 0.99
    and float16_parity["prediction_agreement_6class"] >= 0.99
    and abs(float16_parity["accuracy_delta_9class"]) <= 0.01
    and abs(float16_parity["accuracy_delta_6class"]) <= 0.01
    and abs(float16_parity["macro_f1_delta_9class"]) <= 0.01
    and abs(float16_parity["macro_f1_delta_6class"]) <= 0.01
)

recommended_path = FLOAT16_MODEL_PATH if float16_passed else FLOAT32_MODEL_PATH


# ---------------------------------------------------------------------------
# Mobile contract and report
# ---------------------------------------------------------------------------


LABELS_PATH.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

mobile_contract = {
    "input": {
        "shape": [1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
        "layout": "NHWC",
        "color_order": "RGB",
        "dtype": "float32",
        "pixel_range": [0.0, 255.0],
        "resize": "224x224",
        "note": "Do not divide pixels by 255 in the app; preprocessing is part of the trained model contract.",
    },
    "model_output_9class": {
        "classes_in_index_order": CLASS_NAMES,
        "probability_aggregation_required": True,
    },
    "app_output_6class": {
        "classes_in_index_order": APP_CLASS_NAMES,
        "groups": APP_GROUPS,
        "rule": "Sum 9-class probabilities within each group, then take argmax.",
    },
}
with MAPPING_PATH.open("w", encoding="utf-8") as file:
    json.dump(mobile_contract, file, indent=2, ensure_ascii=False)

report = {
    "status": "tflite_export_completed",
    "training_performed": False,
    "test_split_used": False,
    "source_model": {
        "path": str(KERAS_MODEL_PATH),
        "sha256": sha256_file(KERAS_MODEL_PATH),
        "size_mb": size_megabytes(KERAS_MODEL_PATH),
    },
    "validation_parity_subset": {
        "image_count": len(validation_subset),
        "images_per_original_class": VERIFY_IMAGES_PER_CLASS,
        "seed": SEED,
    },
    "keras_reference_metrics": keras_metrics,
    "float32": {
        "path": str(FLOAT32_MODEL_PATH),
        "sha256": sha256_file(FLOAT32_MODEL_PATH),
        "size_mb": size_megabytes(FLOAT32_MODEL_PATH),
        "io": float32_io,
        "metrics": float32_metrics,
        "parity": float32_parity,
    },
    "float16": {
        "path": str(FLOAT16_MODEL_PATH),
        "sha256": sha256_file(FLOAT16_MODEL_PATH),
        "size_mb": size_megabytes(FLOAT16_MODEL_PATH),
        "io": float16_io,
        "metrics": float16_metrics,
        "parity": float16_parity,
        "passed_mobile_selection_gate": float16_passed,
    },
    "recommended_mobile_model": str(recommended_path),
    "mobile_contract_path": str(MAPPING_PATH),
    "labels_path": str(LABELS_PATH),
}

# Remove prediction arrays from the persisted metric blocks.
for metrics_block in (
    report["keras_reference_metrics"],
    report["float32"]["metrics"],
    report["float16"]["metrics"],
):
    metrics_block.pop("predicted_9", None)
    metrics_block.pop("predicted_6", None)

with REPORT_PATH.open("w", encoding="utf-8") as file:
    json.dump(json_safe(report), file, indent=2, ensure_ascii=False)


print("\nTFLITE EXPORT RESULTS")
print(f"{'Model':<12} {'Size (MB)':>11} {'Agree 9':>10} {'Agree 6':>10} {'Max |diff|':>12}")
print("-" * 60)
print(
    f"{'float32':<12} {size_megabytes(FLOAT32_MODEL_PATH):>11.2f} "
    f"{float32_parity['prediction_agreement_9class']:>10.4f} "
    f"{float32_parity['prediction_agreement_6class']:>10.4f} "
    f"{float32_parity['max_absolute_probability_difference']:>12.6f}"
)
print(
    f"{'float16':<12} {size_megabytes(FLOAT16_MODEL_PATH):>11.2f} "
    f"{float16_parity['prediction_agreement_9class']:>10.4f} "
    f"{float16_parity['prediction_agreement_6class']:>10.4f} "
    f"{float16_parity['max_absolute_probability_difference']:>12.6f}"
)
print(f"\nRecommended mobile model: {recommended_path}")
print(f"Float16 selection gate:    {'PASS' if float16_passed else 'FAIL'}")
print(f"Mobile mapping:            {MAPPING_PATH}")
print(f"Export report:             {REPORT_PATH}")
print("Training performed:        False")
print("Test split used:           False")

'''
Xuất TFLite đã thành công. Nên chọn bản **Float32** cho phiên bản đầu tiên.

* Float32: `8.52 MB`, đồng thuận với Keras `100%` ở cả 9 và 6 lớp.
* Sai khác xác suất tối đa chỉ `0.000009`.
* Float16 nhỏ hơn (`4.33 MB`) nhưng lệch 1/90 dự đoán ở cấp 6 lớp, nên không vượt cổng kiểm tra bảo thủ.
* Các cảnh báo `TensorSpec`, `Interpreter deprecated` và `Ignored output_format` không phải lỗi.

Các file dùng cho ứng dụng:

```text
models\tflite\chemsort_mobilenetv2_float32.tflite
models\tflite\labels_9.txt
models\tflite\chemsort_mapping.json
```

Bước kế tiếp là **tích hợp model vào Flutter**:

1. Thêm ba file trên vào assets.
2. Nạp model bằng TFLite interpreter.
3. Resize ảnh thành `224×224`, RGB, `float32`, phạm vi `0–255`.
4. Không chia pixel cho `255`.
5. Nhận 9 xác suất rồi cộng xác suất để chuyển thành 6 lớp ChemSort.
6. So sánh kết quả Flutter với Python bằng cùng một ảnh.
7. Đo thời gian suy luận trên điện thoại thật.

Nếu đã có project Flutter, hãy gửi `pubspec.yaml` và cấu trúc thư mục `lib`; nếu chưa có, bước tiếp theo sẽ là tạo project Flutter từ đầu.
'''
