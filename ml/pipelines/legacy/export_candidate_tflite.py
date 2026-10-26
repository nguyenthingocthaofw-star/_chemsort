"""Export and verify ChemSort MobileNetV3Large or EfficientNet-Lite0.

Run this file from the ChemSort project root, one candidate per process:

    python -m pipelines.legacy.export_candidate_tflite --model mobilenetv3large
    python -m pipelines.legacy.export_candidate_tflite --model efficientnet_lite0

The script rebuilds the selected architecture, loads its winning validation
checkpoint, exports float32 and float16 TFLite files, and checks parity on a
deterministic class-balanced validation subset. It never reads the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


# EfficientNet-Lite0 was trained through the legacy TF Hub Keras layer, so its
# compatibility mode must be selected before TensorFlow is imported.
parser = argparse.ArgumentParser(
    description="Export a trained ChemSort candidate to TensorFlow Lite."
)
parser.add_argument(
    "--model",
    choices=("mobilenetv3large", "efficientnet_lite0"),
    required=True,
)
args = parser.parse_args()

if args.model == "efficientnet_lite0":
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
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
from chemsort.tfhub_compat import import_tensorflow_hub
from sklearn.metrics import accuracy_score, f1_score

hub = None
if args.model == "efficientnet_lite0":
    try:
        hub = import_tensorflow_hub()
    except (ImportError, ModuleNotFoundError, RuntimeError) as error:
        if "pkg_resources" in str(error):
            command = (
                'python -m pip install --force-reinstall "setuptools==80.9.0"'
            )
        else:
            command = (
                'python -m pip install "tensorflow-hub==0.16.1" '
                '"tf-keras~=2.21.0"'
            )
        raise SystemExit(
            "Could not import TensorFlow Hub. Run:\n\n"
            f"  {command}\n\n"
            f"Original import error: {error}"
        ) from error


# ---------------------------------------------------------------------------
# Configuration and immutable class contract
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/raw/RealWaste")
MANIFEST_PATH = Path("data/manifest.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = MODEL_DIR / "tflite"
REPORT_ROOT = Path("reports")
VALIDATION_REPORT_DIR = REPORT_ROOT / "validation"
REPORT_DIR = REPORT_ROOT / "tflite"

IMAGE_SIZE = (224, 224)
VERIFY_IMAGES_PER_CLASS = 10
SEED = 42
MIN_EXPECTED_TFLITE_BYTES = 1_000_000

CLASS_NAMES = list(MODEL_CLASS_NAMES)
APP_CLASS_NAMES = list(CONTRACT_APP_CLASS_NAMES)
APP_GROUPS = {
    name: list(labels) for name, labels in CONTRACT_APP_GROUPS.items()
}

MODEL_CONFIGS = {
    "mobilenetv3large": {
        "display_name": "MobileNetV3Large",
        "weights_path": MODEL_DIR
        / "mobilenetv3large_no_aug_finetuned.weights.h5",
        "training_metrics_path": VALIDATION_REPORT_DIR
        / "mobilenetv3large_no_aug_finetuned"
        / "mobilenetv3large_no_aug_finetuned_metrics.json",
    },
    "efficientnet_lite0": {
        "display_name": "EfficientNet-Lite0",
        "weights_path": MODEL_DIR / "efficientnet_lite0_no_aug.weights.h5",
        "training_metrics_path": VALIDATION_REPORT_DIR
        / "efficientnet_lite0_no_aug"
        / "efficientnet_lite0_no_aug_metrics.json",
    },
}

config = MODEL_CONFIGS[args.model]
DISPLAY_NAME = config["display_name"]
WEIGHTS_PATH = config["weights_path"]
TRAINING_METRICS_PATH = config["training_metrics_path"]

FLOAT32_PATH = OUTPUT_DIR / f"{args.model}_float32.tflite"
FLOAT16_PATH = OUTPUT_DIR / f"{args.model}_float16.tflite"
REPORT_PATH = REPORT_DIR / f"{args.model}_tflite_export_report.json"
LABELS_PATH = OUTPUT_DIR / "labels_9.txt"
MAPPING_PATH = OUTPUT_DIR / "chemsort_mapping.json"
COMPARISON_PATH = REPORT_DIR / "tflite_candidate_comparison.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# General helpers
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


def size_mb(path: Path) -> float:
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


def normalize_outputs(
    outputs: np.ndarray,
    source: str = "model",
) -> np.ndarray:
    outputs = np.asarray(outputs, dtype=np.float32)
    expected_shape = (None, len(CLASS_NAMES))
    if outputs.ndim != 2 or outputs.shape[1] != len(CLASS_NAMES):
        raise SystemExit(
            f"{source} returned shape {outputs.shape}; expected "
            f"[batch, {expected_shape[1]}]."
        )
    if not np.isfinite(outputs).all():
        invalid_count = int((~np.isfinite(outputs)).sum())
        raise SystemExit(
            f"{source} returned {invalid_count} NaN/Inf value(s). "
            "Export/parity is invalid; do not integrate this file."
        )

    row_sums = outputs.sum(axis=1)
    looks_like_probabilities = (
        np.all(outputs >= -1e-6)
        and np.all(outputs <= 1.0 + 1e-6)
        and np.allclose(row_sums, 1.0, atol=1e-4)
    )
    if looks_like_probabilities:
        return outputs

    probabilities = tf.nn.softmax(outputs, axis=1).numpy()
    if not np.isfinite(probabilities).all():
        raise SystemExit(
            f"Softmax-normalized output from {source} contains NaN/Inf. "
            "Export/parity is invalid."
        )
    return probabilities


class_index = {name: index for index, name in enumerate(CLASS_NAMES)}
app_index = {name: index for index, name in enumerate(APP_CLASS_NAMES)}


def aggregate_to_app6(probabilities: np.ndarray) -> np.ndarray:
    return np.column_stack(aggregate_probability_columns(probabilities))


def evaluate_probabilities(
    probabilities: np.ndarray,
    true_9: np.ndarray,
    true_6: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    predicted_9 = np.argmax(probabilities, axis=1).astype(np.int32)
    predicted_6 = np.argmax(aggregate_to_app6(probabilities), axis=1).astype(
        np.int32
    )
    metrics = {
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
    return metrics, predicted_9, predicted_6


# ---------------------------------------------------------------------------
# Rebuild the exact trained architecture and load its winning weights
# ---------------------------------------------------------------------------


def classifier_head(features: tf.Tensor) -> tf.Tensor:
    features = tf.keras.layers.Dropout(0.30, name="dropout")(features)
    return tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="predictions",
    )(features)


def build_mobilenetv3large():
    backbone = tf.keras.applications.MobileNetV3Large(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        alpha=1.0,
        minimalistic=False,
        include_top=False,
        weights=None,
        include_preprocessing=False,
    )
    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="image",
    )
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenetv3_preprocess",
    )(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pool"
    )(x)
    return tf.keras.Model(
        inputs,
        classifier_head(x),
        name="chemsort_mobilenetv3large",
    )


def build_efficientnet_lite0():
    module_url = (
        "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
    )
    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="image",
    )
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 255.0,
        name="efficientnet_lite_preprocess",
    )(inputs)
    feature_extractor = hub.KerasLayer(
        module_url,
        trainable=False,
        output_shape=[1280],
        name="efficientnet_lite0_feature_vector",
    )
    x = feature_extractor(x)
    return tf.keras.Model(
        inputs,
        classifier_head(x),
        name="chemsort_efficientnet_lite0",
    )


require_path(DATA_ROOT, "Dataset root")
require_path(MANIFEST_PATH, "Manifest")
require_path(WEIGHTS_PATH, "Winning weights")
require_path(TRAINING_METRICS_PATH, "Training metrics")

model = (
    build_mobilenetv3large()
    if args.model == "mobilenetv3large"
    else build_efficientnet_lite0()
)
model.load_weights(str(WEIGHTS_PATH))
if model.output_shape[-1] != len(CLASS_NAMES):
    raise SystemExit(
        f"Model output has {model.output_shape[-1]} classes; "
        f"expected {len(CLASS_NAMES)}."
    )

with TRAINING_METRICS_PATH.open("r", encoding="utf-8") as file:
    training_metrics = json.load(file)
full_validation_metrics = training_metrics["validation"]


# ---------------------------------------------------------------------------
# Deterministic, balanced validation subset for numerical parity
# ---------------------------------------------------------------------------

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

subset_parts = []
for _, class_df in validation_df.groupby("original_label", sort=True):
    subset_parts.append(
        class_df.sample(
            n=min(VERIFY_IMAGES_PER_CLASS, len(class_df)),
            random_state=SEED,
        )
    )
validation_subset = (
    pd.concat(subset_parts, ignore_index=True)
    .sort_values(["original_label", "path"])
    .reset_index(drop=True)
)

absolute_paths = [
    (DATA_ROOT / str(relative_path)).resolve()
    for relative_path in validation_subset["path"]
]
missing_images = [str(path) for path in absolute_paths if not path.is_file()]
if missing_images:
    raise SystemExit(
        f"Parity subset contains {len(missing_images)} missing image(s).\n"
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


print(f"Loading {len(absolute_paths)} validation images for parity...")
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
# Keras reference inference (performed before export so bad source weights fail
# immediately instead of producing misleading TFLite artifacts)
# ---------------------------------------------------------------------------

print("Running Keras reference inference...")
keras_probabilities = normalize_outputs(
    model.predict(validation_images, batch_size=32, verbose=1),
    source=f"{DISPLAY_NAME} Keras reference",
)
keras_metrics, keras_predicted_9, keras_predicted_6 = evaluate_probabilities(
    keras_probabilities,
    true_9,
    true_6,
)


# ---------------------------------------------------------------------------
# Export through the route compatible with each Keras model family
# ---------------------------------------------------------------------------


def configure_converter(converter, use_float16: bool):
    if use_float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    return converter


def write_checked_tflite(path: Path, model_bytes: bytes) -> None:
    byte_count = len(model_bytes)
    if byte_count < MIN_EXPECTED_TFLITE_BYTES:
        raise SystemExit(
            f"Refusing incomplete TFLite export ({byte_count / 1024**2:.2f} "
            f"MB): {path}. Expected at least "
            f"{MIN_EXPECTED_TFLITE_BYTES / 1024**2:.2f} MB for this model."
        )
    path.write_bytes(model_bytes)


def convert_from_keras(use_float16: bool) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return configure_converter(converter, use_float16).convert()


def convert_from_saved_model(
    saved_model_directory: Path,
    use_float16: bool,
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_model_directory)
    )
    return configure_converter(converter, use_float16).convert()


if args.model == "mobilenetv3large":
    # Keras 3 application model: convert the loaded Keras model directly.
    # A separately captured tf.function can omit resource variables and create
    # an implausibly tiny, NaN-producing TFLite graph.
    print(f"Exporting {DISPLAY_NAME} directly from the Keras model...")
    print("Converting float32 model...")
    float32_bytes = convert_from_keras(use_float16=False)
    write_checked_tflite(FLOAT32_PATH, float32_bytes)

    print("Converting float16-weight model...")
    float16_bytes = convert_from_keras(use_float16=True)
    write_checked_tflite(FLOAT16_PATH, float16_bytes)
else:
    # TF Hub EfficientNet-Lite0 uses legacy tf.keras. Its native Keras 2
    # SavedModel export preserves the Hub layer and all tracked variables.
    print(f"Exporting {DISPLAY_NAME} through a Keras SavedModel...")
    with tempfile.TemporaryDirectory(
        prefix="chemsort_tflite_"
    ) as temp_directory:
        saved_model_directory = Path(temp_directory) / "saved_model"
        model.save(
            str(saved_model_directory),
            save_format="tf",
            include_optimizer=False,
        )

        print("Converting float32 model...")
        float32_bytes = convert_from_saved_model(
            saved_model_directory,
            use_float16=False,
        )
        write_checked_tflite(FLOAT32_PATH, float32_bytes)

        print("Converting float16-weight model...")
        float16_bytes = convert_from_saved_model(
            saved_model_directory,
            use_float16=True,
        )
        write_checked_tflite(FLOAT16_PATH, float16_bytes)


# ---------------------------------------------------------------------------
# TFLite inference and parity calculations
# ---------------------------------------------------------------------------


def run_tflite(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, dict]:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=max(1, min(4, os.cpu_count() or 1)),
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

    input_scale, input_zero = input_detail.get("quantization", (0.0, 0))
    output_scale, output_zero = output_detail.get("quantization", (0.0, 0))
    outputs = []

    for image in images:
        value = image[np.newaxis, ...]
        if np.issubdtype(input_detail["dtype"], np.integer):
            if input_scale == 0:
                raise SystemExit("Quantized input has invalid scale=0.")
            value = np.round(value / input_scale + input_zero)
            limits = np.iinfo(input_detail["dtype"])
            value = np.clip(value, limits.min, limits.max)
        value = value.astype(input_detail["dtype"])

        interpreter.set_tensor(input_detail["index"], value)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])[0]
        if np.issubdtype(output_detail["dtype"], np.integer):
            if output_scale == 0:
                raise SystemExit("Quantized output has invalid scale=0.")
            output = (output.astype(np.float32) - output_zero) * output_scale
        outputs.append(np.asarray(output, dtype=np.float32))

    io_metadata = {
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_shape": output_detail["shape"].tolist(),
        "output_dtype": np.dtype(output_detail["dtype"]).name,
    }
    return normalize_outputs(
        np.stack(outputs),
        source=f"TFLite {model_path.name}",
    ), io_metadata

print("Checking float32 TFLite parity...")
float32_probabilities, float32_io = run_tflite(
    FLOAT32_PATH,
    validation_images,
)
float32_metrics, float32_predicted_9, float32_predicted_6 = (
    evaluate_probabilities(float32_probabilities, true_9, true_6)
)

print("Checking float16 TFLite parity...")
float16_probabilities, float16_io = run_tflite(
    FLOAT16_PATH,
    validation_images,
)
float16_metrics, float16_predicted_9, float16_predicted_6 = (
    evaluate_probabilities(float16_probabilities, true_9, true_6)
)


def parity_summary(
    probabilities: np.ndarray,
    metrics: dict,
    predicted_9: np.ndarray,
    predicted_6: np.ndarray,
) -> dict:
    return {
        "max_absolute_probability_difference": float(
            np.max(np.abs(probabilities - keras_probabilities))
        ),
        "mean_absolute_probability_difference": float(
            np.mean(np.abs(probabilities - keras_probabilities))
        ),
        "prediction_agreement_9class": float(
            np.mean(predicted_9 == keras_predicted_9)
        ),
        "prediction_agreement_6class": float(
            np.mean(predicted_6 == keras_predicted_6)
        ),
        "accuracy_delta_9class": (
            metrics["accuracy_9class"] - keras_metrics["accuracy_9class"]
        ),
        "macro_f1_delta_9class": (
            metrics["macro_f1_9class"] - keras_metrics["macro_f1_9class"]
        ),
        "accuracy_delta_6class": (
            metrics["accuracy_6class"] - keras_metrics["accuracy_6class"]
        ),
        "macro_f1_delta_6class": (
            metrics["macro_f1_6class"] - keras_metrics["macro_f1_6class"]
        ),
    }


float32_parity = parity_summary(
    float32_probabilities,
    float32_metrics,
    float32_predicted_9,
    float32_predicted_6,
)
float16_parity = parity_summary(
    float16_probabilities,
    float16_metrics,
    float16_predicted_9,
    float16_predicted_6,
)


def passes_gate(parity: dict) -> bool:
    return bool(
        parity["prediction_agreement_9class"] >= 0.99
        and parity["prediction_agreement_6class"] >= 0.99
        and abs(parity["accuracy_delta_9class"]) <= 0.01
        and abs(parity["macro_f1_delta_9class"]) <= 0.01
        and abs(parity["accuracy_delta_6class"]) <= 0.01
        and abs(parity["macro_f1_delta_6class"]) <= 0.01
    )


float32_passed = passes_gate(float32_parity)
float16_passed = passes_gate(float16_parity)
if float16_passed:
    recommended_path = FLOAT16_PATH
    recommended_format = "float16"
elif float32_passed:
    recommended_path = FLOAT32_PATH
    recommended_format = "float32"
else:
    recommended_path = None
    recommended_format = None


# ---------------------------------------------------------------------------
# Persist mobile contract, report and cross-candidate table
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
        "note": "Do not divide by 255 in Flutter; preprocessing is embedded.",
    },
    "model_output_9class": {"classes_in_index_order": CLASS_NAMES},
    "app_output_6class": {
        "classes_in_index_order": APP_CLASS_NAMES,
        "groups": APP_GROUPS,
        "rule": "Sum probabilities within each group, then take argmax.",
    },
}
with MAPPING_PATH.open("w", encoding="utf-8") as file:
    json.dump(mobile_contract, file, indent=2, ensure_ascii=False)

report = {
    "status": "tflite_export_completed",
    "candidate": DISPLAY_NAME,
    "architecture_key": args.model,
    "source_weights": {
        "path": str(WEIGHTS_PATH),
        "sha256": sha256_file(WEIGHTS_PATH),
    },
    "full_validation_metrics": full_validation_metrics,
    "parity_subset": {
        "image_count": len(validation_subset),
        "images_per_original_class": VERIFY_IMAGES_PER_CLASS,
        "seed": SEED,
        "keras_metrics": keras_metrics,
    },
    "float32": {
        "path": str(FLOAT32_PATH),
        "sha256": sha256_file(FLOAT32_PATH),
        "size_mb": size_mb(FLOAT32_PATH),
        "io": float32_io,
        "metrics": float32_metrics,
        "parity": float32_parity,
        "passed_gate": float32_passed,
    },
    "float16": {
        "path": str(FLOAT16_PATH),
        "sha256": sha256_file(FLOAT16_PATH),
        "size_mb": size_mb(FLOAT16_PATH),
        "io": float16_io,
        "metrics": float16_metrics,
        "parity": float16_parity,
        "passed_gate": float16_passed,
    },
    "recommended_mobile_model": (
        str(recommended_path) if recommended_path is not None else None
    ),
    "recommended_format": recommended_format,
    "training_performed": False,
    "test_split_used": False,
}
with REPORT_PATH.open("w", encoding="utf-8") as file:
    json.dump(json_safe(report), file, indent=2, ensure_ascii=False)


def selected_block(candidate_report: dict) -> tuple[str, dict] | None:
    selected_format = candidate_report.get("recommended_format")
    if selected_format in ("float32", "float16"):
        return selected_format, candidate_report[selected_format]

    selected_path = str(candidate_report.get("recommended_mobile_model", ""))
    if "float16" in selected_path:
        return "float16", candidate_report["float16"]
    if "float32" in selected_path:
        return "float32", candidate_report["float32"]
    return None


def refresh_candidate_comparison() -> pd.DataFrame:
    rows = []

    v2_report_path = REPORT_DIR / "mobilenetv2_tflite_export_report.json"
    v2_validation_path = (
        VALIDATION_REPORT_DIR
        / "mobilenetv2_no_aug_finetuned"
        / "mobilenetv2_no_aug_finetuned_comparison.json"
    )
    if v2_report_path.exists() and v2_validation_path.exists():
        with v2_report_path.open("r", encoding="utf-8") as file:
            v2_report = json.load(file)
        with v2_validation_path.open("r", encoding="utf-8") as file:
            v2_validation = json.load(file).get("finetuned", {})
        selected = selected_block(v2_report)
        if selected is not None:
            selected_format, block = selected
            rows.append(
                {
                    "candidate": "MobileNetV2",
                    "validation_macro_f1_6class": v2_validation.get(
                        "macro_f1_6class"
                    ),
                    "tflite_format": selected_format,
                    "tflite_size_mb": block["size_mb"],
                    "agreement_9class": block["parity"][
                        "prediction_agreement_9class"
                    ],
                    "agreement_6class": block["parity"][
                        "prediction_agreement_6class"
                    ],
                    "max_probability_difference": block["parity"][
                        "max_absolute_probability_difference"
                    ],
                }
            )

    for architecture_key in ("mobilenetv3large", "efficientnet_lite0"):
        path = REPORT_DIR / f"{architecture_key}_tflite_export_report.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            candidate_report = json.load(file)
        selected = selected_block(candidate_report)
        if selected is None:
            continue
        selected_format, block = selected
        rows.append(
            {
                "candidate": candidate_report["candidate"],
                "validation_macro_f1_6class": candidate_report[
                    "full_validation_metrics"
                ]["macro_f1_6class"],
                "tflite_format": selected_format,
                "tflite_size_mb": block["size_mb"],
                "agreement_9class": block["parity"][
                    "prediction_agreement_9class"
                ],
                "agreement_6class": block["parity"][
                    "prediction_agreement_6class"
                ],
                "max_probability_difference": block["parity"][
                    "max_absolute_probability_difference"
                ],
            }
        )

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison = comparison.sort_values(
            "validation_macro_f1_6class",
            ascending=False,
        )
        comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    return comparison


comparison = refresh_candidate_comparison()

print("\nTFLITE CANDIDATE EXPORT RESULTS")
print(f"Candidate: {DISPLAY_NAME}")
print(
    f"{'Model':<12} {'Size (MB)':>11} {'Agree 9':>10} "
    f"{'Agree 6':>10} {'Max |diff|':>12} {'Gate':>7}"
)
print("-" * 69)
for name, path, parity, passed in (
    ("float32", FLOAT32_PATH, float32_parity, float32_passed),
    ("float16", FLOAT16_PATH, float16_parity, float16_passed),
):
    print(
        f"{name:<12} {size_mb(path):>11.2f} "
        f"{parity['prediction_agreement_9class']:>10.4f} "
        f"{parity['prediction_agreement_6class']:>10.4f} "
        f"{parity['max_absolute_probability_difference']:>12.6f} "
        f"{('PASS' if passed else 'FAIL'):>7}"
    )

print(
    "Recommended mobile model: "
    + (str(recommended_path) if recommended_path is not None else "NONE")
)
print(f"Export report:            {REPORT_PATH}")
print("Training performed:       False")
print("Test split used:          False")

if not comparison.empty:
    print("\nAVAILABLE TFLITE COMPARISON")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"Comparison CSV: {COMPARISON_PATH}")

if recommended_path is None:
    raise SystemExit(
        "Neither TFLite variant passed parity. Do not integrate this candidate."
    )
