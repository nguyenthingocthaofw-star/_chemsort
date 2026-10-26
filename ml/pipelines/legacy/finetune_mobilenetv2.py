"""Fine-tune the winning ChemSort MobileNetV2 no-augmentation baseline.

Run this module from the project root:

    python -m pipelines.legacy.finetune_mobilenetv2

The script:
1. Loads ``models/mobilenetv2_no_aug.keras``.
2. Measures the unchanged baseline on validation data.
3. Unfreezes the final 30 MobileNetV2 layers while keeping BatchNorm frozen.
4. Fine-tunes with learning rate 1e-5 and no augmentation.
5. Compares baseline and fine-tuned validation metrics at both 9-class and
   ChemSort 6-class levels.

The test split is never loaded or evaluated.
"""

from __future__ import annotations

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

from chemsort.contracts import (
    APP_CLASS_NAMES,
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
# Experiment configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/raw/RealWaste")
MANIFEST_PATH = Path("data/manifest.csv")

MODEL_DIR = Path("models")

BASELINE_MODEL_PATH = MODEL_DIR / "mobilenetv2_no_aug.keras"
RUN_NAME = "mobilenetv2_no_aug_finetuned"
REPORT_DIR = Path("reports/validation") / RUN_NAME
BEST_MODEL_PATH = MODEL_DIR / f"{RUN_NAME}.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 8
FINE_TUNE_LEARNING_RATE = 1e-5
UNFREEZE_LAST_LAYERS = 30
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_PATH = REPORT_DIR / f"{RUN_NAME}_history.csv"
COMPARISON_PATH = REPORT_DIR / f"{RUN_NAME}_comparison.json"

tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# Manifest and label contract
# ---------------------------------------------------------------------------

if not MANIFEST_PATH.exists():
    raise SystemExit(f"Manifest not found: {MANIFEST_PATH.resolve()}")

if not DATA_ROOT.exists():
    raise SystemExit(f"Dataset root not found: {DATA_ROOT.resolve()}")

if not BASELINE_MODEL_PATH.exists():
    raise SystemExit(
        "Baseline model not found: "
        f"{BASELINE_MODEL_PATH.resolve()}\n"
        "Run the no-augmentation baseline first."
    )

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = {"path", "original_label", "split"}
missing_columns = required_columns - set(manifest.columns)
if missing_columns:
    raise SystemExit(
        "manifest.csv is missing columns: " + ", ".join(sorted(missing_columns))
    )

class_names = sorted(manifest["original_label"].dropna().unique().tolist())

expected_class_names = list(MODEL_CLASS_NAMES)

if class_names != expected_class_names:
    raise SystemExit(
        "Unexpected class names.\n"
        f"Expected: {expected_class_names}\n"
        f"Found:    {class_names}"
    )

label_to_id = {label: index for index, label in enumerate(class_names)}
manifest["label_id"] = manifest["original_label"].map(label_to_id)

if manifest["label_id"].isna().any():
    raise SystemExit("Some manifest labels could not be converted to label IDs.")


# ---------------------------------------------------------------------------
# Data pipeline: no augmentation
# ---------------------------------------------------------------------------


def load_image(path: tf.Tensor, label: tf.Tensor):
    """Read one image, convert it to RGB and resize it to 224 x 224."""

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


def create_dataset(split_name: str, training: bool = False):
    """Create the specified manifest split without augmentation."""

    split_df = (
        manifest.loc[manifest["split"] == split_name]
        .copy()
        .reset_index(drop=True)
    )

    if split_df.empty:
        raise SystemExit(f"Split '{split_name}' is empty.")

    paths = np.asarray(
        [str((DATA_ROOT / relative_path).resolve()) for relative_path in split_df["path"]],
        dtype=str,
    )
    labels = split_df["label_id"].to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(split_df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, split_df


train_dataset, train_df = create_dataset("train", training=True)
val_dataset, val_df = create_dataset("val", training=False)
true_labels = val_df["label_id"].to_numpy(dtype=np.int32)

print("Fine-tuning configuration:")
print(f"  Baseline model:      {BASELINE_MODEL_PATH}")
print(f"  Train images:        {len(train_df)}")
print(f"  Validation images:   {len(val_df)}")
print(f"  Augmentation:        OFF")
print(f"  Unfreeze last:       {UNFREEZE_LAST_LAYERS} backbone layers")
print(f"  Learning rate:       {FINE_TUNE_LEARNING_RATE}")
print(f"  Maximum epochs:      {FINE_TUNE_EPOCHS}")
print("  Test set used:       False")


# ---------------------------------------------------------------------------
# Shared validation evaluation helpers
# ---------------------------------------------------------------------------

app_class_names = list(APP_CLASS_NAMES)

original_to_app = dict(enumerate(model_to_app_ids()))

app_true_labels = np.asarray(
    [original_to_app[int(label)] for label in true_labels],
    dtype=np.int32,
)


def aggregate_probabilities_to_app6(probabilities: np.ndarray):
    """Convert nine model probabilities into six ChemSort probabilities."""

    return np.column_stack(aggregate_probability_columns(probabilities))


def calculate_validation_metrics(probabilities: np.ndarray):
    """Calculate 9-class and correctly aggregated 6-class validation metrics."""

    predicted_9 = np.argmax(probabilities, axis=1)
    labels_9 = np.arange(len(class_names))

    report_9_dict = classification_report(
        true_labels,
        predicted_9,
        labels=labels_9,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_9_text = classification_report(
        true_labels,
        predicted_9,
        labels=labels_9,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    probabilities_6 = aggregate_probabilities_to_app6(probabilities)
    predicted_6 = np.argmax(probabilities_6, axis=1)
    labels_6 = np.arange(len(app_class_names))

    report_6_dict = classification_report(
        app_true_labels,
        predicted_6,
        labels=labels_6,
        target_names=app_class_names,
        output_dict=True,
        zero_division=0,
    )
    report_6_text = classification_report(
        app_true_labels,
        predicted_6,
        labels=labels_6,
        target_names=app_class_names,
        digits=4,
        zero_division=0,
    )

    return {
        "predicted_9": predicted_9,
        "predicted_6": predicted_6,
        "report_9_text": report_9_text,
        "report_6_text": report_6_text,
        "accuracy_9": float(accuracy_score(true_labels, predicted_9)),
        "macro_f1_9": float(
            f1_score(
                true_labels,
                predicted_9,
                labels=labels_9,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy_6": float(accuracy_score(app_true_labels, predicted_6)),
        "macro_f1_6": float(
            f1_score(
                app_true_labels,
                predicted_6,
                labels=labels_6,
                average="macro",
                zero_division=0,
            )
        ),
        "organic_precision": float(report_6_dict["Organic"]["precision"]),
        "organic_recall": float(report_6_dict["Organic"]["recall"]),
        "plastic_precision": float(report_6_dict["Plastic"]["precision"]),
        "plastic_recall": float(report_6_dict["Plastic"]["recall"]),
        "report_9_dict": report_9_dict,
        "report_6_dict": report_6_dict,
    }


def save_confusion_matrix(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    labels: np.ndarray,
    display_names: list[str],
    title: str,
    output_path: Path,
):
    """Save a confusion matrix without opening a GUI window."""

    matrix = confusion_matrix(true_values, predicted_values, labels=labels)
    figure, axis = plt.subplots(figsize=(12, 10))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=display_names,
    )
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


# ---------------------------------------------------------------------------
# Baseline measurement before changing trainable layers
# ---------------------------------------------------------------------------

model = tf.keras.models.load_model(str(BASELINE_MODEL_PATH))

if model.output_shape[-1] != len(class_names):
    raise SystemExit(
        f"Model output has {model.output_shape[-1]} classes; expected {len(class_names)}."
    )

print("\nMeasuring unchanged baseline on validation data...")
baseline_probabilities = model.predict(val_dataset, verbose=1)
baseline_metrics = calculate_validation_metrics(baseline_probabilities)


# ---------------------------------------------------------------------------
# Locate and configure the nested MobileNetV2 backbone
# ---------------------------------------------------------------------------

backbone_candidates = [
    layer
    for layer in model.layers
    if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower()
]

if len(backbone_candidates) != 1:
    candidate_names = [layer.name for layer in backbone_candidates]
    raise SystemExit(
        "Could not uniquely locate the MobileNetV2 backbone. "
        f"Candidates: {candidate_names}"
    )

backbone = backbone_candidates[0]
backbone.trainable = True

# Freeze the complete backbone first.
for layer in backbone.layers:
    layer.trainable = False

# Re-enable only the final layers, while BatchNormalization stays frozen.
for layer in backbone.layers[-UNFREEZE_LAST_LAYERS:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

unfrozen_backbone_layers = [
    layer.name for layer in backbone.layers if layer.trainable
]

print("\nBackbone:")
print(f"  Name:                       {backbone.name}")
print(f"  Total layers:               {len(backbone.layers)}")
print(f"  Trainable backbone layers:  {len(unfrozen_backbone_layers)}")
print("  BatchNormalization layers:  frozen")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

trainable_parameter_count = int(
    sum(np.prod(weight.shape) for weight in model.trainable_weights)
)
print(f"  Trainable parameters:       {trainable_parameter_count:,}")


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.20,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(str(HISTORY_PATH)),
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
    shuffle=False,
)

best_model = tf.keras.models.load_model(str(BEST_MODEL_PATH))


# ---------------------------------------------------------------------------
# Fine-tuned validation metrics
# ---------------------------------------------------------------------------

print("\nMeasuring fine-tuned model on validation data...")
finetuned_probabilities = best_model.predict(val_dataset, verbose=1)
finetuned_metrics = calculate_validation_metrics(finetuned_probabilities)

(REPORT_DIR / f"{RUN_NAME}_9class_val_report.txt").write_text(
    finetuned_metrics["report_9_text"],
    encoding="utf-8",
)
(REPORT_DIR / f"{RUN_NAME}_6class_val_report.txt").write_text(
    finetuned_metrics["report_6_text"],
    encoding="utf-8",
)

save_confusion_matrix(
    true_labels,
    finetuned_metrics["predicted_9"],
    np.arange(len(class_names)),
    class_names,
    "MobileNetV2 fine-tuned - 9-class validation",
    REPORT_DIR / f"{RUN_NAME}_9class_val_confusion_matrix.png",
)

save_confusion_matrix(
    app_true_labels,
    finetuned_metrics["predicted_6"],
    np.arange(len(app_class_names)),
    app_class_names,
    "MobileNetV2 fine-tuned - ChemSort 6-class validation",
    REPORT_DIR / f"{RUN_NAME}_6class_val_confusion_matrix.png",
)


# ---------------------------------------------------------------------------
# Fine-tuning curves
# ---------------------------------------------------------------------------

epochs_completed = range(1, len(history.history["loss"]) + 1)
figure, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(epochs_completed, history.history["accuracy"], label="Train")
axes[0].plot(epochs_completed, history.history["val_accuracy"], label="Validation")
axes[0].set_title("Fine-tuning accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(epochs_completed, history.history["loss"], label="Train")
axes[1].plot(epochs_completed, history.history["val_loss"], label="Validation")
axes[1].set_title("Fine-tuning loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

figure.tight_layout()
figure.savefig(REPORT_DIR / f"{RUN_NAME}_training_curves.png", dpi=180)
plt.close(figure)


# ---------------------------------------------------------------------------
# Baseline-versus-fine-tuned comparison
# ---------------------------------------------------------------------------

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)


def compact_metrics(metrics: dict):
    return {
        "accuracy_9class": metrics["accuracy_9"],
        "macro_f1_9class": metrics["macro_f1_9"],
        "accuracy_6class": metrics["accuracy_6"],
        "macro_f1_6class": metrics["macro_f1_6"],
        "organic_precision_6class": metrics["organic_precision"],
        "organic_recall_6class": metrics["organic_recall"],
        "plastic_precision_6class": metrics["plastic_precision"],
        "plastic_recall_6class": metrics["plastic_recall"],
    }


comparison = {
    "baseline_model": str(BASELINE_MODEL_PATH),
    "finetuned_model": str(BEST_MODEL_PATH),
    "configuration": {
        "augmentation": False,
        "unfreeze_last_layers": UNFREEZE_LAST_LAYERS,
        "batch_normalization_frozen": True,
        "learning_rate": FINE_TUNE_LEARNING_RATE,
        "maximum_epochs": FINE_TUNE_EPOCHS,
        "epochs_completed": len(history.history["loss"]),
        "best_epoch_by_val_loss": best_epoch,
        "trainable_parameters": trainable_parameter_count,
        "seed": SEED,
    },
    "baseline": compact_metrics(baseline_metrics),
    "finetuned": compact_metrics(finetuned_metrics),
    "delta_finetuned_minus_baseline": {
        "accuracy_9class": finetuned_metrics["accuracy_9"]
        - baseline_metrics["accuracy_9"],
        "macro_f1_9class": finetuned_metrics["macro_f1_9"]
        - baseline_metrics["macro_f1_9"],
        "accuracy_6class": finetuned_metrics["accuracy_6"]
        - baseline_metrics["accuracy_6"],
        "macro_f1_6class": finetuned_metrics["macro_f1_6"]
        - baseline_metrics["macro_f1_6"],
        "organic_recall_6class": finetuned_metrics["organic_recall"]
        - baseline_metrics["organic_recall"],
        "plastic_precision_6class": finetuned_metrics["plastic_precision"]
        - baseline_metrics["plastic_precision"],
        "plastic_recall_6class": finetuned_metrics["plastic_recall"]
        - baseline_metrics["plastic_recall"],
    },
    "test_set_used": False,
}

with COMPARISON_PATH.open("w", encoding="utf-8") as file:
    json.dump(comparison, file, indent=2, ensure_ascii=False)

print("\nValidation comparison:")
print(
    f"{'Metric':<28} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}"
)
print("-" * 64)

comparison_rows = [
    ("9-class Accuracy", "accuracy_9"),
    ("9-class Macro-F1", "macro_f1_9"),
    ("6-class Accuracy", "accuracy_6"),
    ("6-class Macro-F1", "macro_f1_6"),
    ("Organic recall", "organic_recall"),
    ("Plastic precision", "plastic_precision"),
    ("Plastic recall", "plastic_recall"),
]

for label, key in comparison_rows:
    baseline_value = baseline_metrics[key]
    finetuned_value = finetuned_metrics[key]
    delta = finetuned_value - baseline_value
    print(
        f"{label:<28} "
        f"{baseline_value:>10.4f} "
        f"{finetuned_value:>12.4f} "
        f"{delta:>+10.4f}"
    )

print("\nFine-tuning complete.")
print(f"Best fine-tuning epoch: {best_epoch}")
print(f"Fine-tuned model: {BEST_MODEL_PATH}")
print(f"Comparison: {COMPARISON_PATH}")
print("Test set used: False")

if finetuned_metrics["macro_f1_6"] > baseline_metrics["macro_f1_6"]:
    print("Result: fine-tuning improved 6-class Macro-F1 on validation.")
else:
    print("Result: baseline still has the higher 6-class Macro-F1 on validation.")
