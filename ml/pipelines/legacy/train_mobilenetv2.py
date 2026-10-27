"""Train a frozen-backbone MobileNetV2 baseline for ChemSort.

Run this file from the ChemSort project root, for example:

    python -m pipelines.legacy.train_mobilenetv2 --augmentation off
    python -m pipelines.legacy.train_mobilenetv2 --augmentation on

The two runs use the same data split and hyperparameters. Only training-time
augmentation changes. Validation and test images are never augmented. This
script evaluates validation data only; the test split remains locked.
"""

from __future__ import annotations

import argparse
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
# Command-line configuration
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train MobileNetV2 with or without augmentation."
)
parser.add_argument(
    "--augmentation",
    choices=("on", "off"),
    required=True,
    help="Enable or disable augmentation for the training split.",
)
args = parser.parse_args()

USE_AUGMENTATION = args.augmentation == "on"
RUN_NAME = "mobilenetv2_aug" if USE_AUGMENTATION else "mobilenetv2_no_aug"


# ---------------------------------------------------------------------------
# Fixed experiment configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/raw/RealWaste")
MANIFEST_PATH = Path("data/manifest.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/validation") / RUN_NAME

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / f"{RUN_NAME}.keras"
HISTORY_PATH = REPORT_DIR / f"{RUN_NAME}_history.csv"
METRICS_PATH = REPORT_DIR / f"{RUN_NAME}_metrics.json"

tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# Manifest and label contract
# ---------------------------------------------------------------------------

if not MANIFEST_PATH.exists():
    raise SystemExit(f"Manifest not found: {MANIFEST_PATH.resolve()}")

if not DATA_ROOT.exists():
    raise SystemExit(f"Dataset root not found: {DATA_ROOT.resolve()}")

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

print(f"Run name: {RUN_NAME}")
print(f"Augmentation: {'ON' if USE_AUGMENTATION else 'OFF'}")
print("Class mapping:")
for label, index in label_to_id.items():
    print(f"  {index}: {label}")

with (MODEL_DIR / "class_names.json").open("w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="train_augmentation",
)


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


def augment_batch(images: tf.Tensor, labels: tf.Tensor):
    """Apply augmentation to a training batch only."""

    return augmentation(images, training=True), labels


def create_dataset(split_name: str, training: bool = False):
    """Create an ordered validation dataset or shuffled training dataset."""

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

    if training and USE_AUGMENTATION:
        dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, split_df


train_dataset, train_df = create_dataset("train", training=True)
val_dataset, val_df = create_dataset("val", training=False)

print("Dataset:")
print(f"  Train images:      {len(train_df)}")
print(f"  Validation images: {len(val_df)}")
print("  Test set used:     False")


# ---------------------------------------------------------------------------
# Frozen-backbone MobileNetV2 baseline
# ---------------------------------------------------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(
    shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    name="image",
)

# Equivalent to MobileNetV2 preprocess_input: [0, 255] -> [-1, 1].
x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1.0,
    name="mobilenet_preprocess",
)(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
x = tf.keras.layers.Dropout(0.30, name="dropout")(x)
outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax",
    name="predictions",
)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="chemsort_mobilenetv2")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.20,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(str(HISTORY_PATH)),
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)

best_model = tf.keras.models.load_model(str(BEST_MODEL_PATH))


# ---------------------------------------------------------------------------
# Validation evaluation: original 9 classes
# ---------------------------------------------------------------------------

probabilities = best_model.predict(val_dataset, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)
true_labels = val_df["label_id"].to_numpy(dtype=np.int32)

all_9_labels = np.arange(len(class_names))

val_accuracy_9 = accuracy_score(true_labels, predicted_labels)
val_macro_f1_9 = f1_score(
    true_labels,
    predicted_labels,
    labels=all_9_labels,
    average="macro",
    zero_division=0,
)

report_9 = classification_report(
    true_labels,
    predicted_labels,
    labels=all_9_labels,
    target_names=class_names,
    digits=4,
    zero_division=0,
)

report_9_path = REPORT_DIR / f"{RUN_NAME}_9class_val_report.txt"
report_9_path.write_text(report_9, encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation evaluation: ChemSort 6 classes using probability aggregation
# ---------------------------------------------------------------------------

app_class_names = list(APP_CLASS_NAMES)

app_probabilities = np.column_stack(aggregate_probability_columns(probabilities))
original_to_app = dict(enumerate(model_to_app_ids()))

app_true_labels = np.asarray(
    [original_to_app[int(label)] for label in true_labels],
    dtype=np.int32,
)
app_predicted_labels = np.argmax(app_probabilities, axis=1)
all_6_labels = np.arange(len(app_class_names))

val_accuracy_6 = accuracy_score(app_true_labels, app_predicted_labels)
val_macro_f1_6 = f1_score(
    app_true_labels,
    app_predicted_labels,
    labels=all_6_labels,
    average="macro",
    zero_division=0,
)

report_6 = classification_report(
    app_true_labels,
    app_predicted_labels,
    labels=all_6_labels,
    target_names=app_class_names,
    digits=4,
    zero_division=0,
)

report_6_path = REPORT_DIR / f"{RUN_NAME}_6class_val_report.txt"
report_6_path.write_text(report_6, encoding="utf-8")


# ---------------------------------------------------------------------------
# Plots and reports
# ---------------------------------------------------------------------------


def save_confusion_matrix(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    labels: np.ndarray,
    display_names: list[str],
    title: str,
    output_path: Path,
):
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


save_confusion_matrix(
    true_labels,
    predicted_labels,
    all_9_labels,
    class_names,
    f"MobileNetV2 ({args.augmentation}) - 9-class validation",
    REPORT_DIR / f"{RUN_NAME}_9class_val_confusion_matrix.png",
)

save_confusion_matrix(
    app_true_labels,
    app_predicted_labels,
    all_6_labels,
    app_class_names,
    f"MobileNetV2 ({args.augmentation}) - ChemSort 6-class validation",
    REPORT_DIR / f"{RUN_NAME}_6class_val_confusion_matrix.png",
)

epochs_completed = range(1, len(history.history["loss"]) + 1)
figure, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(epochs_completed, history.history["accuracy"], label="Train")
axes[0].plot(epochs_completed, history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(epochs_completed, history.history["loss"], label="Train")
axes[1].plot(epochs_completed, history.history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

figure.suptitle(f"{RUN_NAME} training curves")
figure.tight_layout()
figure.savefig(REPORT_DIR / f"{RUN_NAME}_training_curves.png", dpi=180)
plt.close(figure)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

metrics = {
    "run_name": RUN_NAME,
    "model": "MobileNetV2",
    "stage": "frozen_backbone_baseline",
    "augmentation": USE_AUGMENTATION,
    "image_size": list(IMAGE_SIZE),
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "learning_rate": LEARNING_RATE,
    "epochs_requested": EPOCHS,
    "epochs_completed": len(history.history["loss"]),
    "best_epoch_by_val_loss": best_epoch,
    "best_val_loss": float(min(history.history["val_loss"])),
    "validation_9class": {
        "accuracy": float(val_accuracy_9),
        "macro_f1": float(val_macro_f1_9),
    },
    "validation_6class_probability_aggregation": {
        "accuracy": float(val_accuracy_6),
        "macro_f1": float(val_macro_f1_6),
    },
    "test_set_used": False,
}

with METRICS_PATH.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2, ensure_ascii=False)

print("\n9-class validation report:")
print(report_9)
print(f"9-class Accuracy: {val_accuracy_9:.4f}")
print(f"9-class Macro-F1: {val_macro_f1_9:.4f}")

print("\nChemSort 6-class validation report:")
print(report_6)
print(f"6-class Accuracy: {val_accuracy_6:.4f}")
print(f"6-class Macro-F1: {val_macro_f1_6:.4f}")

print("\nTraining complete.")
print(f"Best epoch: {best_epoch}")
print(f"Best model: {BEST_MODEL_PATH}")
print(f"Metrics: {METRICS_PATH}")
print("Test set used: False")
