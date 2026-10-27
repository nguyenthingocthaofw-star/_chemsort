"""Train comparable ChemSort MobileNetV3Large and EfficientNet-Lite0 candidates.

Run from the ChemSort project root. Start with MobileNetV3Large baseline:

    python -m pipelines.legacy.train_candidate --model mobilenetv3large --stage baseline

Then, only after reviewing its validation result, fine-tune it:

    python -m pipelines.legacy.train_candidate --model mobilenetv3large --stage finetune

If the final fine-tuning epoch is still improving, continue from its checkpoint
without repeating the first eight epochs:

    python -m pipelines.legacy.train_candidate --model mobilenetv3large --stage finetune --resume --epochs 4

EfficientNet-Lite0 is provided by a legacy TF Hub feature-vector module, so
this script supports its frozen-backbone baseline only:

    python -m pipelines.legacy.train_candidate --model efficientnet_lite0 --stage baseline

All candidates use the same manifest, class order, seed, image size, head,
callbacks, and 9-to-6 probability aggregation. The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


# Parse before importing TensorFlow. EfficientNet-Lite0's legacy TF Hub module
# needs the Keras 2 compatibility package under TensorFlow 2.16+.
parser = argparse.ArgumentParser(
    description="Train a comparable ChemSort mobile model candidate."
)
parser.add_argument(
    "--model",
    choices=("mobilenetv3large", "efficientnet_lite0"),
    required=True,
    help="Backbone architecture to train.",
)
parser.add_argument(
    "--stage",
    choices=("baseline", "finetune"),
    default="baseline",
    help="Train a frozen baseline or fine-tune the saved baseline.",
)
parser.add_argument(
    "--augmentation",
    choices=("off", "on"),
    default="off",
    help="Training-only augmentation. Keep off for the controlled comparison.",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Continue MobileNetV3 fine-tuning from its current best checkpoint.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Override the number of epochs for this invocation.",
)
args = parser.parse_args()

if args.epochs is not None and args.epochs < 1:
    raise SystemExit("--epochs must be at least 1.")
if args.resume and not (
    args.model == "mobilenetv3large" and args.stage == "finetune"
):
    raise SystemExit(
        "--resume is supported only with "
        "--model mobilenetv3large --stage finetune."
    )

if args.model == "efficientnet_lite0":
    if args.stage != "baseline":
        raise SystemExit(
            "EfficientNet-Lite0 fine-tuning is disabled because the official "
            "feature-vector module is legacy TF Hub format. Run --stage baseline."
        )
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

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
from chemsort.tfhub_compat import import_tensorflow_hub
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

hub = None
if args.model == "efficientnet_lite0":
    try:
        hub = import_tensorflow_hub()
    except (ImportError, ModuleNotFoundError, RuntimeError) as error:
        raise SystemExit(
            "EfficientNet-Lite0 requires TensorFlow Hub and the Keras 2 "
            "compatibility package. Install them in the active environment:\n\n"
            "  python -m pip install \"tensorflow-hub==0.16.1\" "
            "\"tf-keras~=2.21.0\"\n\n"
            f"Original import error: {error}"
        ) from error


# ---------------------------------------------------------------------------
# Controlled experiment configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/raw/RealWaste")
MANIFEST_PATH = Path("data/manifest.csv")
MODEL_DIR = Path("models")
REPORT_ROOT = Path("reports")
VALIDATION_REPORT_DIR = REPORT_ROOT / "validation"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
BASELINE_EPOCHS = 15
FINE_TUNE_EPOCHS = 8
BASELINE_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
UNFREEZE_LAST_LAYERS = 30
SEED = 42

USE_AUGMENTATION = args.augmentation == "on"
AUGMENTATION_SUFFIX = "aug" if USE_AUGMENTATION else "no_aug"
BASE_RUN_NAME = f"{args.model}_{AUGMENTATION_SUFFIX}"
STANDARD_FINE_TUNE_RUN_NAME = f"{BASE_RUN_NAME}_finetuned"
if args.stage == "baseline":
    RUN_NAME = BASE_RUN_NAME
elif args.resume:
    RUN_NAME = f"{STANDARD_FINE_TUNE_RUN_NAME}_extended"
else:
    RUN_NAME = STANDARD_FINE_TUNE_RUN_NAME

default_epochs = (
    BASELINE_EPOCHS
    if args.stage == "baseline"
    else (4 if args.resume else FINE_TUNE_EPOCHS)
)
EPOCHS = args.epochs if args.epochs is not None else default_epochs
LEARNING_RATE = (
    BASELINE_LEARNING_RATE
    if args.stage == "baseline"
    else FINE_TUNE_LEARNING_RATE
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = VALIDATION_REPORT_DIR / RUN_NAME
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BEST_WEIGHTS_PATH = MODEL_DIR / f"{RUN_NAME}.weights.h5"
BEST_MODEL_PATH = MODEL_DIR / f"{RUN_NAME}.keras"
BASELINE_WEIGHTS_PATH = MODEL_DIR / f"{BASE_RUN_NAME}.weights.h5"
FINE_TUNED_WEIGHTS_PATH = (
    MODEL_DIR / f"{STANDARD_FINE_TUNE_RUN_NAME}.weights.h5"
)
FINE_TUNED_METRICS_PATH = (
    VALIDATION_REPORT_DIR
    / STANDARD_FINE_TUNE_RUN_NAME
    / f"{STANDARD_FINE_TUNE_RUN_NAME}_metrics.json"
)
FINE_TUNED_HISTORY_PATH = (
    VALIDATION_REPORT_DIR
    / STANDARD_FINE_TUNE_RUN_NAME
    / f"{STANDARD_FINE_TUNE_RUN_NAME}_history.csv"
)
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

with (MODEL_DIR / "class_names.json").open("w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Identical tf.data pipeline for every candidate
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
    return augmentation(images, training=True), labels


def create_dataset(split_name: str, training: bool = False):
    split_df = (
        manifest.loc[manifest["split"] == split_name]
        .copy()
        .reset_index(drop=True)
    )
    if split_df.empty:
        raise SystemExit(f"Split '{split_name}' is empty.")

    paths = np.asarray(
        [
            str((DATA_ROOT / relative_path).resolve())
            for relative_path in split_df["path"]
        ],
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


# ---------------------------------------------------------------------------
# Architecture builders with raw [0, 255] RGB input contract
# ---------------------------------------------------------------------------


def classifier_head(features: tf.Tensor) -> tf.Tensor:
    features = tf.keras.layers.Dropout(0.30, name="dropout")(features)
    return tf.keras.layers.Dense(
        len(class_names),
        activation="softmax",
        name="predictions",
    )(features)


def build_mobilenetv3large():
    backbone = tf.keras.applications.MobileNetV3Large(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        alpha=1.0,
        minimalistic=False,
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,
    )
    backbone.trainable = False

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="image",
    )
    # MobileNetV3 without built-in preprocessing expects [-1, 1].
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenetv3_preprocess",
    )(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pool"
    )(x)
    outputs = classifier_head(x)
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="chemsort_mobilenetv3large",
    )
    metadata = {
        "display_name": "MobileNetV3Large",
        "preprocessing": "embedded [0,255] to [-1,1] rescaling",
        "pretrained_weights": "ImageNet via tf.keras.applications",
        "backbone_fine_tuning_supported": True,
    }
    return model, backbone, metadata


def build_efficientnet_lite0():
    module_url = (
        "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
    )
    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="image",
    )
    # TF Hub image modules use float images in [0, 1].
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
    outputs = classifier_head(x)
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="chemsort_efficientnet_lite0",
    )
    metadata = {
        "display_name": "EfficientNet-Lite0",
        "preprocessing": "embedded [0,255] to [0,1] rescaling",
        "pretrained_weights": module_url,
        "backbone_fine_tuning_supported": False,
        "limitation": "Official module is legacy TF Hub feature-vector format.",
    }
    return model, feature_extractor, metadata


if args.model == "mobilenetv3large":
    model, backbone, architecture_metadata = build_mobilenetv3large()
else:
    model, backbone, architecture_metadata = build_efficientnet_lite0()


# ---------------------------------------------------------------------------
# Baseline or controlled fine-tuning configuration
# ---------------------------------------------------------------------------

unfrozen_backbone_layers = []
source_weights_path = None
initial_epoch = 0
effective_stage = "finetune_continued" if args.resume else args.stage

if args.stage == "finetune":
    source_weights_path = (
        FINE_TUNED_WEIGHTS_PATH if args.resume else BASELINE_WEIGHTS_PATH
    )
    if not source_weights_path.exists():
        raise SystemExit(
            "Source weights not found: "
            f"{source_weights_path.resolve()}\n"
            + (
                "Run the normal --stage finetune command first."
                if args.resume
                else "Run --stage baseline first with the same model and augmentation."
            )
        )

    model.load_weights(str(source_weights_path))
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-UNFREEZE_LAST_LAYERS:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    unfrozen_backbone_layers = [
        layer.name for layer in backbone.layers if layer.trainable
    ]

    if args.resume:
        if FINE_TUNED_METRICS_PATH.exists():
            with FINE_TUNED_METRICS_PATH.open("r", encoding="utf-8") as file:
                previous_metrics = json.load(file)
            initial_epoch = int(previous_metrics.get("epochs_completed", 0))
        elif FINE_TUNED_HISTORY_PATH.exists():
            initial_epoch = len(pd.read_csv(FINE_TUNED_HISTORY_PATH))
        else:
            initial_epoch = FINE_TUNE_EPOCHS

final_epoch = initial_epoch + EPOCHS

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

total_parameters = int(model.count_params())
trainable_parameters = int(
    sum(np.prod(variable.shape) for variable in model.trainable_weights)
)

print("Controlled candidate experiment:")
print(f"  Run name:             {RUN_NAME}")
print(f"  Architecture:         {architecture_metadata['display_name']}")
print(f"  Stage:                {effective_stage}")
print(f"  Augmentation:         {'ON' if USE_AUGMENTATION else 'OFF'}")
print(f"  Train images:         {len(train_df)}")
print(f"  Validation images:    {len(val_df)}")
print(f"  Input contract:       RGB float32 [0, 255]")
print(f"  Learning rate:        {LEARNING_RATE}")
if args.resume:
    print(f"  Resume checkpoint:    {source_weights_path}")
    print(f"  Previous epochs:      {initial_epoch}")
    print(f"  Additional epochs:    {EPOCHS}")
    print(f"  Final epoch limit:    {final_epoch}")
else:
    print(f"  Maximum epochs:       {EPOCHS}")
print(f"  Total parameters:     {total_parameters:,}")
print(f"  Trainable parameters: {trainable_parameters:,}")
if args.stage == "finetune":
    print(f"  Unfrozen layers:      {len(unfrozen_backbone_layers)}")
    print("  BatchNorm:            frozen")
print("  Test set used:        False")

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_WEIGHTS_PATH),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4 if args.stage == "baseline" else 3,
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
    initial_epoch=initial_epoch,
    epochs=final_epoch,
    callbacks=callbacks,
)
model.load_weights(str(BEST_WEIGHTS_PATH))

# Save the selected weights as a complete model. The weights checkpoint remains
# the reconstruction fallback for legacy TF Hub environments.
model_artifact = str(BEST_MODEL_PATH)
model_save_error = None
try:
    model.save(str(BEST_MODEL_PATH))
except Exception as error:  # Preserve successful training if Hub serialization fails.
    model_save_error = f"{type(error).__name__}: {error}"
    model_artifact = str(BEST_WEIGHTS_PATH)
    print("WARNING: full-model save failed; best weights remain available.")
    print(model_save_error)


# ---------------------------------------------------------------------------
# Validation evaluation at 9-class and ChemSort 6-class levels
# ---------------------------------------------------------------------------

probabilities = np.asarray(model.predict(val_dataset, verbose=1))
predicted_9 = np.argmax(probabilities, axis=1).astype(np.int32)
true_9 = val_df["label_id"].to_numpy(dtype=np.int32)
labels_9 = np.arange(len(class_names))

report_9_dict = classification_report(
    true_9,
    predicted_9,
    labels=labels_9,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)
report_9_text = classification_report(
    true_9,
    predicted_9,
    labels=labels_9,
    target_names=class_names,
    digits=4,
    zero_division=0,
)

app_class_names = list(APP_CLASS_NAMES)
probabilities_6 = np.column_stack(aggregate_probability_columns(probabilities))
original_to_app = dict(enumerate(model_to_app_ids()))
true_6 = np.asarray(
    [original_to_app[int(label)] for label in true_9],
    dtype=np.int32,
)
predicted_6 = np.argmax(probabilities_6, axis=1).astype(np.int32)
labels_6 = np.arange(len(app_class_names))

report_6_dict = classification_report(
    true_6,
    predicted_6,
    labels=labels_6,
    target_names=app_class_names,
    output_dict=True,
    zero_division=0,
)
report_6_text = classification_report(
    true_6,
    predicted_6,
    labels=labels_6,
    target_names=app_class_names,
    digits=4,
    zero_division=0,
)

validation_metrics = {
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

(REPORT_DIR / f"{RUN_NAME}_9class_val_report.txt").write_text(
    report_9_text,
    encoding="utf-8",
)
(REPORT_DIR / f"{RUN_NAME}_6class_val_report.txt").write_text(
    report_6_text,
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# Plots, metrics and available-candidate comparison
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
    true_9,
    predicted_9,
    labels_9,
    class_names,
    f"{architecture_metadata['display_name']} - 9-class validation",
    REPORT_DIR / f"{RUN_NAME}_9class_val_confusion_matrix.png",
)
save_confusion_matrix(
    true_6,
    predicted_6,
    labels_6,
    app_class_names,
    f"{architecture_metadata['display_name']} - 6-class validation",
    REPORT_DIR / f"{RUN_NAME}_6class_val_confusion_matrix.png",
)

epochs_completed = range(
    initial_epoch + 1,
    initial_epoch + len(history.history["loss"]) + 1,
)
figure, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(epochs_completed, history.history["accuracy"], label="Train")
axes[0].plot(
    epochs_completed,
    history.history["val_accuracy"],
    label="Validation",
)
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(epochs_completed, history.history["loss"], label="Train")
axes[1].plot(
    epochs_completed,
    history.history["val_loss"],
    label="Validation",
)
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
figure.suptitle(f"{RUN_NAME} training curves")
figure.tight_layout()
figure.savefig(REPORT_DIR / f"{RUN_NAME}_training_curves.png", dpi=180)
plt.close(figure)

best_epoch = int(
    initial_epoch + np.argmin(history.history["val_loss"]) + 1
)
total_epochs_completed = initial_epoch + len(history.history["loss"])
metrics = {
    "run_name": RUN_NAME,
    "architecture": architecture_metadata,
    "stage": effective_stage,
    "resumed": args.resume,
    "source_weights": (
        str(source_weights_path) if source_weights_path is not None else None
    ),
    "augmentation": USE_AUGMENTATION,
    "image_size": list(IMAGE_SIZE),
    "input_contract": "RGB float32 [0,255]",
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "learning_rate": LEARNING_RATE,
    "initial_epoch": initial_epoch,
    "additional_epochs_requested": EPOCHS,
    "epochs_requested": final_epoch,
    "epochs_completed": total_epochs_completed,
    "best_epoch_by_val_loss": best_epoch,
    "best_val_loss": float(min(history.history["val_loss"])),
    "total_parameters": total_parameters,
    "trainable_parameters": trainable_parameters,
    "unfrozen_backbone_layers": unfrozen_backbone_layers,
    "model_artifact": model_artifact,
    "best_weights": str(BEST_WEIGHTS_PATH),
    "model_save_error": model_save_error,
    "validation": validation_metrics,
    "classification_report_9class": report_9_dict,
    "classification_report_6class": report_6_dict,
    "test_set_used": False,
}
with METRICS_PATH.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2, ensure_ascii=False)


def add_standard_candidate(
    rows: list[dict],
    name: str,
    stage: str,
    candidate_metrics: dict,
    parameters: int | None,
):
    rows.append(
        {
            "candidate": name,
            "stage": stage,
            "accuracy_9class": candidate_metrics["accuracy_9class"],
            "macro_f1_9class": candidate_metrics["macro_f1_9class"],
            "accuracy_6class": candidate_metrics["accuracy_6class"],
            "macro_f1_6class": candidate_metrics["macro_f1_6class"],
            "organic_recall_6class": candidate_metrics.get(
                "organic_recall_6class"
            ),
            "plastic_precision_6class": candidate_metrics.get(
                "plastic_precision_6class"
            ),
            "plastic_recall_6class": candidate_metrics.get(
                "plastic_recall_6class"
            ),
            "total_parameters": parameters,
        }
    )


def refresh_validation_comparison():
    rows = []

    mobilenetv2_path = (
        VALIDATION_REPORT_DIR
        / "mobilenetv2_no_aug_finetuned"
        / "mobilenetv2_no_aug_finetuned_comparison.json"
    )
    if mobilenetv2_path.exists():
        with mobilenetv2_path.open("r", encoding="utf-8") as file:
            mobilenetv2_data = json.load(file)
        v2_metrics = mobilenetv2_data.get("finetuned")
        if v2_metrics:
            add_standard_candidate(
                rows,
                "MobileNetV2",
                "finetuned",
                v2_metrics,
                None,
            )

    candidate_paths = [
        (
            "MobileNetV3Large",
            [
                VALIDATION_REPORT_DIR
                / "mobilenetv3large_no_aug_finetuned_extended"
                / "mobilenetv3large_no_aug_finetuned_extended_metrics.json",
                VALIDATION_REPORT_DIR
                / "mobilenetv3large_no_aug_finetuned"
                / "mobilenetv3large_no_aug_finetuned_metrics.json",
                VALIDATION_REPORT_DIR
                / "mobilenetv3large_no_aug"
                / "mobilenetv3large_no_aug_metrics.json",
            ],
        ),
        (
            "EfficientNet-Lite0",
            [
                VALIDATION_REPORT_DIR
                / "efficientnet_lite0_no_aug"
                / "efficientnet_lite0_no_aug_metrics.json"
            ],
        ),
    ]

    for display_name, metric_paths in candidate_paths:
        available_candidates = []
        for metric_path in metric_paths:
            if not metric_path.exists():
                continue
            with metric_path.open("r", encoding="utf-8") as file:
                available_candidates.append(json.load(file))

        if not available_candidates:
            continue

        # Keep the strongest validation candidate. A continuation that fails
        # to improve cannot displace the earlier checkpoint.
        candidate_data = max(
            available_candidates,
            key=lambda item: item["validation"]["macro_f1_6class"],
        )
        add_standard_candidate(
            rows,
            display_name,
            candidate_data["stage"],
            candidate_data["validation"],
            candidate_data.get("total_parameters"),
        )

    if not rows:
        return

    comparison_df = pd.DataFrame(rows).sort_values(
        "macro_f1_6class",
        ascending=False,
    )
    comparison_path = VALIDATION_REPORT_DIR / "model_validation_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    print("\nAVAILABLE MODEL COMPARISON (VALIDATION ONLY)")
    print(
        comparison_df[
            [
                "candidate",
                "stage",
                "accuracy_9class",
                "macro_f1_9class",
                "accuracy_6class",
                "macro_f1_6class",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"Comparison CSV: {comparison_path}")


print("\n9-class validation report:")
print(report_9_text)
print("ChemSort 6-class validation report:")
print(report_6_text)

print("VALIDATION SUMMARY")
print(f"  9-class Accuracy:    {validation_metrics['accuracy_9class']:.4f}")
print(f"  9-class Macro-F1:    {validation_metrics['macro_f1_9class']:.4f}")
print(f"  6-class Accuracy:    {validation_metrics['accuracy_6class']:.4f}")
print(f"  6-class Macro-F1:    {validation_metrics['macro_f1_6class']:.4f}")
print(f"  Organic recall:      {validation_metrics['organic_recall_6class']:.4f}")
print(f"  Plastic precision:   {validation_metrics['plastic_precision_6class']:.4f}")
print(f"  Plastic recall:      {validation_metrics['plastic_recall_6class']:.4f}")
print(f"  Best epoch:          {best_epoch}")
print(f"  Model artifact:      {model_artifact}")
print(f"  Metrics:             {METRICS_PATH}")
print("  Test set used:       False")

refresh_validation_comparison()
