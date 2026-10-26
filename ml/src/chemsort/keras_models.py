"""Keras transfer-learning candidates with a raw RGB [0, 255] input contract.

Every model embeds its own normalisation, so callers (training, export and
the app) feed the same 224x224 RGB tensor. All share one classifier head:
global average pooling, dropout 0.3, softmax over the 9 RealWaste classes.

TensorFlow is imported inside the functions so this module stays importable
for tests and for the YOLO environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import CHEMSORT_CONTRACT
from .preprocess import IMAGE_SIZE

RUN_CONFIG_FILE = "train_config.json"
WEIGHTS_FILE = "best.weights.h5"
KERAS_FILE = "best.keras"


@dataclass(frozen=True, slots=True)
class KerasCandidate:
    key: str
    display_name: str
    preprocessing: str
    fine_tune_supported: bool = True
    needs_legacy_keras: bool = False


KERAS_CANDIDATES: dict[str, KerasCandidate] = {
    candidate.key: candidate
    for candidate in (
        KerasCandidate("mobilenetv2", "MobileNetV2", "[0,255] -> [-1,1] rescaling"),
        KerasCandidate("mobilenetv3small", "MobileNetV3Small", "[0,255] -> [-1,1] rescaling"),
        KerasCandidate("mobilenetv3large", "MobileNetV3Large", "[0,255] -> [-1,1] rescaling"),
        KerasCandidate("efficientnetb0", "EfficientNetB0", "built-in rescaling + ImageNet normalisation"),
        KerasCandidate("efficientnetv2b0", "EfficientNetV2B0", "built-in [0,255] -> [-1,1]"),
        KerasCandidate(
            "efficientnet_lite0",
            "EfficientNet-Lite0",
            "[0,255] -> [0,1] rescaling (TF Hub)",
            fine_tune_supported=False,
            needs_legacy_keras=True,
        ),
    )
}

EFFICIENTNET_LITE0_URL = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"


def _backbone(tf: Any, key: str) -> tuple[Any, Any]:
    """Return (preprocessed_input_fn, backbone) for a candidate key."""
    shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    applications = tf.keras.applications
    to_minus_one = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="to_minus_one_one")
    if key == "mobilenetv2":
        return to_minus_one, applications.MobileNetV2(input_shape=shape, include_top=False, weights="imagenet")
    if key == "mobilenetv3small":
        return to_minus_one, applications.MobileNetV3Small(
            input_shape=shape, include_top=False, weights="imagenet", include_preprocessing=False
        )
    if key == "mobilenetv3large":
        return to_minus_one, applications.MobileNetV3Large(
            input_shape=shape, include_top=False, weights="imagenet", include_preprocessing=False
        )
    if key == "efficientnetb0":
        return None, applications.EfficientNetB0(input_shape=shape, include_top=False, weights="imagenet")
    if key == "efficientnetv2b0":
        return None, applications.EfficientNetV2B0(
            input_shape=shape, include_top=False, weights="imagenet", include_preprocessing=True
        )
    raise KeyError(key)


def build_model(key: str, dropout: float = 0.30) -> tuple[Any, Any, dict[str, Any]]:
    """Return ``(model, backbone, metadata)``; the backbone starts frozen."""
    import tensorflow as tf

    if key not in KERAS_CANDIDATES:
        raise SystemExit(f"Unknown Keras model {key!r}; choose from {sorted(KERAS_CANDIDATES)}")
    candidate = KERAS_CANDIDATES[key]
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")

    if key == "efficientnet_lite0":
        from .tfhub_compat import import_tensorflow_hub

        hub = import_tensorflow_hub()
        x = tf.keras.layers.Rescaling(1.0 / 255.0, name="to_zero_one")(inputs)
        backbone = hub.KerasLayer(
            EFFICIENTNET_LITE0_URL,
            trainable=False,
            output_shape=[1280],
            name="efficientnet_lite0_feature_vector",
        )
        features = backbone(x)
    else:
        preprocess, backbone = _backbone(tf, key)
        backbone.trainable = False
        x = preprocess(inputs) if preprocess is not None else inputs
        x = backbone(x, training=False)
        features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)

    features = tf.keras.layers.Dropout(dropout, name="dropout")(features)
    outputs = tf.keras.layers.Dense(
        len(CHEMSORT_CONTRACT.model_class_names), activation="softmax", name="predictions"
    )(features)
    model = tf.keras.Model(inputs, outputs, name=f"chemsort_{key}")
    metadata = {
        "key": key,
        "display_name": candidate.display_name,
        "preprocessing": f"embedded {candidate.preprocessing}",
        "pretrained_weights": EFFICIENTNET_LITE0_URL if key == "efficientnet_lite0" else "ImageNet (keras.applications)",
        "fine_tune_supported": candidate.fine_tune_supported,
    }
    return model, backbone, metadata


def unfreeze_top_layers(tf: Any, backbone: Any, count: int) -> list[str]:
    """Unfreeze the last ``count`` backbone layers; BatchNorm stays frozen."""
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-count:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    return [layer.name for layer in backbone.layers if layer.trainable]


def read_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / RUN_CONFIG_FILE
    if not path.is_file():
        raise SystemExit(f"Run config not found: {path}. Train the model first.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_run_config(run_dir: Path, config: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_CONFIG_FILE
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_trained_model(run_dir: Path) -> Any:
    """Rebuild the architecture from the run config and load best weights.

    Rebuilding avoids depending on full-model serialisation, which does not
    work for the TF Hub EfficientNet-Lite0 layer.
    """
    config = read_run_config(run_dir)
    if config.get("backend") != "keras":
        raise SystemExit(f"{run_dir.name} is not a Keras run")
    model, _, _ = build_model(config["model"])
    weights = run_dir / WEIGHTS_FILE
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    model.load_weights(str(weights))
    return model
