"""On-device-compatible single-image classification for TFLite artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .contracts import CHEMSORT_CONTRACT
from .metrics import aggregate_to_app, normalize_outputs
from .predictors import TFLitePredictor
from .preprocess import InputSpec


@dataclass(frozen=True, slots=True)
class Prediction:
    material: str
    confidence: float
    model_class: str
    model_confidence: float
    probabilities_6: tuple[float, ...]


class TFLiteImageClassifier:
    """Load a ChemSort TFLite model and classify one image at a time.

    The input contract comes from the ``input`` block of
    ``chemsort_mapping.json`` when given; otherwise it is inferred from the
    tensor layout (NCHW → YOLO 0..1, NHWC → Keras 0..255).
    """

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        *,
        mapping_path: Path | None = None,
        num_threads: int = 4,
    ) -> None:
        self._labels = self._read_labels(labels_path)
        CHEMSORT_CONTRACT.require_model_order(self._labels)
        spec = None
        if mapping_path is not None:
            mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
            if "input" in mapping:
                spec = InputSpec.from_mapping_input(mapping["input"])
        self._predictor = TFLitePredictor(model_path, input_spec=spec, threads=max(1, num_threads))

    @property
    def input_spec(self) -> InputSpec:
        return self._predictor.input_spec

    @staticmethod
    def _read_labels(path: Path) -> tuple[str, ...]:
        if not path.is_file():
            raise FileNotFoundError(f"Labels file not found: {path.resolve()}")
        return tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )

    @staticmethod
    def preprocess_image(image: Image.Image) -> np.ndarray:
        """Legacy Keras contract: stretched 224x224 RGB, raw 0..255, NHWC."""
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
        return np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

    @classmethod
    def preprocess(cls, image_path: Path) -> np.ndarray:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path.resolve()}")
        with Image.open(image_path) as image:
            return cls.preprocess_image(image)

    def predict(self, image_path: Path) -> Prediction:
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image not found: {Path(image_path).resolve()}")
        batch = self.input_spec.batch([image_path])
        probabilities_9 = normalize_outputs(self._predictor.predict_batch(batch))
        probabilities_6 = aggregate_to_app(probabilities_9)[0]
        probabilities_9 = probabilities_9[0]
        model_id = int(np.argmax(probabilities_9))
        material_id = int(np.argmax(probabilities_6))
        return Prediction(
            material=CHEMSORT_CONTRACT.app_class_names[material_id],
            confidence=float(probabilities_6[material_id]),
            model_class=self._labels[model_id],
            model_confidence=float(probabilities_9[model_id]),
            probabilities_6=tuple(float(value) for value in probabilities_6),
        )
