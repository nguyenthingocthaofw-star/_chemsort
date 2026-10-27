"""Framework adapters that turn image paths into 9-class probabilities.

All adapters share ``InputSpec`` preprocessing, so a Keras model, a PyTorch
YOLO checkpoint and their TFLite exports are scored on identical tensors.
TensorFlow and PyTorch are imported lazily: the TF and YOLO environments can
be separate.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .contracts import CHEMSORT_CONTRACT
from .metrics import normalize_outputs
from .preprocess import YOLO_INPUT, InputSpec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Predictor(Protocol):
    input_spec: InputSpec
    description: dict[str, Any]

    def predict_batch(self, batch: np.ndarray) -> np.ndarray: ...


def predict_paths(
    predictor: Predictor,
    paths: Sequence[str],
    batch_size: int = 32,
    progress_every: int = 200,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run a predictor over files; returns probabilities and timing stats."""
    outputs: list[np.ndarray] = []
    inference_seconds = 0.0
    preprocess_seconds = 0.0
    done = 0
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        begin = time.perf_counter()
        batch = predictor.input_spec.batch(list(chunk))
        preprocess_seconds += time.perf_counter() - begin
        begin = time.perf_counter()
        outputs.append(np.asarray(predictor.predict_batch(batch), dtype=np.float64))
        inference_seconds += time.perf_counter() - begin
        previous = done
        done += len(chunk)
        if progress_every and (done // progress_every > previous // progress_every or done == len(paths)):
            print(f"  {done}/{len(paths)} images")
    probabilities = normalize_outputs(np.concatenate(outputs, axis=0))
    expected = (len(paths), len(CHEMSORT_CONTRACT.model_class_names))
    if probabilities.shape != expected:
        raise ValueError(f"Predictions have shape {probabilities.shape}, expected {expected}")
    timing = {
        "inference_ms_per_image": 1000 * inference_seconds / max(1, len(paths)),
        "preprocess_ms_per_image": 1000 * preprocess_seconds / max(1, len(paths)),
        "batch_size": batch_size,
    }
    return probabilities, timing


# ---------------------------------------------------------------------------
# TFLite
# ---------------------------------------------------------------------------


def make_tflite_interpreter(model_path: Path, threads: int = 4) -> Any:
    """Prefer TensorFlow's interpreter, fall back to LiteRT / tflite_runtime."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    errors = []
    try:
        import tensorflow as tf  # type: ignore

        tf.get_logger().setLevel("ERROR")
        return tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
    except ImportError as error:
        errors.append(str(error))
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=str(model_path), num_threads=threads)
    except ImportError as error:
        errors.append(str(error))
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=str(model_path), num_threads=threads)
    except ImportError as error:
        errors.append(str(error))
    raise RuntimeError(
        "No TFLite interpreter available. Install one of: tensorflow, "
        "ai-edge-litert, tflite-runtime.\n" + "\n".join(errors)
    )


def infer_tflite_spec(shape: Sequence[int], pixel_max: float | None) -> InputSpec:
    """Layout comes from the tensor; pixel range must come from the exporter."""
    layout = "NCHW" if len(shape) == 4 and int(shape[1]) == 3 else "NHWC"
    if pixel_max is None:
        # YOLO exports are 0..1; Keras exports embed preprocessing and take 0..255.
        pixel_max = 1.0 if layout == "NCHW" else 255.0
    return InputSpec(layout=layout, pixel_max=pixel_max, size=int(shape[2]))


class TFLitePredictor:
    def __init__(
        self,
        model_path: Path,
        input_spec: InputSpec | None = None,
        pixel_max: float | None = None,
        threads: int = 4,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")
        self._interpreter = make_tflite_interpreter(self.model_path, threads)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        shape = [int(value) for value in self._input["shape"]]
        self.input_spec = input_spec or infer_tflite_spec(shape, pixel_max)
        if list(self.input_spec.shape) != shape:
            raise ValueError(
                f"Input spec {self.input_spec.shape} does not match tensor {shape}"
            )
        output_size = int(np.prod(self._output["shape"]))
        if output_size != len(CHEMSORT_CONTRACT.model_class_names):
            raise ValueError(f"TFLite output has {output_size} values, expected 9")
        self.description = {
            "backend": "tflite",
            "model_path": str(self.model_path),
            "model_sha256": sha256_file(self.model_path),
            "model_size_mb": self.model_path.stat().st_size / (1024 * 1024),
            "input_shape": shape,
            "input_dtype": np.dtype(self._input["dtype"]).name,
            "output_shape": [int(value) for value in self._output["shape"]],
            "output_dtype": np.dtype(self._output["dtype"]).name,
            "threads": threads,
        }

    def _quantize(self, value: np.ndarray) -> np.ndarray:
        dtype = np.dtype(self._input["dtype"])
        if np.issubdtype(dtype, np.integer):
            scale, zero = self._input.get("quantization", (0.0, 0))
            if not scale:
                raise ValueError("Quantized input tensor has scale 0")
            limits = np.iinfo(dtype)
            value = np.clip(np.round(value / scale + zero), limits.min, limits.max)
        return value.astype(dtype)

    def _dequantize(self, value: np.ndarray) -> np.ndarray:
        if np.issubdtype(np.dtype(self._output["dtype"]), np.integer):
            scale, zero = self._output.get("quantization", (0.0, 0))
            if not scale:
                raise ValueError("Quantized output tensor has scale 0")
            return (value.astype(np.float32) - zero) * scale
        return value.astype(np.float32)

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        rows = []
        for item in batch:
            self._interpreter.set_tensor(self._input["index"], self._quantize(item[np.newaxis]))
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output["index"]).reshape(-1)
            rows.append(self._dequantize(output))
        return np.stack(rows)


# ---------------------------------------------------------------------------
# Keras
# ---------------------------------------------------------------------------


class KerasPredictor:
    def __init__(self, model: Any, input_spec: InputSpec, description: dict[str, Any]) -> None:
        self._model = model
        self.input_spec = input_spec
        self.description = {"backend": "keras", **description}

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray(self._model(batch, training=False))


# ---------------------------------------------------------------------------
# Ultralytics YOLO (PyTorch)
# ---------------------------------------------------------------------------


def yolo_class_names(names: dict[int, str] | Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, dict):
        return tuple(names[index] for index in sorted(names))
    return tuple(names)


def torch_device_name(device: str | None, cuda_available: bool) -> str:
    """Translate Ultralytics-style device strings ("0", "0,1", "cpu")."""
    if device in (None, ""):
        return "cuda" if cuda_available else "cpu"
    device = str(device)
    if device.split(",")[0].isdigit():
        return f"cuda:{device.split(',')[0]}"
    return device


class TorchYoloPredictor:
    """Runs the raw Ultralytics classification network on our own tensors.

    Going through ``model.model`` instead of ``YOLO.predict`` keeps the input
    identical to the app (centre crop + area resize, 0..1) and avoids
    Ultralytics' own resize from changing the scores.
    """

    def __init__(self, weights: Path, device: str | None = None) -> None:
        import torch
        from ultralytics import YOLO

        self._torch = torch
        self.weights = Path(weights).resolve()
        yolo = YOLO(str(self.weights))
        names = yolo_class_names(yolo.names)
        CHEMSORT_CONTRACT.require_model_order(names)
        self._device = torch.device(torch_device_name(device, torch.cuda.is_available()))
        self._network = yolo.model.float().to(self._device).eval()
        self.input_spec = YOLO_INPUT
        self.description = {
            "backend": "pytorch",
            "model_path": str(self.weights),
            "model_sha256": sha256_file(self.weights),
            "model_size_mb": self.weights.stat().st_size / (1024 * 1024),
            "parameters": int(sum(p.numel() for p in self._network.parameters())),
            "device": str(self._device),
        }

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(np.ascontiguousarray(batch)).to(self._device)
            output = self._network(tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            return output.float().cpu().numpy()
