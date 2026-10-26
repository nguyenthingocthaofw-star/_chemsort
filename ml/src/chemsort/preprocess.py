"""Image preprocessing that matches the Flutter app bit for bit.

``app_preprocess`` ports ``preprocessImage`` / ``boxResize`` from
``lib/services/yolo_classifier.dart``: EXIF orientation, RGB, centre square
crop, area-average resize to 224x224 (nearest neighbour when upscaling),
pixels scaled to 0..1. Evaluating every model with this function means the
reported scores are the scores the phone actually gets.

``InputSpec`` describes how a particular model artifact wants that tensor:
layout (NHWC/NCHW), pixel range (0..1 or 0..255) and framing. The spec is
written into ``chemsort_mapping.json`` next to each exported model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 224

FRAMING_CENTER_CROP = "center_square_crop+area_average"
FRAMING_SQUASH = "squash_bilinear"


def load_rgb(source: str | Path | Image.Image) -> np.ndarray:
    """Decode an image as HWC uint8 RGB with EXIF orientation applied."""
    if isinstance(source, Image.Image):
        image = ImageOps.exif_transpose(source)
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")
    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image)
        return np.asarray(oriented.convert("RGB"), dtype=np.uint8)


def center_square_crop(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return rgb[top : top + side, left : left + side]


def box_resize(square: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Area-average resize of a square RGB image, scaled to 0..1.

    Mirrors Dart ``boxResize``: block edges are ``i * side // size``; when the
    image is not larger than ``size`` it falls back to nearest neighbour.
    """
    side = square.shape[0]
    if square.shape[1] != side:
        raise ValueError(f"Expected a square image, got {square.shape}")
    if side <= size:
        index = np.arange(size) * side // size
        sampled = square[index][:, index]
        return (sampled.astype(np.float32) / 255).astype(np.float32)

    edges = np.arange(size + 1) * side // size
    starts = edges[:-1]
    sums = np.add.reduceat(square.astype(np.float64), starts, axis=0)
    sums = np.add.reduceat(sums, starts, axis=1)
    lengths = np.diff(edges).astype(np.float64)
    area = np.outer(lengths, lengths)[:, :, np.newaxis] * 255.0
    return (sums / area).astype(np.float32)


def app_preprocess(source: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """HWC float32 tensor in 0..1, exactly as the app feeds the model."""
    rgb = source if isinstance(source, np.ndarray) else load_rgb(source)
    return box_resize(center_square_crop(rgb))


def squash_resize(
    source: str | Path | Image.Image | np.ndarray,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Legacy Keras contract: stretch the whole image to size x size, 0..1."""
    rgb = source if isinstance(source, np.ndarray) else load_rgb(source)
    resized = Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def to_channels_first(hwc: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(hwc, (2, 0, 1)))


@dataclass(frozen=True, slots=True)
class InputSpec:
    """How one model artifact expects its input tensor."""

    layout: str = "NHWC"
    pixel_max: float = 1.0
    framing: str = FRAMING_CENTER_CROP
    size: int = IMAGE_SIZE

    def __post_init__(self) -> None:
        if self.layout not in ("NHWC", "NCHW"):
            raise ValueError(f"Unsupported layout: {self.layout}")
        if self.pixel_max not in (1.0, 255.0):
            raise ValueError(f"pixel_max must be 1.0 or 255.0: {self.pixel_max}")
        if self.framing not in (FRAMING_CENTER_CROP, FRAMING_SQUASH):
            raise ValueError(f"Unsupported framing: {self.framing}")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        if self.layout == "NCHW":
            return (1, 3, self.size, self.size)
        return (1, self.size, self.size, 3)

    def tensor(self, source: Any) -> np.ndarray:
        """Single preprocessed image without the batch axis."""
        if self.framing == FRAMING_CENTER_CROP:
            hwc = app_preprocess(source)
        else:
            hwc = squash_resize(source, self.size)
        if self.pixel_max != 1.0:
            hwc = hwc * np.float32(self.pixel_max)
        if self.layout == "NCHW":
            return to_channels_first(hwc)
        return hwc

    def batch(self, sources: list[Any]) -> np.ndarray:
        return np.stack([self.tensor(source) for source in sources]).astype(
            np.float32
        )

    def to_mapping_input(self) -> dict[str, Any]:
        """``input`` block of ``chemsort_mapping.json``."""
        return {
            "shape": list(self.shape),
            "layout": self.layout,
            "color_order": "RGB",
            "dtype": "float32",
            "pixel_range": [0.0, float(self.pixel_max)],
            "resize": self.framing,
            "note": (
                "Pixels scaled to 0..1 before inference."
                if self.pixel_max == 1.0
                else "Do not divide by 255; preprocessing is embedded."
            ),
        }

    @classmethod
    def from_mapping_input(cls, block: dict[str, Any]) -> "InputSpec":
        shape = [int(value) for value in block.get("shape", [])]
        layout = block.get("layout") or (
            "NCHW" if len(shape) == 4 and shape[1] == 3 else "NHWC"
        )
        pixel_range = block.get("pixel_range", [0.0, 255.0])
        framing = block.get("resize")
        if framing not in (FRAMING_CENTER_CROP, FRAMING_SQUASH):
            framing = FRAMING_SQUASH
        # Index 2 is a spatial axis in both [1,3,H,W] and [1,H,W,3].
        size = shape[2] if len(shape) == 4 else IMAGE_SIZE
        return cls(
            layout=layout,
            pixel_max=float(pixel_range[1]),
            framing=framing,
            size=int(size),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputSpec":
        return cls(**data)


# Ultralytics YOLO-cls: no mean/std normalisation, 0..1, channels-first.
YOLO_INPUT = InputSpec(layout="NCHW", pixel_max=1.0)
# Keras models embed their own normalisation and take raw 0..255 pixels.
KERAS_INPUT = InputSpec(layout="NHWC", pixel_max=255.0)
# Models exported by the original TensorFlow-only pipeline.
LEGACY_KERAS_INPUT = InputSpec(layout="NHWC", pixel_max=255.0, framing=FRAMING_SQUASH)
