"""The Python preprocessing must match lib/services/yolo_classifier.dart."""

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chemsort.preprocess import (  # noqa: E402
    FRAMING_SQUASH,
    KERAS_INPUT,
    YOLO_INPUT,
    InputSpec,
    app_preprocess,
    box_resize,
    center_square_crop,
    to_channels_first,
)


def square(side: int, value) -> np.ndarray:
    ys, xs = np.mgrid[0:side, 0:side]
    plane = np.vectorize(value)(xs, ys).astype(np.uint8)
    return np.repeat(plane[:, :, np.newaxis], 3, axis=2)


class BoxResizeTests(unittest.TestCase):
    # The three cases mirror test/yolo_preprocess_test.dart.
    def test_224_input_is_only_scaled(self) -> None:
        rgb = (np.arange(224 * 224 * 3) % 251).astype(np.uint8).reshape(224, 224, 3)
        np.testing.assert_allclose(box_resize(rgb), rgb / 255, atol=1e-6)

    def test_2x_downscale_averages_blocks(self) -> None:
        tensor = box_resize(square(448, lambda x, y: 0 if x % 2 == 0 else 200))
        np.testing.assert_allclose(tensor, 100 / 255, atol=1e-6)

    def test_non_integer_downscale_keeps_ramp_monotonic(self) -> None:
        tensor = box_resize(square(1000, lambda x, y: x * 255 // 999))
        self.assertTrue(np.all(np.diff(tensor[0, :, 0]) >= 0))
        self.assertTrue(np.all((tensor >= 0) & (tensor <= 1)))

    def test_block_edges_match_dart_integer_division(self) -> None:
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, size=(500, 500, 3), dtype=np.uint8)
        edges = [i * 500 // 224 for i in range(225)]
        expected = rgb[edges[7]:edges[8], edges[100]:edges[101]].reshape(-1, 3).mean(axis=0) / 255
        np.testing.assert_allclose(box_resize(rgb)[7, 100], expected, atol=1e-6)

    def test_upscale_uses_nearest_neighbour(self) -> None:
        rgb = square(100, lambda x, y: x)
        tensor = box_resize(rgb)
        self.assertAlmostEqual(float(tensor[0, 223, 0]), (223 * 100 // 224) / 255, places=6)

    def test_rejects_non_square(self) -> None:
        with self.assertRaises(ValueError):
            box_resize(np.zeros((10, 20, 3), dtype=np.uint8))


class FramingTests(unittest.TestCase):
    def test_center_crop_takes_middle_square(self) -> None:
        rgb = np.zeros((100, 300, 3), dtype=np.uint8)
        rgb[:, 100:200] = 255
        cropped = center_square_crop(rgb)
        self.assertEqual(cropped.shape, (100, 100, 3))
        self.assertTrue(np.all(cropped == 255))

    def test_app_preprocess_applies_exif_orientation(self) -> None:
        image = Image.new("RGB", (40, 20), (255, 0, 0))
        image.paste((0, 0, 255), (0, 0, 20, 20))
        exif = image.getexif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
        path = PROJECT_ROOT / "tests" / "_exif_tmp.jpg"
        try:
            image.save(path, exif=exif.tobytes(), quality=100)
            tensor = app_preprocess(path)
            self.assertEqual(tensor.shape, (224, 224, 3))
        finally:
            path.unlink(missing_ok=True)

    def test_channels_first_planes(self) -> None:
        hwc = np.zeros((224, 224, 3), dtype=np.float32)
        hwc[..., 0], hwc[..., 1], hwc[..., 2] = 0.1, 0.5, 0.9
        chw = to_channels_first(hwc)
        self.assertEqual(chw.shape, (3, 224, 224))
        np.testing.assert_allclose(chw[1], 0.5)


class InputSpecTests(unittest.TestCase):
    def test_yolo_and_keras_specs(self) -> None:
        image = Image.new("RGB", (300, 200), (10, 20, 30))
        yolo = YOLO_INPUT.batch([image])
        keras = KERAS_INPUT.batch([image])
        self.assertEqual(yolo.shape, (1, 3, 224, 224))
        self.assertEqual(keras.shape, (1, 224, 224, 3))
        np.testing.assert_allclose(yolo[0, :, 0, 0], np.array([10, 20, 30]) / 255, atol=1e-6)
        np.testing.assert_allclose(keras[0, 0, 0], [10, 20, 30], atol=1e-4)

    def test_mapping_round_trip(self) -> None:
        for spec in (YOLO_INPUT, KERAS_INPUT, InputSpec("NHWC", 255.0, FRAMING_SQUASH)):
            self.assertEqual(InputSpec.from_mapping_input(spec.to_mapping_input()), spec)

    def test_legacy_mapping_means_squash(self) -> None:
        legacy = {"shape": [1, 224, 224, 3], "layout": "NHWC", "pixel_range": [0.0, 255.0], "resize": "224x224"}
        self.assertEqual(InputSpec.from_mapping_input(legacy).framing, FRAMING_SQUASH)

    def test_app_bundle_mapping_matches_yolo_contract(self) -> None:
        import json

        mapping = PROJECT_ROOT.parent / "assets" / "chemsort" / "chemsort_mapping.json"
        if not mapping.is_file():
            self.skipTest("Flutter assets not present")
        block = json.loads(mapping.read_text(encoding="utf-8"))["input"]
        self.assertEqual(InputSpec.from_mapping_input(block), YOLO_INPUT)


if __name__ == "__main__":
    unittest.main()
