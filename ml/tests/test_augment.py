import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from chemsort.augment import (  # noqa: E402
    PRESETS,
    YOLO_PRESETS,
    augment_image,
    inscribed_scale,
    limit_size,
    yolo_hyperparameters,
)
from chemsort.contracts import MODEL_CLASS_NAMES  # noqa: E402
from chemsort.data import dataset_tag, load_manifest  # noqa: E402
from tools.augment_dataset import job_seed, plan_jobs  # noqa: E402


def gradient_image(width: int = 160, height: int = 120) -> Image.Image:
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    array = np.stack([np.broadcast_to(x, (height, width)), np.broadcast_to(y, (height, width)),
                      np.full((height, width), 128.0)], axis=-1)
    return Image.fromarray(array.astype(np.uint8))


class AugmentImageTests(unittest.TestCase):
    def test_presets_keep_size_and_mode(self) -> None:
        image = gradient_image()
        for name in PRESETS:
            output = augment_image(image, name, np.random.default_rng(0))
            self.assertEqual(output.size, image.size, name)
            self.assertEqual(output.mode, "RGB", name)

    def test_same_seed_is_deterministic(self) -> None:
        image = gradient_image()
        first = np.asarray(augment_image(image, "strong", np.random.default_rng(7)))
        second = np.asarray(augment_image(image, "strong", np.random.default_rng(7)))
        third = np.asarray(augment_image(image, "strong", np.random.default_rng(8)))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))

    def test_none_preset_is_identity(self) -> None:
        image = gradient_image()
        output = augment_image(image, "none", np.random.default_rng(0))
        np.testing.assert_array_equal(np.asarray(output), np.asarray(image))

    def test_rotation_leaves_no_empty_corners(self) -> None:
        # A white image rotated without the inscribed crop would show black corners.
        white = Image.new("RGB", (200, 150), (255, 255, 255))
        config = PRESETS["strong"].__class__("rot", hflip=0.0, rotate_degrees=30, zoom=0.2, translate=0.12)
        for seed in range(20):
            output = np.asarray(augment_image(white, config, np.random.default_rng(seed)))
            self.assertGreater(output.min(), 200, f"seed {seed}")

    def test_inscribed_scale(self) -> None:
        self.assertAlmostEqual(inscribed_scale(100, 100, 0), 1.0)
        self.assertAlmostEqual(inscribed_scale(100, 100, 45), 1 / np.sqrt(2), places=6)
        self.assertLess(inscribed_scale(200, 100, 10), 1.0)

    def test_limit_size(self) -> None:
        self.assertEqual(limit_size(gradient_image(1000, 500), 512).size, (512, 256))
        self.assertEqual(limit_size(gradient_image(100, 50), 512).size, (100, 50))

    def test_yolo_medium_matches_ultralytics_defaults(self) -> None:
        self.assertEqual(
            yolo_hyperparameters("medium"),
            {"fliplr": 0.5, "flipud": 0.0, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
             "erasing": 0.4, "auto_augment": "randaugment"},
        )
        self.assertEqual(set(YOLO_PRESETS), set(PRESETS))


class AugmentedManifestTests(unittest.TestCase):
    def make_train(self, counts: dict[str, int]) -> pd.DataFrame:
        rows = []
        for label, count in counts.items():
            for index in range(count):
                rows.append({"label": label, "abs_path": f"/x/{label}/{index}.jpg",
                             "rel_path": f"{label}/{index}.jpg", "source": "realwaste",
                             "sha256": f"{label}{index}".ljust(64, "0")})
        return pd.DataFrame(rows)

    def test_balance_plan_tops_up_each_class(self) -> None:
        counts = {name: 3 for name in MODEL_CLASS_NAMES}
        counts["Plastic"] = 10
        counts["Vegetation"] = 4
        args = argparse.Namespace(seed=1, target_per_class=None, mode="balance", copies=1, limit_per_class=None)
        jobs = pd.DataFrame(plan_jobs(self.make_train(counts), args))
        added = jobs["label"].value_counts()
        self.assertNotIn("Plastic", added)
        self.assertEqual(added["Vegetation"], 6)
        self.assertEqual(added["Glass"], 7)
        # Parents are reused round-robin: 7 copies over 3 parents -> 3/2/2.
        glass = jobs.loc[jobs["label"] == "Glass"]
        self.assertEqual(sorted(glass["rel_path"].value_counts().tolist()), [2, 2, 3])
        self.assertFalse(jobs.duplicated(["rel_path", "copy"]).any())

    def test_multiply_plan(self) -> None:
        counts = {name: 2 for name in MODEL_CLASS_NAMES}
        args = argparse.Namespace(seed=1, target_per_class=None, mode="multiply", copies=2, limit_per_class=None)
        self.assertEqual(len(plan_jobs(self.make_train(counts), args)), 2 * 2 * len(MODEL_CLASS_NAMES))

    def test_job_seed_is_stable(self) -> None:
        self.assertEqual(job_seed(42, "abc", 0), job_seed(42, "abc", 0))
        self.assertNotEqual(job_seed(42, "abc", 0), job_seed(42, "abc", 1))

    def write_manifest(self, folder: Path, augmented_split: str) -> Path:
        rows = []
        for index, label in enumerate(MODEL_CLASS_NAMES):
            rows.append({"path": f"Src/{label}/{index}.jpg", "original_label": label, "split": "train",
                         "source": "realwaste", "source_class": label, "sha256": f"{index}".ljust(64, "a"), "base": "raw"})
        rows.append({"path": "augmented/v2_medium/images/Glass/x_a0.jpg", "original_label": "Glass",
                     "split": augmented_split, "source": "augmented", "source_class": "realwaste",
                     "sha256": "f" * 64, "base": "data"})
        path = folder / "manifest_v2_aug_medium.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_load_augmented_manifest_resolves_bases(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            raw_dir = folder / "data" / "raw"
            frame = load_manifest(self.write_manifest(folder, "train"), raw_dir=raw_dir)
            augmented = frame.loc[frame["source"] == "augmented", "abs_path"].iloc[0]
            original = frame.loc[frame["source"] == "realwaste", "abs_path"].iloc[0]
            self.assertEqual(Path(augmented), raw_dir.parent / "augmented/v2_medium/images/Glass/x_a0.jpg")
            self.assertTrue(Path(original).is_relative_to(raw_dir))
            self.assertEqual(frame.attrs["dataset"], "v2_aug_medium")

    def test_augmented_rows_outside_train_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "outside the train split"):
                load_manifest(self.write_manifest(Path(temp), "val"))

    def test_dataset_tag_strips_prefix(self) -> None:
        self.assertEqual(dataset_tag(Path("x/manifest_v3.csv")), "v3")
        self.assertEqual(dataset_tag(Path("x/custom.csv")), "custom")


if __name__ == "__main__":
    unittest.main()
