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

from chemsort.contracts import MODEL_CLASS_NAMES  # noqa: E402
from chemsort.data import load_manifest, select_split, stratified_sample  # noqa: E402
from chemsort.evaluation import write_evaluation_report  # noqa: E402
from chemsort.yolo_data import build_classification_tree, tree_class_order  # noqa: E402
from tools.build_dataset_v2 import (  # noqa: E402
    cap_per_group,
    exact_dedup,
    holdout_split,
    min_hamming_distance,
)


class ManifestTests(unittest.TestCase):
    def test_v1_manifest(self) -> None:
        frame = load_manifest("v1")
        self.assertEqual(len(frame), 4752)
        self.assertEqual(frame["split"].value_counts().to_dict(), {"train": 3326, "val": 713, "test": 713})
        self.assertTrue(frame["rel_path"].str.startswith("RealWaste/").all())
        self.assertEqual(set(frame["source"]), {"realwaste"})

    def test_v2_manifest_matches_published_counts(self) -> None:
        frame = load_manifest("v2")
        self.assertEqual(
            frame["split"].value_counts().to_dict(),
            {"train": 8717, "val": 713, "test": 713, "test_ext": 594},
        )
        self.assertEqual(set(select_split(frame, "val")["source"]), {"realwaste"})
        self.assertNotIn("Vegetation", set(select_split(frame, "test_ext")["label"]))
        self.assertTrue(frame["image_id"].str.match(r"^[a-z_]+_[0-9a-f]{16}\.").all())

    def test_v1_and_v2_share_val_and_test(self) -> None:
        v1, v2 = load_manifest("v1"), load_manifest("v2")
        for split in ("val", "test"):
            self.assertEqual(set(select_split(v1, split)["rel_path"]), set(select_split(v2, split)["rel_path"]))

    def test_label_ids_follow_contract(self) -> None:
        frame = load_manifest("v2")
        for name, index in zip(MODEL_CLASS_NAMES, range(9)):
            self.assertTrue((frame.loc[frame["label"] == name, "label_id"] == index).all())

    def test_stratified_sample(self) -> None:
        sample = stratified_sample(select_split(load_manifest("v2"), "val"), 3)
        self.assertEqual(len(sample), 27)
        self.assertEqual(sample["label"].value_counts().max(), 3)


def write_images(root: Path, frame: pd.DataFrame) -> None:
    for path in frame["abs_path"]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), (120, 60, 30)).save(path)


class YoloTreeAndReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        frame = load_manifest("v2", raw_dir=self.root / "raw")
        parts = [group.head(2) for _, group in frame.groupby(["split", "label"])]
        self.frame = pd.concat(parts).reset_index(drop=True)
        self.frame.attrs.update(frame.attrs)
        write_images(self.root, self.frame)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_tree_has_all_classes_and_is_idempotent(self) -> None:
        output = self.root / "tree"
        stamp = build_classification_tree(self.frame, output)
        self.assertEqual(tree_class_order(output), MODEL_CLASS_NAMES)
        self.assertTrue((output / "test_ext" / "Vegetation").is_dir())
        self.assertEqual(sum(stamp["counts"]["val"].values()), 18)
        self.assertEqual(build_classification_tree(self.frame, output)["manifest_digest"], stamp["manifest_digest"])
        with self.assertRaises(SystemExit):
            build_classification_tree(self.frame.iloc[:-1], output)

    def test_evaluation_report_files(self) -> None:
        subset = select_split(self.frame, "val")
        rng = np.random.default_rng(0)
        probabilities = rng.dirichlet(np.ones(9), len(subset))
        document = write_evaluation_report(self.root / "report", "unit", subset, probabilities, {"run_name": "unit"},
                                           bootstrap_samples=50)
        for name in ("metrics.json", "predictions.csv", "report_6class.txt", "confusion_matrix_9class.png",
                     "confusion_matrix_6class.png", "per_class_6class.png", "reliability_6class.png"):
            self.assertTrue((self.root / "report" / name).is_file(), name)
        self.assertEqual(document["n_images"], len(subset))
        predictions = pd.read_csv(self.root / "report" / "predictions.csv")
        self.assertEqual(predictions["correct_6"].mean(), document["summary"]["accuracy_6class"])


class DatasetV2BuilderTests(unittest.TestCase):
    def test_min_hamming_distance(self) -> None:
        queries = np.array([0b1111, 0], dtype=np.uint64)
        references = np.array([0b0111, 2**63], dtype=np.uint64)
        np.testing.assert_array_equal(min_hamming_distance(queries, references), [1, 1])

    def test_exact_dedup_drops_base_and_cross_source(self) -> None:
        external = pd.DataFrame({
            "path": ["b/1", "a/1", "a/2"], "source": ["garbage_classification", "trashnet", "trashnet"],
            "sha256": ["x", "x", "base"],
        })
        result = exact_dedup(external, {"base"})
        self.assertEqual(result["path"].tolist(), ["a/1"])  # TrashNet wins, base duplicate dropped

    def test_cap_and_holdout(self) -> None:
        frame = pd.DataFrame({
            "path": [f"p{i:03d}" for i in range(500)], "source": "trashnet", "original_label": "Glass",
        })
        capped = cap_per_group(frame, 400, 42)
        self.assertEqual(len(capped), 400)
        split = holdout_split(capped, 0.10, 42)
        self.assertEqual((split["split"] == "test_ext").sum(), 40)
        self.assertEqual(holdout_split(capped, 0.10, 42)["split"].tolist(), split["split"].tolist())


if __name__ == "__main__":
    unittest.main()
