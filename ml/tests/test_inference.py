import unittest
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chemsort.inference import TFLiteImageClassifier


class TFLiteImageClassifierTests(unittest.TestCase):
    def test_preprocess_matches_model_input_contract(self) -> None:
        image = Image.new("RGB", (32, 16), color=(10, 20, 30))
        result = TFLiteImageClassifier.preprocess_image(image)

        self.assertEqual(result.shape, (1, 224, 224, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[0, 0, 0], [10.0, 20.0, 30.0])

    def test_preprocess_rejects_missing_image(self) -> None:
        with self.assertRaises(FileNotFoundError):
            TFLiteImageClassifier.preprocess(Path("missing-image.jpg"))


if __name__ == "__main__":
    unittest.main()
