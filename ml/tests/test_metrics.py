import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chemsort.metrics import (  # noqa: E402
    bootstrap_intervals,
    classification_metrics,
    evaluate_probabilities,
    expected_calibration_error,
    format_classification_report,
    labels_to_app,
    mcnemar_test,
    normalize_outputs,
)


def one_hot(labels, classes=9, confidence=0.9):
    probabilities = np.full((len(labels), classes), (1 - confidence) / (classes - 1))
    probabilities[np.arange(len(labels)), labels] = confidence
    return probabilities


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        labels = np.repeat(np.arange(9), 5)
        result = evaluate_probabilities(labels, one_hot(labels))
        for key in ("accuracy_9class", "macro_f1_9class", "accuracy_6class", "macro_f1_6class", "mcc_6class"):
            self.assertAlmostEqual(result["summary"][key], 1.0)
        self.assertEqual(np.trace(result["6class"]["confusion_matrix"]), 45)

    def test_matches_sklearn_macro_f1(self) -> None:
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 9, 300)
        probabilities = rng.dirichlet(np.ones(9), 300)
        block = classification_metrics(labels, probabilities, [str(i) for i in range(9)])
        expected = f1_score(labels, probabilities.argmax(1), labels=range(9), average="macro", zero_division=0)
        self.assertAlmostEqual(block["macro_f1"], expected)

    def test_six_class_sums_group_probabilities(self) -> None:
        # Paper 0.25 + Cardboard 0.25 beats Plastic 0.40 at the 6-group level.
        probabilities = np.array([[0.25, 0.01, 0.01, 0.01, 0.01, 0.25, 0.40, 0.03, 0.03]])
        result = evaluate_probabilities([5], probabilities)
        self.assertEqual(result["summary"]["accuracy_9class"], 0.0)
        self.assertEqual(result["summary"]["accuracy_6class"], 1.0)

    def test_absent_class_lowers_macro_but_not_present_macro(self) -> None:
        labels = np.repeat(np.arange(8), 4)  # no Vegetation, like test_ext
        block = classification_metrics(labels, one_hot(labels), [str(i) for i in range(9)])
        self.assertAlmostEqual(block["macro_f1"], 8 / 9)
        self.assertAlmostEqual(block["macro_f1_present"], 1.0)
        self.assertEqual(block["classes_present"], 8)
        self.assertIn("only 8/9 classes", format_classification_report(block))

    def test_specificity(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = one_hot(np.array([0, 1, 1, 1]), classes=2)
        block = classification_metrics(labels, probabilities, ["a", "b"])
        self.assertAlmostEqual(block["per_class"]["a"]["specificity"], 1.0)
        self.assertAlmostEqual(block["per_class"]["b"]["specificity"], 0.5)

    def test_normalize_applies_softmax_to_logits(self) -> None:
        probabilities = normalize_outputs(np.array([[2.0, 1.0, 0.0]]))
        np.testing.assert_allclose(probabilities.sum(), 1.0)
        self.assertEqual(probabilities.argmax(), 0)

    def test_labels_to_app(self) -> None:
        np.testing.assert_array_equal(labels_to_app(range(9)), [0, 1, 2, 3, 4, 0, 5, 4, 4])

    def test_ece_zero_when_confidence_equals_accuracy(self) -> None:
        labels = np.array([0, 0, 0, 0])
        probabilities = np.array([[1.0, 0.0]] * 4)
        self.assertAlmostEqual(expected_calibration_error(probabilities, labels), 0.0)

    def test_bootstrap_is_deterministic_and_brackets_estimate(self) -> None:
        rng = np.random.default_rng(3)
        labels = rng.integers(0, 6, 400)
        predictions = np.where(rng.random(400) < 0.8, labels, rng.integers(0, 6, 400))
        first = bootstrap_intervals(labels, predictions, 6, samples=300)
        second = bootstrap_intervals(labels, predictions, 6, samples=300)
        self.assertEqual(first, second)
        accuracy = float(np.mean(labels == predictions))
        self.assertLess(first["accuracy"]["lower"], accuracy)
        self.assertGreater(first["accuracy"]["upper"], accuracy)

    def test_mcnemar(self) -> None:
        same = mcnemar_test([True, False, True], [True, False, True])
        self.assertEqual(same["p_value"], 1.0)
        a = np.array([True] * 60 + [False] * 40)
        b = np.array([True] * 30 + [False] * 70)
        result = mcnemar_test(a, b)
        self.assertEqual(result["only_a_correct"], 30)
        self.assertEqual(result["method"], "chi2_continuity")
        self.assertLess(result["p_value"], 0.001)


if __name__ == "__main__":
    unittest.main()
