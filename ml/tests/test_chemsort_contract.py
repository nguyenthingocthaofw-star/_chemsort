import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from chemsort.contracts import (  # noqa: E402
    APP_CLASS_NAMES,
    APP_GROUPS,
    CHEMSORT_CONTRACT,
    ClassGroup,
    MODEL_CLASS_NAMES,
    ModelContract,
    aggregate_probability_columns,
    model_to_app_ids,
    validate_contract,
)
from validate_chemsort_assets import (  # noqa: E402
    EXPECTED_LABELS,
    EXPECTED_MATERIAL_ORDER,
)


class ChemSortContractTests(unittest.TestCase):
    def test_contract_is_an_immutable_value_object(self) -> None:
        self.assertIsInstance(CHEMSORT_CONTRACT, ModelContract)
        with self.assertRaises(AttributeError):
            CHEMSORT_CONTRACT.model_class_names = ()

    def test_contract_is_internally_consistent(self) -> None:
        validate_contract()

    def test_every_model_output_is_mapped_exactly_once(self) -> None:
        mapped = [label for labels in APP_GROUPS.values() for label in labels]
        self.assertCountEqual(mapped, MODEL_CLASS_NAMES)
        self.assertEqual(len(mapped), len(set(mapped)))

    def test_mapping_preserves_locked_output_order(self) -> None:
        self.assertEqual(
            model_to_app_ids(),
            (0, 1, 2, 3, 4, 0, 5, 4, 4),
        )
        self.assertEqual(APP_CLASS_NAMES[4], "Other")

    def test_rejects_different_model_class_order(self) -> None:
        reversed_names = tuple(reversed(MODEL_CLASS_NAMES))
        with self.assertRaises(ValueError):
            model_to_app_ids(reversed_names)

    def test_portable_asset_validator_matches_shared_contract(self) -> None:
        expected_material_keys = tuple(
            name.lower().replace("/", "_") for name in APP_CLASS_NAMES
        )
        self.assertEqual(tuple(EXPECTED_LABELS), MODEL_CLASS_NAMES)
        self.assertEqual(tuple(EXPECTED_MATERIAL_ORDER), expected_material_keys)

    def test_probability_aggregation_preserves_mass_and_order(self) -> None:
        probabilities = np.asarray(
            [[0.10, 0.05, 0.15, 0.20, 0.03, 0.10, 0.25, 0.04, 0.08]],
            dtype=np.float32,
        )
        aggregated = np.column_stack(
            aggregate_probability_columns(probabilities)
        )
        np.testing.assert_allclose(
            aggregated,
            [[0.20, 0.05, 0.15, 0.20, 0.15, 0.25]],
        )
        np.testing.assert_allclose(aggregated.sum(axis=1), [1.0])

    def test_probability_aggregation_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_probability_columns(np.zeros((1, 6), dtype=np.float32))

    def test_rejects_incomplete_group_mapping(self) -> None:
        with self.assertRaises(ValueError):
            ModelContract(
                model_class_names=("A", "B"),
                groups=(ClassGroup("Only A", ("A",)),),
            )


if __name__ == "__main__":
    unittest.main()
