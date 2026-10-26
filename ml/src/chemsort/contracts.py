"""Stable model-to-app contract shared by training, export, and evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ClassGroup:
    """One Flutter-facing class and the model outputs aggregated into it."""

    app_name: str
    model_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.app_name.strip():
            raise ValueError("app_name must not be blank")
        if not self.model_names:
            raise ValueError(f"{self.app_name} must contain a model class")


@dataclass(frozen=True, slots=True)
class ModelContract:
    """Immutable ordering and aggregation contract exported to Flutter."""

    model_class_names: tuple[str, ...]
    groups: tuple[ClassGroup, ...]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def app_class_names(self) -> tuple[str, ...]:
        return tuple(group.app_name for group in self.groups)

    @property
    def app_groups(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {group.app_name: group.model_names for group in self.groups}
        )

    def validate(self) -> None:
        if not self.model_class_names:
            raise ValueError("model_class_names must not be empty")
        if len(self.model_class_names) != len(set(self.model_class_names)):
            raise ValueError("model_class_names contains duplicates")
        if len(self.app_class_names) != len(set(self.app_class_names)):
            raise ValueError("app class names contain duplicates")

        mapped = [name for group in self.groups for name in group.model_names]
        if len(mapped) != len(set(mapped)):
            raise ValueError("A model class is mapped more than once")
        if set(mapped) != set(self.model_class_names):
            missing = sorted(set(self.model_class_names) - set(mapped))
            unexpected = sorted(set(mapped) - set(self.model_class_names))
            raise ValueError(
                "Groups do not cover the model contract; "
                f"missing={missing}, unexpected={unexpected}"
            )

    def require_model_order(self, names: Sequence[str]) -> None:
        if tuple(names) != self.model_class_names:
            raise ValueError(
                "Model class order differs from the locked ChemSort contract"
            )

    def model_to_app_ids(
        self,
        model_class_names: Sequence[str] | None = None,
    ) -> tuple[int, ...]:
        names = (
            self.model_class_names
            if model_class_names is None
            else model_class_names
        )
        self.require_model_order(names)
        app_id_by_model_name = {
            model_name: app_id
            for app_id, group in enumerate(self.groups)
            for model_name in group.model_names
        }
        return tuple(app_id_by_model_name[name] for name in names)

    def app_group_indices(
        self,
        model_class_names: Sequence[str] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        names = (
            self.model_class_names
            if model_class_names is None
            else model_class_names
        )
        self.require_model_order(names)
        index = {name: position for position, name in enumerate(names)}
        return tuple(
            tuple(index[model_name] for model_name in group.model_names)
            for group in self.groups
        )

    def aggregate_probability_columns(self, probabilities: Any) -> tuple[Any, ...]:
        shape = getattr(probabilities, "shape", ())
        if len(shape) != 2 or shape[1] != len(self.model_class_names):
            raise ValueError(
                f"Expected a [batch, {len(self.model_class_names)}] probability "
                f"matrix; received shape={shape}"
            )
        return tuple(
            probabilities[:, indices].sum(axis=1)
            for indices in self.app_group_indices()
        )


CHEMSORT_CONTRACT = ModelContract(
    model_class_names=(
        "Cardboard",
        "Food Organics",
        "Glass",
        "Metal",
        "Miscellaneous Trash",
        "Paper",
        "Plastic",
        "Textile Trash",
        "Vegetation",
    ),
    groups=(
        ClassGroup("Paper/Cardboard", ("Cardboard", "Paper")),
        ClassGroup("Organic", ("Food Organics",)),
        ClassGroup("Glass", ("Glass",)),
        ClassGroup("Metal", ("Metal",)),
        ClassGroup(
            "Other",
            ("Miscellaneous Trash", "Textile Trash", "Vegetation"),
        ),
        ClassGroup("Plastic", ("Plastic",)),
    ),
)

# Compatibility aliases keep existing CLI entry points stable.
MODEL_CLASS_NAMES = CHEMSORT_CONTRACT.model_class_names
APP_CLASS_NAMES = CHEMSORT_CONTRACT.app_class_names
APP_GROUPS = CHEMSORT_CONTRACT.app_groups


def validate_contract() -> None:
    CHEMSORT_CONTRACT.validate()


def model_to_app_ids(
    model_class_names: Sequence[str] = MODEL_CLASS_NAMES,
) -> tuple[int, ...]:
    return CHEMSORT_CONTRACT.model_to_app_ids(model_class_names)


def app_group_indices(
    model_class_names: Sequence[str] = MODEL_CLASS_NAMES,
) -> tuple[tuple[int, ...], ...]:
    return CHEMSORT_CONTRACT.app_group_indices(model_class_names)


def aggregate_probability_columns(probabilities: Any) -> tuple[Any, ...]:
    return CHEMSORT_CONTRACT.aggregate_probability_columns(probabilities)
