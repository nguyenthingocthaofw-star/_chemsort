"""Manifest loading shared by every training and evaluation pipeline.

Two manifest formats are supported:

* v1 (``data/manifest.csv``): RealWaste only, ``path`` relative to
  ``data/raw/RealWaste``.
* v2 (``data/manifest_v2.csv``): RealWaste plus three single-object sources,
  ``path`` relative to ``data/raw`` and extra ``source``/``source_class``
  columns. The ``test_ext`` split holds single-object images never used for
  training or model selection.

Augmented manifests written by ``tools.augment_dataset`` are v2 manifests
with an extra ``base`` column: ``raw`` rows resolve against ``data/raw``,
``data`` rows (the generated images, ``source == "augmented"``) against
``data/``. Augmented rows are only allowed in the train split.

Both are normalised to the same columns so downstream code never branches on
the version.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import CHEMSORT_CONTRACT
from .paths import MANIFESTS, RAW_DIR

SPLITS = ("train", "val", "test", "test_ext")
EVALUATION_SPLITS = ("val", "test", "test_ext")
REQUIRED_COLUMNS = {"path", "original_label", "split"}


def resolve_manifest(name_or_path: str | Path) -> Path:
    """Accept ``v1``/``v2`` shortcuts or an explicit CSV path."""
    if isinstance(name_or_path, str) and name_or_path in MANIFESTS:
        path = MANIFESTS[name_or_path]
    else:
        path = Path(name_or_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return path


def dataset_tag(manifest_path: Path) -> str:
    for tag, path in MANIFESTS.items():
        if path.resolve() == manifest_path.resolve():
            return tag
    stem = manifest_path.stem
    return stem[len("manifest_"):] if stem.startswith("manifest_") else stem


def load_manifest(
    name_or_path: str | Path = "v2",
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Return the manifest with normalised columns.

    Added columns: ``rel_path`` (relative to ``raw_dir``), ``abs_path``,
    ``source``, ``source_class``, ``label``, ``label_id``, ``label_6``,
    ``label_6_id`` and ``image_id`` (``<source>_<sha16>``, the file name used
    in the Ultralytics folder layout).
    """
    manifest_path = resolve_manifest(name_or_path)
    frame = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"{manifest_path.name} is missing columns: {sorted(missing)}"
        )

    frame = frame.copy()
    if "source" not in frame.columns:
        # v1 manifest: RealWaste only, paths relative to raw/RealWaste.
        frame["source"] = "realwaste"
        frame["source_class"] = frame["original_label"]
        frame["rel_path"] = "RealWaste/" + frame["path"].astype(str)
    else:
        frame["rel_path"] = frame["path"].astype(str)
    frame["rel_path"] = frame["rel_path"].str.replace("\\", "/", regex=False)

    model_names = CHEMSORT_CONTRACT.model_class_names
    found = tuple(sorted(frame["original_label"].dropna().unique()))
    if found != model_names:
        raise ValueError(
            "Manifest labels differ from the locked 9-class contract.\n"
            f"Expected: {list(model_names)}\nFound:    {list(found)}"
        )
    unknown_splits = set(frame["split"].unique()) - set(SPLITS)
    if unknown_splits:
        raise ValueError(f"Unknown split names: {sorted(unknown_splits)}")

    label_to_id = {name: index for index, name in enumerate(model_names)}
    to_app = CHEMSORT_CONTRACT.model_to_app_ids()
    frame["label"] = frame["original_label"]
    frame["label_id"] = frame["label"].map(label_to_id).astype("int32")
    frame["label_6_id"] = frame["label_id"].map(lambda i: to_app[i]).astype("int32")
    frame["label_6"] = frame["label_6_id"].map(
        lambda i: CHEMSORT_CONTRACT.app_class_names[i]
    )
    if "base" in frame.columns:
        leaked = frame.loc[(frame["source"] == "augmented") & (frame["split"] != "train")]
        if not leaked.empty:
            raise ValueError(f"{len(leaked)} augmented image(s) outside the train split")
        roots = {"raw": raw_dir, "data": raw_dir.parent}
        frame["base"] = frame["base"].fillna("raw")
        unknown = set(frame["base"]) - set(roots)
        if unknown:
            raise ValueError(f"Unknown base values: {sorted(unknown)}")
        frame["abs_path"] = [str(roots[base] / rel) for base, rel in zip(frame["base"], frame["rel_path"])]
    else:
        frame["abs_path"] = frame["rel_path"].map(lambda rel: str(raw_dir / rel))

    if "sha256" in frame.columns:
        suffixes = frame["rel_path"].map(lambda rel: Path(rel).suffix.lower())
        frame["image_id"] = (
            frame["source"].astype(str)
            + "_"
            + frame["sha256"].astype(str).str[:16]
            + suffixes
        )
    else:
        frame["image_id"] = frame["rel_path"].map(lambda rel: Path(rel).name)

    frame.attrs["manifest_path"] = str(manifest_path)
    frame.attrs["dataset"] = dataset_tag(manifest_path)
    return frame


def select_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    subset = frame.loc[frame["split"] == split].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError(
            f"Split {split!r} is empty in {frame.attrs.get('manifest_path')}"
        )
    subset.attrs.update(frame.attrs)
    return subset


def missing_files(frame: pd.DataFrame) -> list[str]:
    return [path for path in frame["abs_path"] if not Path(path).is_file()]


def require_files(frame: pd.DataFrame, raw_dir: Path = RAW_DIR) -> None:
    missing = missing_files(frame)
    if missing:
        preview = "\n  ".join(missing[:5])
        raise SystemExit(
            f"{len(missing)} image(s) listed in the manifest are missing under "
            f"{raw_dir}.\n  {preview}\n"
            "Extract the dataset first: python -m tools.extract_data_bundle"
        )


def stratified_sample(
    frame: pd.DataFrame,
    per_class: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic subset with at most ``per_class`` rows per 9-class label."""
    parts = [
        group.sample(n=min(per_class, len(group)), random_state=seed)
        for _, group in frame.groupby("label", sort=True)
    ]
    sample = pd.concat(parts).sort_values("rel_path").reset_index(drop=True)
    sample.attrs.update(frame.attrs)
    return sample


def split_counts(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(frame["label"], frame["split"])
    columns = [split for split in SPLITS if split in table.columns]
    table = table[columns]
    table["total"] = table.sum(axis=1)
    return table
