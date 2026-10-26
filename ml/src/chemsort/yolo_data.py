"""Build the ``split/class/image`` folder tree that Ultralytics YOLO-cls reads.

Files are hard links into ``data/raw`` (no extra disk space); when hard links
are impossible (different drive, FAT/exFAT) they are copied. Each image is
named ``<source>_<sha16>.<ext>``, the naming of the published processed
dataset, so a tree built here and the extracted ``data/processed/dataset``
are interchangeable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from .contracts import CHEMSORT_CONTRACT
from .data import SPLITS

STAMP_FILE = "chemsort_tree.json"


def _manifest_digest(frame: pd.DataFrame) -> str:
    rows = frame[["rel_path", "label", "split"]].sort_values("rel_path")
    payload = rows.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _place(source: Path, destination: Path, mode: str) -> str:
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    if mode == "symlink":
        destination.symlink_to(source)
        return "symlink"
    shutil.copy2(source, destination)
    return "copy"


def build_classification_tree(
    frame: pd.DataFrame,
    output_dir: Path,
    splits: tuple[str, ...] = SPLITS,
    mode: str = "hardlink",
    rebuild: bool = False,
) -> dict[str, object]:
    """Create ``output_dir/<split>/<class>/<image_id>`` for the given splits.

    Every split gets a folder for all 9 classes, even when empty, so
    Ultralytics derives the same class indices for train and val.
    """
    output_dir = Path(output_dir)
    digest = _manifest_digest(frame[frame["split"].isin(splits)])
    stamp_path = output_dir / STAMP_FILE
    if stamp_path.is_file() and not rebuild:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        if stamp.get("manifest_digest") == digest:
            return stamp
        raise SystemExit(
            f"{output_dir} was built from a different manifest. "
            "Pass --rebuild-dataset to recreate it."
        )
    if output_dir.exists() and rebuild:
        shutil.rmtree(output_dir)

    counts: dict[str, dict[str, int]] = {}
    methods: dict[str, int] = {}
    for split in splits:
        subset = frame[frame["split"] == split]
        counts[split] = {}
        for name in CHEMSORT_CONTRACT.model_class_names:
            (output_dir / split / name).mkdir(parents=True, exist_ok=True)
            counts[split][name] = 0
        for row in subset.itertuples(index=False):
            source = Path(row.abs_path)
            if not source.is_file():
                raise SystemExit(f"Missing image: {source}")
            destination = output_dir / split / row.label / row.image_id
            if not destination.exists():
                method = _place(source, destination, mode)
                methods[method] = methods.get(method, 0) + 1
            counts[split][row.label] += 1

    stamp = {
        "manifest": frame.attrs.get("manifest_path"),
        "manifest_digest": digest,
        "splits": list(splits),
        "counts": counts,
        "placement": methods,
        "class_order": list(CHEMSORT_CONTRACT.model_class_names),
    }
    stamp_path.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    return stamp


def tree_class_order(dataset_dir: Path, split: str = "train") -> tuple[str, ...]:
    """Class order Ultralytics will use: sorted sub-folder names."""
    return tuple(sorted(path.name for path in (dataset_dir / split).iterdir() if path.is_dir()))
