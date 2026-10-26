"""Describe a manifest: class balance, sources, image sizes and colour stats.

Writes to ``reports/data/<dataset>/``::

    summary.json                 totals, imbalance ratio, duplicates
    class_split_counts.csv       9 classes x splits
    group_split_counts.csv       6 app groups x splits
    source_class_counts.csv      source x class (all splits)
    class_distribution.png       per-class counts per split
    source_distribution.png      train images per class, stacked by source
    image_sizes.png              width/height scatter and aspect ratios
    pixel_stats.csv/.png         mean RGB and brightness per class (--pixel-sample)

    python -m tools.dataset_stats
    python -m tools.dataset_stats --manifest v1
    python -m tools.dataset_stats --manifest data/augmented/v2_medium/manifest_v2_aug_medium.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemsort.contracts import APP_CLASS_NAMES, MODEL_CLASS_NAMES  # noqa: E402
from chemsort.paths import REPORTS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="v2")
    parser.add_argument("--pixel-sample", type=int, default=60, help="Images per class for colour stats (0 = skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def counts_table(frame: pd.DataFrame, row: str, order) -> pd.DataFrame:
    from chemsort.data import SPLITS

    table = pd.crosstab(frame[row], frame["split"]).reindex(index=list(order), fill_value=0)
    table = table[[split for split in SPLITS if split in table.columns]]
    table["total"] = table.sum(axis=1)
    table.loc["total"] = table.sum()
    return table


def main() -> int:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from chemsort.data import load_manifest, stratified_sample
    from chemsort.preprocess import app_preprocess

    frame = load_manifest(args.manifest)
    dataset = frame.attrs["dataset"]
    output_dir = args.output_dir or REPORTS_DIR / "data" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = counts_table(frame, "label", MODEL_CLASS_NAMES)
    groups = counts_table(frame, "label_6", APP_CLASS_NAMES)
    sources = pd.crosstab(frame["source"], frame["label"]).reindex(columns=list(MODEL_CLASS_NAMES), fill_value=0)
    classes.to_csv(output_dir / "class_split_counts.csv")
    groups.to_csv(output_dir / "group_split_counts.csv")
    sources.to_csv(output_dir / "source_class_counts.csv")

    train_counts = classes.drop(index="total")["train"] if "train" in classes else pd.Series(dtype=int)
    summary = {
        "manifest": frame.attrs["manifest_path"],
        "dataset": dataset,
        "images": int(len(frame)),
        "splits": frame["split"].value_counts().to_dict(),
        "sources": frame["source"].value_counts().to_dict(),
        "train_imbalance_ratio_max_over_min": float(train_counts.max() / max(train_counts.min(), 1)) if len(train_counts) else None,
        "duplicate_sha256": int(frame["sha256"].duplicated().sum()) if "sha256" in frame else None,
        "classes_missing_per_split": {
            split: [name for name in MODEL_CLASS_NAMES if classes.loc[name, split] == 0]
            for split in classes.columns if split != "total"
        },
    }
    if {"width", "height"} <= set(frame.columns):
        summary["image_size"] = {
            "width": frame["width"].describe().round(1).to_dict(),
            "height": frame["height"].describe().round(1).to_dict(),
            "smaller_than_224": int(((frame["width"] < 224) | (frame["height"] < 224)).sum()),
        }

    # Class distribution per split.
    table = classes.drop(index="total").drop(columns="total")
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / len(table.columns)
    positions = np.arange(len(table))
    for index, split in enumerate(table.columns):
        ax.bar(positions + (index - (len(table.columns) - 1) / 2) * width, table[split], width, label=split)
    ax.set_xticks(positions, table.index, rotation=30, ha="right")
    ax.set_ylabel("images")
    ax.set_title(f"Images per class and split - {dataset}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=130)
    plt.close(fig)

    # Train images by source.
    train = frame.loc[frame["split"] == "train"]
    by_source = pd.crosstab(train["label"], train["source"]).reindex(list(MODEL_CLASS_NAMES), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    by_source.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_ylabel("train images")
    ax.set_xlabel("")
    ax.set_title(f"Train images per class by source - {dataset}")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(output_dir / "source_distribution.png", dpi=130)
    plt.close(fig)

    if {"width", "height"} <= set(frame.columns):
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
        for source, group in frame.groupby("source"):
            left.scatter(group["width"], group["height"], s=6, alpha=0.4, label=source)
        left.axvline(224, color="grey", lw=0.8, ls="--")
        left.axhline(224, color="grey", lw=0.8, ls="--")
        left.set_xscale("log")
        left.set_yscale("log")
        left.set_xlabel("width (px)")
        left.set_ylabel("height (px)")
        left.set_title("Image sizes (dashed: model input 224)")
        left.legend(markerscale=3, fontsize=8)
        right.hist(frame["width"] / frame["height"], bins=40, color="#4C72B0")
        right.set_xlabel("aspect ratio (width / height)")
        right.set_ylabel("images")
        right.set_title("Aspect ratios (centre crop keeps the middle square)")
        fig.tight_layout()
        fig.savefig(output_dir / "image_sizes.png", dpi=130)
        plt.close(fig)

    if args.pixel_sample > 0:
        sample = stratified_sample(train, args.pixel_sample, args.seed)
        rows = []
        for label, path in zip(sample["label"], sample["abs_path"]):
            image = app_preprocess(path).astype(np.float32)
            rows.append({"class": label, "r": image[..., 0].mean(), "g": image[..., 1].mean(), "b": image[..., 2].mean(),
                         "brightness": image.mean(), "contrast": image.std()})
        pixels = pd.DataFrame(rows).groupby("class").mean().reindex(list(MODEL_CLASS_NAMES)).round(2)
        pixels.to_csv(output_dir / "pixel_stats.csv")
        fig, ax = plt.subplots(figsize=(12, 4.5))
        positions = np.arange(len(pixels))
        for offset, (channel, color) in enumerate((("r", "#C44E52"), ("g", "#55A868"), ("b", "#4C72B0"))):
            ax.bar(positions + (offset - 1) * 0.27, pixels[channel], 0.27, color=color, label=channel.upper())
        ax.set_xticks(positions, pixels.index, rotation=30, ha="right")
        ax.set_ylabel("mean pixel (0-255)")
        ax.set_title(f"Mean colour per class on the model input ({args.pixel_sample} train images/class)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "pixel_stats.png", dpi=130)
        plt.close(fig)
        summary["pixel_stats_images_per_class"] = args.pixel_sample

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(classes.to_string())
    print(f"\nReports: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
