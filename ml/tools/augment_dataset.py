"""Offline augmentation: write augmented training images and a manifest.

Only the **train** split is augmented; val/test/test_ext rows are copied to
the new manifest untouched, so every reported metric stays comparable.

Modes:

* ``balance`` (default): every class is topped up to ``--target-per-class``
  (default: the largest train class) with augmented copies of its own
  images. Parents are used round-robin, so each original is reused about the
  same number of times.
* ``multiply``: ``--copies`` augmented versions of every train image.

Output (``data/augmented/<dataset>_<preset>/``)::

    images/<class>/<source>_<sha16>_a<k>.jpg   generated images
    manifest_<dataset>_aug_<preset>.csv        originals + generated rows
    augmentation_report.json                   policy, seed, counts, timing
    class_counts.csv, class_counts.png         train counts before / after
    examples.png                               parents next to augmented copies

Train on it with ``--manifest data/augmented/<...>/manifest_<...>.csv``. Each
image is generated from a seed derived from ``--seed`` and the parent hash,
so reruns are reproducible regardless of worker count.

    python -m tools.augment_dataset
    python -m tools.augment_dataset --preset strong --target-per-class 2000
    python -m tools.augment_dataset --mode multiply --copies 1 --preset light
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemsort.augment import PRESETS, augment_image, get_preset, limit_size  # noqa: E402
from chemsort.contracts import MODEL_CLASS_NAMES  # noqa: E402
from chemsort.paths import DATA_DIR  # noqa: E402

AUGMENTED_DIR = DATA_DIR / "augmented"
MANIFEST_COLUMNS = [
    "path", "original_label", "split", "source", "source_class", "width", "height",
    "sha256", "phash", "base", "parent_path", "preset", "aug_seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="v2", help="v1, v2 or a CSV path")
    parser.add_argument("--preset", choices=sorted(set(PRESETS) - {"none"}), default="medium")
    parser.add_argument("--mode", choices=("balance", "multiply"), default="balance")
    parser.add_argument("--target-per-class", type=int, default=None, help="balance mode (default: largest class)")
    parser.add_argument("--copies", type=int, default=1, help="multiply mode: copies per train image")
    parser.add_argument("--max-side", type=int, default=512, help="Downscale parents first (0 = keep size)")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-class", type=int, default=None, help="Only generate this many per class (smoke)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def plan_jobs(train: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    """Return one job per image to generate."""
    rng = np.random.default_rng(args.seed)
    counts = train["label"].value_counts()
    target = args.target_per_class or int(counts.max())
    jobs: list[dict] = []
    for label in MODEL_CLASS_NAMES:
        group = train.loc[train["label"] == label].sort_values("rel_path").reset_index(drop=True)
        needed = max(0, target - len(group)) if args.mode == "balance" else len(group) * args.copies
        if args.limit_per_class is not None:
            needed = min(needed, args.limit_per_class)
        if needed == 0 or group.empty:
            continue
        rounds = -(-needed // len(group))
        order = np.concatenate([rng.permutation(len(group)) for _ in range(rounds)])[:needed]
        uses: dict[int, int] = {}
        for index in order:
            row = group.iloc[int(index)]
            copy = uses.get(int(index), 0)
            uses[int(index)] = copy + 1
            jobs.append({
                "label": label,
                "abs_path": row["abs_path"],
                "rel_path": row["rel_path"],
                "source": row["source"],
                "sha256": str(row.get("sha256", "")) or hashlib.sha256(row["rel_path"].encode()).hexdigest(),
                "copy": copy,
            })
    return jobs


def job_seed(base_seed: int, sha256: str, copy: int) -> int:
    return int(np.random.SeedSequence([base_seed, zlib.crc32(sha256.encode()), copy]).generate_state(1)[0])


def run_job(job: dict, preset: str, output_root: str, max_side: int, quality: int, base_seed: int) -> dict:
    import imagehash
    from PIL import Image

    from chemsort.preprocess import load_rgb

    seed = job_seed(base_seed, job["sha256"], job["copy"])
    image = limit_size(Image.fromarray(load_rgb(job["abs_path"])), max_side or None)
    augmented = augment_image(image, preset, np.random.default_rng(seed))
    name = f"{job['source']}_{job['sha256'][:16]}_a{job['copy']}.jpg"
    relative = Path("images") / job["label"] / name
    target = Path(output_root) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    augmented.save(target, format="JPEG", quality=quality, optimize=True)
    return {
        "relative": relative.as_posix(),
        "original_label": job["label"],
        "source_class": job["source"],
        "width": augmented.width,
        "height": augmented.height,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "phash": str(imagehash.phash(augmented)),
        "parent_path": job["rel_path"],
        "aug_seed": seed,
    }


def plot_counts(table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(table))
    ax.bar(positions, table["original"], label="original", color="#4C72B0")
    ax.bar(positions, table["augmented"], bottom=table["original"], label="augmented", color="#DD8452")
    ax.set_xticks(positions, table.index, rotation=30, ha="right")
    ax.set_ylabel("train images")
    ax.set_title("Train split per class: original vs augmented")
    for x, total in zip(positions, table["total"]):
        ax.text(x, total, str(int(total)), ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def plot_examples(rows: pd.DataFrame, output_root: Path, raw_lookup: dict[str, str], output_path: Path, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from chemsort.preprocess import app_preprocess

    picks = pd.concat([
        group.sample(n=min(2, len(group)), random_state=seed)
        for _, group in rows.groupby("original_label", sort=True)
    ]).reset_index(drop=True)
    if picks.empty:
        return
    labels = list(dict.fromkeys(picks["original_label"]))
    fig, axes = plt.subplots(len(labels), 4, figsize=(8.4, 2.2 * len(labels)), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for row_index, label in enumerate(labels):
        group = picks.loc[picks["original_label"] == label].reset_index(drop=True)
        for pair, row in group.iterrows():
            parent = app_preprocess(raw_lookup[row["parent_path"]])
            child = app_preprocess(output_root / row["relative"])
            for offset, (image, title) in enumerate(((parent, "original"), (child, "augmented"))):
                axis = axes[row_index, pair * 2 + offset]
                axis.imshow(image)
                axis.set_title(f"{label}\n{title}" if pair == 0 and offset == 0 else title, fontsize=8)
    fig.suptitle("Original vs augmented (model input, 224x224 centre crop)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    from chemsort.data import load_manifest, require_files, select_split

    manifest = load_manifest(args.manifest)
    if "base" in manifest.columns:
        raise SystemExit("This manifest is already augmented; start from the original manifest.")
    dataset = manifest.attrs["dataset"]
    preset = get_preset(args.preset)
    output_root = (args.output_dir or AUGMENTED_DIR / f"{dataset}_{args.preset}").resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} exists. Use --overwrite to regenerate it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    train = select_split(manifest, "train")
    require_files(train)
    jobs = plan_jobs(train, args)
    print(f"Dataset {dataset}: {len(train)} train images -> generating {len(jobs)} ({args.mode}, preset {args.preset})")

    started = time.perf_counter()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        worker = partial(run_job, preset=args.preset, output_root=str(output_root),
                         max_side=args.max_side, quality=args.quality, base_seed=args.seed)
        for index, result in enumerate(pool.map(worker, jobs, chunksize=32), 1):
            results.append(result)
            if index % 500 == 0 or index == len(jobs):
                print(f"  {index}/{len(jobs)}", flush=True)
    elapsed = time.perf_counter() - started

    generated = pd.DataFrame(results)
    try:
        data_relative = output_root.relative_to(DATA_DIR).as_posix()
    except ValueError as error:
        raise SystemExit(f"--output-dir must be inside {DATA_DIR}") from error

    original_rows = pd.read_csv(manifest.attrs["manifest_path"])
    original_rows["base"] = "raw"
    original_rows["parent_path"] = ""
    original_rows["preset"] = ""
    original_rows["aug_seed"] = ""
    if not generated.empty:
        generated_rows = pd.DataFrame({
            "path": data_relative + "/" + generated["relative"],
            "original_label": generated["original_label"],
            "split": "train",
            "source": "augmented",
            "source_class": generated["source_class"],
            "width": generated["width"],
            "height": generated["height"],
            "sha256": generated["sha256"],
            "phash": generated["phash"],
            "base": "data",
            "parent_path": generated["parent_path"],
            "preset": args.preset,
            "aug_seed": generated["aug_seed"],
        })
    else:
        generated_rows = pd.DataFrame(columns=MANIFEST_COLUMNS)
    combined = pd.concat([original_rows.reindex(columns=MANIFEST_COLUMNS), generated_rows], ignore_index=True)
    manifest_path = output_root / f"manifest_{dataset}_aug_{args.preset}.csv"
    combined.to_csv(manifest_path, index=False)

    # Re-load through the shared loader: checks labels, splits and file paths.
    check = load_manifest(manifest_path)
    require_files(check)

    table = pd.DataFrame({
        "original": train["label"].value_counts(),
        "augmented": generated["original_label"].value_counts() if not generated.empty else 0,
    }).reindex(list(MODEL_CLASS_NAMES)).fillna(0).astype(int)
    table["total"] = table["original"] + table["augmented"]
    table.index.name = "class"
    table.to_csv(output_root / "class_counts.csv")
    plot_counts(table, output_root / "class_counts.png")
    if not generated.empty:
        raw_lookup = dict(zip(train["rel_path"], train["abs_path"]))
        plot_examples(generated, output_root, raw_lookup, output_root / "examples.png", args.seed)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": manifest.attrs["manifest_path"],
        "output_manifest": str(manifest_path),
        "dataset": dataset,
        "mode": args.mode,
        "target_per_class": args.target_per_class or (int(train["label"].value_counts().max()) if args.mode == "balance" else None),
        "copies": args.copies if args.mode == "multiply" else None,
        "preset": preset.to_dict(),
        "seed": args.seed,
        "max_side": args.max_side,
        "jpeg_quality": args.quality,
        "train_original": int(len(train)),
        "train_generated": int(len(generated)),
        "train_total": int(len(train) + len(generated)),
        "splits": check["split"].value_counts().to_dict(),
        "class_counts": table.reset_index().to_dict(orient="records"),
        "seconds": round(elapsed, 1),
        "command": " ".join(sys.argv),
        "note": "Only the train split is augmented; val/test/test_ext are unchanged.",
    }
    (output_root / "augmentation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table.to_string())
    print(f"\nGenerated {len(generated)} images in {elapsed:.0f}s")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
