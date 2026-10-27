"""Build dataset v2: RealWaste v1 split plus three single-object sources.

Steps (README section 2.2):

1. keep the v1 RealWaste split unchanged (val/test stay RealWaste-only);
2. map each source's labels to the 9 RealWaste classes;
3. drop exact duplicates (SHA-256) against RealWaste and across sources;
4. drop images whose pHash is within ``--near-distance`` of any RealWaste
   image, so nothing added can "peek" at val/test;
5. cap each (source, class) at ``--cap`` images;
6. hold out ``--holdout`` of each (source, class) as ``test_ext``;
7. drop test_ext images that are near duplicates of any training image.

The committed ``data/manifest_v2.csv`` is the locked version used for every
reported number. This tool writes ``data/manifest_v2.rebuilt.csv`` by default
and compares it with the locked file; pass ``--output data/manifest_v2.csv
--force`` only when you intend to create a new dataset version (then retrain
and re-evaluate everything).

    python -m tools.build_dataset_v2
    python -m tools.build_dataset_v2 --cap 600 --output data/manifest_v3.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemsort.contracts import CHEMSORT_CONTRACT  # noqa: E402
from chemsort.paths import DATA_DIR, MANIFESTS, RAW_DIR  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SOURCES: dict[str, dict] = {
    "drinking_waste": {
        "root": "DrinkingWasteClassification/rawimgs",
        "labels": {"AluCan": "Metal", "Glass": "Glass", "HDPEM": "Plastic", "PET": "Plastic"},
    },
    "trashnet": {
        "root": "TrashNet",
        "labels": {
            "cardboard": "Cardboard", "glass": "Glass", "metal": "Metal",
            "paper": "Paper", "plastic": "Plastic", "trash": "Miscellaneous Trash",
        },
    },
    "garbage_classification": {
        "root": "Garbage_Classification",
        "labels": {
            "battery": "Miscellaneous Trash", "biological": "Food Organics",
            "brown-glass": "Glass", "cardboard": "Cardboard", "clothes": "Textile Trash",
            "green-glass": "Glass", "metal": "Metal", "paper": "Paper", "plastic": "Plastic",
            "shoes": "Textile Trash", "trash": "Miscellaneous Trash", "white-glass": "Glass",
        },
    },
}
COLUMNS = ["path", "original_label", "split", "source", "source_class", "width", "height", "sha256", "phash"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_source(raw_dir: Path, source: str, spec: dict) -> tuple[list[dict], list[dict]]:
    """Fingerprint every readable image of one source."""
    import imagehash
    from PIL import Image

    records, invalid = [], []
    root = raw_dir / spec["root"]
    if not root.is_dir():
        raise SystemExit(f"Source folder not found: {root}")
    for source_class, label in spec["labels"].items():
        class_dir = root / source_class
        if not class_dir.is_dir():
            raise SystemExit(f"Missing class folder: {class_dir}")
        for path in sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
            relative = path.relative_to(raw_dir).as_posix()
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    phash = str(imagehash.phash(image.convert("RGB")))
            except Exception as error:  # Corrupt or unsupported files are reported, not fatal.
                invalid.append({"path": relative, "error": str(error)})
                continue
            records.append({
                "path": relative, "original_label": label, "split": "", "source": source,
                "source_class": source_class, "width": width, "height": height,
                "sha256": sha256_file(path), "phash": phash,
            })
    return records, invalid


def phash_array(values: pd.Series) -> np.ndarray:
    return np.asarray([int(value, 16) for value in values], dtype=np.uint64)


POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def min_hamming_distance(queries: np.ndarray, references: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Smallest Hamming distance from each 64-bit query hash to any reference."""
    result = np.full(len(queries), 65, dtype=np.int64)
    if len(references) == 0:
        return result
    for start in range(0, len(queries), chunk):
        block = queries[start : start + chunk]
        xor = block[:, np.newaxis] ^ references[np.newaxis, :]
        bits = POPCOUNT[xor.view(np.uint8)].reshape(len(block), len(references), 8)
        result[start : start + chunk] = bits.sum(axis=2, dtype=np.int64).min(axis=1)
    return result


def exact_dedup(external: pd.DataFrame, base_sha: set[str]) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(SOURCES)}
    frame = external.assign(_order=external["source"].map(order)).sort_values(["_order", "path"])
    frame = frame[~frame["sha256"].isin(base_sha)]
    return frame.drop_duplicates("sha256", keep="first").drop(columns="_order")


def cap_per_group(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby(["source", "original_label"], sort=True):
        parts.append(group.sample(n=min(cap, len(group)), random_state=seed) if len(group) > cap else group)
    return pd.concat(parts).sort_values("path")


def holdout_split(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Stratified by (source, class): round(n * fraction) rows go to test_ext."""
    rng = np.random.default_rng(seed)
    frame = frame.sort_values("path").copy()
    frame["split"] = "train"
    for _, group in frame.groupby(["source", "original_label"], sort=True):
        count = int(round(len(group) * fraction))
        if count:
            chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
            frame.loc[chosen, "split"] = "test_ext"
    return frame


def load_base(manifest_v1: Path) -> pd.DataFrame:
    base = pd.read_csv(manifest_v1)
    base["path"] = "RealWaste/" + base["path"].astype(str)
    base["source"] = "realwaste"
    base["source_class"] = base["original_label"]
    return base


def build(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    base = load_base(args.base_manifest)
    external_records, invalid = [], []
    for source, spec in SOURCES.items():
        if source in args.exclude_sources:
            continue
        print(f"Scanning {source} ...")
        records, bad = scan_source(args.raw_dir, source, spec)
        external_records += records
        invalid += [{"source": source, **item} for item in bad]
    external = pd.DataFrame(external_records, columns=COLUMNS)
    stats = {"scanned": len(external), "invalid": len(invalid)}

    external = exact_dedup(external, set(base["sha256"]))
    stats["after_exact_dedup"] = len(external)

    distance = min_hamming_distance(phash_array(external["phash"]), phash_array(base["phash"]))
    external = external[distance > args.near_distance]
    stats["after_realwaste_near_dedup"] = len(external)

    external = cap_per_group(external, args.cap, args.seed)
    stats["after_cap"] = len(external)

    external = holdout_split(external, args.holdout, args.seed)
    train_hashes = phash_array(pd.concat([
        base.loc[base["split"] == "train", "phash"], external.loc[external["split"] == "train", "phash"]
    ]))
    holdout = external["split"] == "test_ext"
    leak = min_hamming_distance(phash_array(external.loc[holdout, "phash"]), train_hashes) <= args.near_distance
    drop_index = external.loc[holdout].index[leak]
    external = external.drop(index=drop_index)
    stats["test_ext_dropped_as_near_dup_of_train"] = int(len(drop_index))

    manifest = pd.concat([base[COLUMNS], external[COLUMNS]], ignore_index=True)
    split_order = {"train": 0, "val": 1, "test": 2, "test_ext": 3}
    manifest = manifest.assign(_s=manifest["split"].map(split_order)).sort_values(
        ["_s", "source", "original_label", "path"]
    ).drop(columns="_s")
    if tuple(sorted(manifest["original_label"].unique())) != CHEMSORT_CONTRACT.model_class_names:
        raise SystemExit("Built manifest does not cover the 9-class contract")
    return manifest, {"external_filtering": stats, "invalid_images": invalid}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--base-manifest", type=Path, default=MANIFESTS["v1"])
    parser.add_argument("--output", type=Path, default=DATA_DIR / "manifest_v2.rebuilt.csv")
    parser.add_argument("--cap", type=int, default=400)
    parser.add_argument("--holdout", type=float, default=0.10)
    parser.add_argument("--near-distance", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-sources", nargs="*", default=[], choices=sorted(SOURCES))
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing output")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists. Pass --force to overwrite it.")

    manifest, details = build(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False, lineterminator="\n")
    stats = pd.crosstab([manifest["original_label"]], [manifest["split"], manifest["source"]])
    stats.to_csv(output.with_name(output.stem + "_split_stats.csv"))
    version = {
        "dataset_name": "RealWaste-ChemSort",
        "base_manifest": str(args.base_manifest),
        "sources": {name: spec["root"] for name, spec in SOURCES.items() if name not in args.exclude_sources},
        "excluded_sources": args.exclude_sources,
        "cap_per_source_label": args.cap,
        "holdout": args.holdout,
        "near_duplicate_distance": args.near_distance,
        "seed": args.seed,
        **details,
        "split_counts": manifest["split"].value_counts().to_dict(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(output),
    }
    output.with_name(output.stem + "_version.json").write_text(
        json.dumps(version, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(details["external_filtering"], indent=2))
    print(manifest.groupby(["split", "source"]).size().to_string())
    locked = MANIFESTS["v2"]
    if locked.is_file() and output != locked.resolve():
        reference = pd.read_csv(locked)
        same = set(zip(reference["path"], reference["split"])) == set(zip(manifest["path"], manifest["split"]))
        print(f"\nIdentical to locked {locked.name}: {same}")
        if not same:
            print("Differences are expected if sources or parameters changed; the locked manifest stays authoritative.")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
