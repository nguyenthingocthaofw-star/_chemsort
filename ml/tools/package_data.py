"""Package the raw dataset and an augmented dataset as zip files.

Archive paths are relative to ``ml/``, so extracting a zip inside ``ml/``
restores the exact layout the pipelines expect::

    ChemSort_Data_raw.zip                 data/raw/**, data/manifest*.csv, split stats
    ChemSort_Data_augmented_<tag>.zip     data/augmented/<tag>/** (images, manifest, report)

The augmented manifest references the original images under ``data/raw``,
so training on it needs both archives. JPEG/PNG files are stored without
recompression (they do not shrink); text files are deflated. Each archive
carries a ``README_<name>.txt`` and a SHA-256 list of its files.

    python -m tools.package_data
    python -m tools.package_data --parts augmented --augmented v2_strong
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemsort.paths import DATA_DIR, FLUTTER_ROOT, ML_ROOT, RAW_DIR  # noqa: E402

STORED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".zip"}
SKIP_NAMES = {"Thumbs.db", ".DS_Store", "desktop.ini"}
METADATA_FILES = [
    "manifest.csv", "manifest_v2.csv", "split_stats.csv", "split_stats_v2.csv",
    "dataset_version.json", "dataset_version_v2.json", "class_names.json",
]

RAW_README = """ChemSort - raw dataset
======================
Extract inside the repository's ml/ folder (creates ml/data/raw and the manifests).

data/raw/RealWaste                      RealWaste (9 classes) - train/val/test of v1 and v2
data/raw/TrashNet                       TrashNet
data/raw/Garbage_Classification         Garbage Classification (12 classes)
data/raw/DrinkingWasteClassification    Drinking Waste Classification
data/manifest.csv                       dataset v1 (RealWaste only)
data/manifest_v2.csv                    dataset v2 (4 sources, train/val/test/test_ext)

Not every file under data/raw is used: the manifests list the images kept
after label mapping, de-duplication and per-source caps (tools/build_dataset_v2.py).
SHA256SUMS_raw.txt lists every file in this archive.
"""

AUGMENTED_README = """ChemSort - augmented dataset ({tag})
====================================
Extract inside the repository's ml/ folder (creates ml/data/augmented/{tag}).
Requires ChemSort_Data_raw.zip as well: original images stay in data/raw.

images/<class>/*.jpg          generated training images (train split only)
{manifest}    original rows (base=raw) + generated rows (base=data)
augmentation_report.json      preset parameters, seed, counts per class
class_counts.csv/.png         train images per class before / after
examples.png                  originals next to augmented copies

Train:  python -m pipelines.train_keras --model efficientnetb0 --manifest data/augmented/{tag}/{manifest}
        python -m pipelines.train_yolo --manifest data/augmented/{tag}/{manifest}
val/test/test_ext are unchanged, so results stay comparable with the v2 runs.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", choices=("raw", "augmented"), default=["raw", "augmented"])
    parser.add_argument("--augmented", default="v2_medium", help="Folder name under data/augmented")
    parser.add_argument("--output-dir", type=Path, default=FLUTTER_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in SKIP_NAMES:
            yield path


def write_archive(target: Path, files: list[Path], readme_name: str, readme: str, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise SystemExit(f"{target} exists. Use --overwrite to replace it.")
    partial = target.with_suffix(".zip.partial")
    sums: list[str] = []
    total = 0
    with zipfile.ZipFile(partial, "w", allowZip64=True) as archive:
        for index, path in enumerate(files, 1):
            arcname = path.relative_to(ML_ROOT).as_posix()
            compression = zipfile.ZIP_STORED if path.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            data = path.read_bytes()
            archive.writestr(zipfile.ZipInfo.from_file(path, arcname), data, compress_type=compression)
            sums.append(f"{hashlib.sha256(data).hexdigest()}  {arcname}")
            total += len(data)
            if index % 5000 == 0:
                print(f"  {index}/{len(files)} files", flush=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        archive.writestr(readme_name, readme + f"\nCreated {stamp} - {len(files)} files, {total / 2**20:.1f} MB\n",
                         compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr(f"SHA256SUMS_{readme_name.removeprefix('README_')}", "\n".join(sums) + "\n",
                         compress_type=zipfile.ZIP_DEFLATED)
    partial.replace(target)
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
    if bad:
        raise SystemExit(f"CRC check failed for {bad} in {target}")
    print(f"Wrote {target} ({target.stat().st_size / 2**30:.2f} GB, {len(files)} files, CRC OK)")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "raw" in args.parts:
        if not RAW_DIR.is_dir():
            raise SystemExit(f"{RAW_DIR} not found. Extract it first: python -m tools.extract_data_bundle")
        files = list(iter_files(RAW_DIR)) + [DATA_DIR / name for name in METADATA_FILES if (DATA_DIR / name).is_file()]
        print(f"Packaging raw data: {len(files)} files")
        write_archive(args.output_dir / "ChemSort_Data_raw.zip", files, "README_raw.txt", RAW_README, args.overwrite)

    if "augmented" in args.parts:
        folder = DATA_DIR / "augmented" / args.augmented
        manifests = sorted(folder.glob("manifest_*.csv"))
        if not manifests:
            raise SystemExit(f"No augmented dataset in {folder}. Run: python -m tools.augment_dataset")
        files = list(iter_files(folder))
        print(f"Packaging augmented data {args.augmented}: {len(files)} files")
        readme = AUGMENTED_README.format(tag=args.augmented, manifest=manifests[0].name)
        write_archive(args.output_dir / f"ChemSort_Data_augmented_{args.augmented}.zip", files,
                      "README_augmented.txt", readme, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
