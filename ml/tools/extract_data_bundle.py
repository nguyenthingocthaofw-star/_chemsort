"""Extract the team data bundle ``ChemSort_Data.zip`` into ``ml/data``.

The bundle holds three nested archives:

* ``ChemSort_data_raw.zip``       -> ``data/raw/`` (RealWaste, TrashNet,
  DrinkingWasteClassification, Garbage_Classification; needed by every
  pipeline because manifests point here)
* ``ChemSort_data_processed.zip`` -> ``data/processed/`` (ready-made
  Ultralytics ``dataset/<split>/<class>/`` tree plus metadata)
* ``ChemSort_flashcards.zip``     -> ``data/content/``

Each inner archive is first copied to a temporary file (random access into a
compressed member is very slow), then extracted. Existing files with the
same size are skipped, so an interrupted run can be resumed.

    python -m tools.extract_data_bundle                 # raw only
    python -m tools.extract_data_bundle --parts raw processed flashcards
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemsort.paths import DATA_BUNDLE_ZIP, DATA_DIR, PROCESSED_DIR  # noqa: E402

PARTS = {
    "raw": ("ChemSort_Data/ChemSort_data_raw.zip", DATA_DIR),
    "processed": ("ChemSort_Data/ChemSort_data_processed.zip", PROCESSED_DIR),
    "flashcards": ("ChemSort_Data/ChemSort_flashcards.zip", DATA_DIR / "content"),
}
SKIPPED_SUFFIXES = {".cache"}


def safe_target(root: Path, member: str) -> Path | None:
    """Reject absolute paths and '..' components (zip-slip)."""
    pure = PurePosixPath(member.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        return None
    target = (root / Path(*pure.parts)).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        return None
    return target


def extract_archive(archive: zipfile.ZipFile, destination: Path) -> tuple[int, int, int]:
    written = skipped = rejected = 0
    members = [info for info in archive.infolist() if not info.is_dir()]
    for index, info in enumerate(members, start=1):
        if Path(info.filename).suffix in SKIPPED_SUFFIXES:
            continue
        target = safe_target(destination, info.filename)
        if target is None:
            rejected += 1
            continue
        if target.is_file() and target.stat().st_size == info.file_size:
            skipped += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)
            written += 1
        if index % 2000 == 0 or index == len(members):
            print(f"  {index}/{len(members)} files")
    return written, skipped, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=DATA_BUNDLE_ZIP)
    parser.add_argument("--parts", nargs="+", choices=sorted(PARTS), default=["raw"])
    args = parser.parse_args()

    if not args.bundle.is_file():
        raise SystemExit(f"Data bundle not found: {args.bundle}")
    with zipfile.ZipFile(args.bundle) as outer:
        names = set(outer.namelist())
        for part in args.parts:
            inner_name, destination = PARTS[part]
            if inner_name not in names:
                raise SystemExit(f"{inner_name} is not in {args.bundle.name}")
            destination.mkdir(parents=True, exist_ok=True)
            print(f"[{part}] copying {inner_name} to a temporary file ...")
            with tempfile.TemporaryDirectory(dir=DATA_DIR, prefix=".bundle_") as temp:
                inner_path = Path(temp) / Path(inner_name).name
                with outer.open(inner_name) as source, inner_path.open("wb") as sink:
                    shutil.copyfileobj(source, sink, 4 * 1024 * 1024)
                print(f"[{part}] extracting into {destination} ...")
                with zipfile.ZipFile(inner_path) as inner:
                    written, skipped, rejected = extract_archive(inner, destination)
            print(f"[{part}] written={written} skipped_existing={skipped} rejected_paths={rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
