from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import imagehash
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


DATA_ROOT = Path("data/raw/RealWaste")
OUTPUT_DIR = Path("data")
EXCLUDED_FILE = OUTPUT_DIR / "excluded.csv"

SEED = 42
NEAR_DUPLICATE_DISTANCE = 4

EXPECTED_COUNTS = {
    "Cardboard": 461,
    "Food Organics": 411,
    "Glass": 420,
    "Metal": 790,
    "Miscellaneous Trash": 495,
    "Paper": 500,
    "Plastic": 921,
    "Textile Trash": 318,
    "Vegetation": 436,
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


if not DATA_ROOT.exists():
    raise SystemExit(f"Không tìm thấy thư mục: {DATA_ROOT.resolve()}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []
invalid_records = []

# 1. Kiểm tra từng lớp và từng ảnh
for label in EXPECTED_COUNTS:
    class_dir = DATA_ROOT / label

    if not class_dir.exists():
        raise SystemExit(f"Thiếu thư mục nhãn: {class_dir}")

    image_paths = sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )

    for image_path in image_paths:
        relative_path = image_path.relative_to(DATA_ROOT).as_posix()

        try:
            with Image.open(image_path) as image:
                image.load()

                width, height = image.size
                mode = image.mode
                perceptual_hash = str(
                    imagehash.phash(image.convert("RGB"))
                )

            records.append({
                "path": relative_path,
                "original_label": label,
                "width": width,
                "height": height,
                "mode": mode,
                "file_size": image_path.stat().st_size,
                "sha256": sha256_file(image_path),
                "phash": perceptual_hash,
            })

        except Exception as error:
            invalid_records.append({
                "path": relative_path,
                "original_label": label,
                "error": str(error),
            })

if not records:
    raise SystemExit("Không tìm thấy ảnh hợp lệ.")

inventory = pd.DataFrame(records).sort_values("path")
inventory.to_csv(OUTPUT_DIR / "inventory.csv", index=False)

invalid = pd.DataFrame(
    invalid_records,
    columns=["path", "original_label", "error"],
)
invalid.to_csv(OUTPUT_DIR / "invalid_images.csv", index=False)

# 2. Kiểm tra số lượng ảnh
counts = (
    inventory["original_label"]
    .value_counts()
    .reindex(EXPECTED_COUNTS.keys(), fill_value=0)
)

print("\nSố ảnh đọc được:")
print(counts.to_string())

for label, expected in EXPECTED_COUNTS.items():
    actual = int(counts[label])

    if actual != expected:
        print(
            f"CẢNH BÁO: {label}: tìm thấy {actual}, "
            f"nguồn chính thức ghi {expected}"
        )

# 3. Tìm ảnh giống hệt nhau bằng SHA-256
exact_duplicates = inventory[
    inventory.duplicated("sha256", keep=False)
].sort_values(["sha256", "path"])

exact_duplicates.to_csv(
    OUTPUT_DIR / "exact_duplicates.csv",
    index=False,
)

# Nếu cùng một file nhưng mang hai nhãn khác nhau thì phải kiểm tra thủ công
label_conflicts = (
    exact_duplicates.groupby("sha256")["original_label"].nunique()
)

if (label_conflicts > 1).any():
    conflicting_hashes = set(
        label_conflicts[label_conflicts > 1].index
    )

    exact_duplicates[
        exact_duplicates["sha256"].isin(conflicting_hashes)
    ].to_csv(
        OUTPUT_DIR / "label_conflicts.csv",
        index=False,
    )

    raise SystemExit(
        "Có ảnh giống hệt nhau nhưng khác nhãn. "
        "Hãy kiểm tra data/label_conflicts.csv."
    )

# Không xóa file raw; chỉ giữ một bản trong manifest
clean = (
    inventory.sort_values("path")
    .drop_duplicates("sha256", keep="first")
    .copy()
)

# 4. Bỏ qua các ảnh được team loại thủ công
if EXCLUDED_FILE.exists():
    excluded = pd.read_csv(EXCLUDED_FILE)

    if "path" not in excluded.columns:
        raise SystemExit("excluded.csv phải có cột path.")

    excluded_paths = set(excluded["path"].astype(str))
    clean = clean[~clean["path"].isin(excluded_paths)].copy()

# 5. Chia 70% train, 15% validation, 15% test
train, remaining = train_test_split(
    clean,
    test_size=0.30,
    random_state=SEED,
    stratify=clean["original_label"],
)

validation, test = train_test_split(
    remaining,
    test_size=0.50,
    random_state=SEED,
    stratify=remaining["original_label"],
)

parts = []

for frame, split_name in [
    (train, "train"),
    (validation, "val"),
    (test, "test"),
]:
    frame = frame.copy()
    frame["split"] = split_name
    parts.append(frame)

manifest = pd.concat(parts, ignore_index=True)

manifest = manifest[
    [
        "path",
        "original_label",
        "split",
        "width",
        "height",
        "sha256",
        "phash",
    ]
].sort_values(["split", "original_label", "path"])

manifest_path = OUTPUT_DIR / "manifest.csv"
manifest.to_csv(manifest_path, index=False)

# 6. Tìm ảnh gần giống nhau bằng perceptual hash
items = [
    (
        row.path,
        row.original_label,
        row.split,
        int(row.phash, 16),
    )
    for row in manifest.itertuples(index=False)
]

near_duplicate_pairs = []

for index, first in enumerate(items):
    path_a, label_a, split_a, hash_a = first

    for second in items[index + 1:]:
        path_b, label_b, split_b, hash_b = second
        distance = (hash_a ^ hash_b).bit_count()

        if distance <= NEAR_DUPLICATE_DISTANCE:
            near_duplicate_pairs.append({
                "path_a": path_a,
                "label_a": label_a,
                "split_a": split_a,
                "path_b": path_b,
                "label_b": label_b,
                "split_b": split_b,
                "phash_distance": distance,
                "cross_split": split_a != split_b,
            })

near_duplicates = pd.DataFrame(
    near_duplicate_pairs,
    columns=[
        "path_a",
        "label_a",
        "split_a",
        "path_b",
        "label_b",
        "split_b",
        "phash_distance",
        "cross_split",
    ],
)

near_duplicates.to_csv(
    OUTPUT_DIR / "near_duplicate_candidates.csv",
    index=False,
)

# 7. Báo cáo phân bố
split_stats = pd.crosstab(
    manifest["original_label"],
    manifest["split"],
).reindex(columns=["train", "val", "test"], fill_value=0)

split_stats["total"] = split_stats.sum(axis=1)
split_stats.to_csv(OUTPUT_DIR / "split_stats.csv")

# 8. Lưu phiên bản dataset
version_info = {
    "dataset_name": "RealWaste-ChemSort",
    "version": "1.0.0",
    "source": "https://doi.org/10.24432/C5SS4G",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "target_split": {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
    },
    "valid_raw_images": len(inventory),
    "manifest_images": len(manifest),
    "invalid_images": len(invalid),
    "split_counts": {
        name: int(value)
        for name, value in manifest["split"].value_counts().items()
    },
    "inventory_sha256": sha256_file(
        OUTPUT_DIR / "inventory.csv"
    ),
    "manifest_sha256": sha256_file(manifest_path),
}

with (OUTPUT_DIR / "dataset_version.json").open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(version_info, file, indent=2, ensure_ascii=False)

print("\nPhân bố train/val/test:")
print(split_stats)

print(f"\nĐã tạo: {manifest_path}")
print(
    "Hãy kiểm tra data/near_duplicate_candidates.csv "
    "trước khi xem manifest là phiên bản cuối."
)