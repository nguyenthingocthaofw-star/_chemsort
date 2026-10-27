from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


DATA_ROOT = Path("data/raw/RealWaste")
MANIFEST_PATH = Path("data/manifest.csv")
REPORT_DIR = Path("reports/data")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

REPORT_DIR.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(MANIFEST_PATH)

# Thứ tự này cũng chính là thứ tự output của model
class_names = sorted(manifest["original_label"].unique())
label_to_id = {
    label: index
    for index, label in enumerate(class_names)
}

print("Class mapping:")

for label, index in label_to_id.items():
    print(f"{index}: {label}")

# Lưu lại để app và model dùng đúng thứ tự nhãn
with Path("data/class_names.json").open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(class_names, file, indent=2, ensure_ascii=False)


def load_image(path, label):
    """Đọc một ảnh, chuyển thành RGB và resize về 224x224."""

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    return image, label


def create_dataset(split_name, training=False):
    """Tạo tf.data.Dataset từ một split trong manifest."""

    split_df = manifest[
        manifest["split"] == split_name
    ].copy()

    paths = [
        str(DATA_ROOT / relative_path)
        for relative_path in split_df["path"]
    ]

    labels = [
        label_to_id[label]
        for label in split_df["original_label"]
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(split_df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_dataset = create_dataset("train", training=True)
val_dataset = create_dataset("val")
test_dataset = create_dataset("test")

# Kiểm tra một batch
images, labels = next(iter(train_dataset))

print("\nBatch kiểm tra:")
print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("Image dtype:", images.dtype)
print("Pixel min:", float(tf.reduce_min(images)))
print("Pixel max:", float(tf.reduce_max(images)))

# Vẽ 9 ảnh để kiểm tra trực quan
plt.figure(figsize=(10, 10))

for index in range(9):
    plt.subplot(3, 3, index + 1)
    plt.imshow(tf.cast(images[index], tf.uint8))
    plt.title(class_names[int(labels[index])])
    plt.axis("off")

plt.tight_layout()

output_path = REPORT_DIR / "data_batch.png"
plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nĐã lưu ảnh kiểm tra tại: {output_path}")
