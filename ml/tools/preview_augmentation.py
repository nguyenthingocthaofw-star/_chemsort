"""Preview augmentation presets on real training images.

For one train image per class, draws the original model input next to
``--samples`` random augmentations of each preset, so a policy can be
checked by eye before training (labels must still be recognisable).

Writes ``reports/data/augmentation/preview_<preset>.png`` per preset and
``presets.json`` with every preset's parameters.

    python -m tools.preview_augmentation
    python -m tools.preview_augmentation --presets strong --samples 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemsort.augment import PRESETS, augment_image  # noqa: E402
from chemsort.paths import REPORTS_DIR  # noqa: E402

OUTPUT_DIR = REPORTS_DIR / "data" / "augmentation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="v2")
    parser.add_argument("--presets", nargs="+", choices=sorted(set(PRESETS) - {"none"}), default=["light", "medium", "strong"])
    parser.add_argument("--samples", type=int, default=5, help="Augmented versions per image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from chemsort.augment import limit_size
    from chemsort.data import load_manifest, require_files, select_split, stratified_sample
    from chemsort.preprocess import app_preprocess, load_rgb

    train = select_split(load_manifest(args.manifest), "train")
    picks = stratified_sample(train, 1, args.seed).sort_values("label").reset_index(drop=True)
    require_files(picks)
    images = [limit_size(Image.fromarray(load_rgb(path)), 512) for path in picks["abs_path"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for preset in args.presets:
        rng = np.random.default_rng(args.seed)
        columns = args.samples + 1
        fig, axes = plt.subplots(len(picks), columns, figsize=(1.9 * columns, 1.95 * len(picks)), squeeze=False)
        for row, (label, image) in enumerate(zip(picks["label"], images)):
            variants = [image] + [augment_image(image, preset, rng) for _ in range(args.samples)]
            for column, variant in enumerate(variants):
                axis = axes[row, column]
                axis.imshow(app_preprocess(variant))
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title("original" if column == 0 else f"aug {column}", fontsize=8)
                if column == 0:
                    axis.set_ylabel(label, fontsize=8)
        fig.suptitle(f"Augmentation preset '{preset}' (model input 224x224)", fontsize=10)
        fig.tight_layout()
        path = args.output_dir / f"preview_{preset}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"Wrote {path}")

    presets = {name: config.to_dict() for name, config in PRESETS.items()}
    (args.output_dir / "presets.json").write_text(json.dumps(presets, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
