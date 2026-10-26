"""Copy an exported, parity-checked TFLite model into the Flutter app.

Copies ``models/<run>/tflite/<run>_<format>.tflite``, ``labels_9.txt`` and
``chemsort_mapping.json`` into ``assets/chemsort/`` and points
``AppConfig.modelAsset`` in ``lib/config.dart`` at the new file. Refuses a
format that failed the export parity gate.

The Dart classifier accepts NCHW and NHWC float32 inputs but always feeds
pixels in 0..1, so only models with a 0..1 input contract (YOLO exports) can
be dropped in without changing ``lib/services/yolo_classifier.dart``.

    python -m tools.sync_flutter_assets --run yolo11n-cls_v2 --format float32
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemsort.paths import FLUTTER_ASSETS_DIR, FLUTTER_ROOT, run_model_dir, run_report_dir  # noqa: E402

CONFIG_DART = FLUTTER_ROOT / "lib" / "config.dart"
MODEL_ASSET_PATTERN = re.compile(r"(static const modelAsset = '\$assetsDir/)([^']+)(';)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--format", choices=("float32", "float16"), default="float32")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tflite_dir = run_model_dir(args.run) / "tflite"
    model = tflite_dir / f"{args.run}_{args.format}.tflite"
    report_path = run_report_dir(args.run) / "export_report.json"
    for path in (model, tflite_dir / "labels_9.txt", tflite_dir / "chemsort_mapping.json", report_path):
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run pipelines.export_tflite --run {args.run} first.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    entry = report["formats"].get(args.format, {})
    if not entry.get("passed_gate"):
        raise SystemExit(f"{args.format} did not pass the export parity gate; refusing to ship it.")
    mapping = json.loads((tflite_dir / "chemsort_mapping.json").read_text(encoding="utf-8"))
    if mapping["input"]["pixel_range"][1] != 1.0:
        raise SystemExit(
            "This model expects 0..255 pixels, but the app feeds 0..1. "
            "Update preprocessImage in lib/services/yolo_classifier.dart first."
        )

    asset_name = model.name
    config_text = CONFIG_DART.read_text(encoding="utf-8")
    if not MODEL_ASSET_PATTERN.search(config_text):
        raise SystemExit(f"Could not find modelAsset in {CONFIG_DART}")
    print(f"Model:   {model} -> {FLUTTER_ASSETS_DIR / asset_name}")
    print(f"Labels and mapping -> {FLUTTER_ASSETS_DIR}")
    print(f"Config:  {CONFIG_DART} modelAsset = '$assetsDir/{asset_name}'")
    if args.dry_run:
        return 0

    FLUTTER_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, FLUTTER_ASSETS_DIR / asset_name)
    shutil.copy2(tflite_dir / "labels_9.txt", FLUTTER_ASSETS_DIR / "labels_9.txt")
    shutil.copy2(tflite_dir / "chemsort_mapping.json", FLUTTER_ASSETS_DIR / "chemsort_mapping.json")
    CONFIG_DART.write_text(MODEL_ASSET_PATTERN.sub(rf"\g<1>{asset_name}\g<3>", config_text), encoding="utf-8")
    print("Done. Remove the previous .tflite from assets/chemsort if it is no longer used, then run flutter test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
