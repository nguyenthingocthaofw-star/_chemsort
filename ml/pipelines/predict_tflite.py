"""Run a ChemSort TFLite artifact on one image.

Defaults to the model bundled in the Flutter app. The input contract (layout,
pixel range, framing) is read from ``chemsort_mapping.json``.

    python -m pipelines.predict_tflite path/to/image.jpg
    python -m pipelines.predict_tflite image.jpg --model models/<run>/tflite/<run>_float16.tflite \
        --mapping models/<run>/tflite/chemsort_mapping.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipelines._bootstrap import configure_utf8_console, ensure_local_package

ensure_local_package()

from chemsort import CHEMSORT_CONTRACT  # noqa: E402
from chemsort.inference import TFLiteImageClassifier  # noqa: E402
from chemsort.paths import FLUTTER_ASSETS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Image to classify")
    parser.add_argument("--model", type=Path, default=FLUTTER_ASSETS_DIR / "yolo11n-cls_data_v2_float32.tflite")
    parser.add_argument("--labels", type=Path, default=None, help="Default: labels_9.txt next to the model")
    parser.add_argument("--mapping", type=Path, default=None, help="Default: chemsort_mapping.json next to the model")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    configure_utf8_console()
    args = parse_args()
    labels = args.labels or args.model.parent / "labels_9.txt"
    mapping = args.mapping or args.model.parent / "chemsort_mapping.json"
    classifier = TFLiteImageClassifier(
        args.model,
        labels,
        mapping_path=mapping if mapping.is_file() else None,
        num_threads=args.threads,
    )
    prediction = classifier.predict(args.image)
    print(
        json.dumps(
            {
                "material": prediction.material,
                "confidence": prediction.confidence,
                "model_class": prediction.model_class,
                "model_confidence": prediction.model_confidence,
                "probabilities_6": dict(zip(CHEMSORT_CONTRACT.app_class_names, prediction.probabilities_6)),
                "input": classifier.input_spec.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
