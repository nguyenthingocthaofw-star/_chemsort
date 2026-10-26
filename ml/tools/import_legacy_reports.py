"""Re-score the legacy v1 final-test predictions with the shared evaluator.

The original TensorFlow pipeline (``pipelines/legacy``) saved per-image
probabilities for its two final-tested models. This tool turns them into
``reports/experiments/legacy_v1_*/<artifact>/test/`` so they appear in
``compare_models`` next to new runs, with the same metrics and plots. The
legacy models were trained on dataset v1 and evaluated with their own
preprocessing (stretched resize, 0..255).

    python -m tools.import_legacy_reports
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemsort.contracts import CHEMSORT_CONTRACT  # noqa: E402
from chemsort.evaluation import print_summary, write_evaluation_report  # noqa: E402
from chemsort.paths import REPORTS_DIR, evaluation_dir  # noqa: E402
from chemsort.preprocess import LEGACY_KERAS_INPUT  # noqa: E402

LEGACY_RUNS = (
    {
        "folder": "efficientnet_lite0_float16_tflite_test",
        "run_name": "legacy_v1_efficientnet_lite0",
        "artifact": "tflite_float16",
        "architecture": "EfficientNet-Lite0",
        "backend": "tflite",
    },
    {
        "folder": "mobilenetv2_no_aug_finetuned_test",
        "run_name": "legacy_v1_mobilenetv2_finetuned",
        "artifact": "keras",
        "architecture": "MobileNetV2",
        "backend": "keras",
    },
)


def main() -> int:
    names = CHEMSORT_CONTRACT.model_class_names
    label_to_id = {name: index for index, name in enumerate(names)}
    imported = 0
    for spec in LEGACY_RUNS:
        folder = REPORTS_DIR / "final_test" / spec["folder"]
        predictions_path = folder / f"{spec['folder']}_predictions.csv"
        summary_path = folder / f"{spec['folder']}_summary.json"
        if not predictions_path.is_file():
            print(f"skip {spec['folder']}: no predictions file")
            continue
        legacy = pd.read_csv(predictions_path, encoding="utf-8-sig")
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        probabilities = legacy[[f"prob_9_{name}" for name in names]].to_numpy(dtype=np.float64)
        frame = pd.DataFrame({
            "rel_path": "RealWaste/" + legacy["path"].astype(str),
            "source": "realwaste",
            "split": legacy["split"],
            "label": legacy["original_label"],
            "label_id": legacy["original_label"].map(label_to_id).astype(int),
        })
        model_path = summary.get("model_path") or summary.get("model_artifact")
        document = write_evaluation_report(
            evaluation_dir(spec["run_name"], spec["artifact"], "test"),
            f"{spec['architecture']} legacy v1 [{spec['artifact']}] - test",
            frame,
            probabilities,
            {
                "run_name": spec["run_name"],
                "artifact": spec["artifact"],
                "split": "test",
                "dataset": "v1",
                "backend": spec["backend"],
                "architecture": spec["architecture"],
                "model": {
                    "backend": spec["backend"],
                    "model_path": model_path,
                    "model_sha256": summary.get("model_sha256"),
                    "model_size_mb": summary.get("model_size_mb"),
                },
                "input_spec": LEGACY_KERAS_INPUT.to_dict(),
                "imported_from": str(predictions_path.relative_to(REPORTS_DIR.parent)),
                "test_set_used": True,
            },
        )
        print_summary(document)
        imported += 1
    print(f"\nImported {imported} legacy evaluation(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
