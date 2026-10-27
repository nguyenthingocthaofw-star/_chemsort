"""Export a trained run to TFLite and verify parity with the reference model.

Writes to ``models/<run>/tflite/``::

    <run>_float32.tflite
    <run>_float16.tflite
    labels_9.txt
    chemsort_mapping.json      9->6 mapping plus the exact input contract

and ``reports/experiments/<run>/export_report.json`` with size, SHA-256,
tensor I/O and parity (prediction agreement and probability difference
against the Keras / PyTorch reference) on a stratified validation subset.

Keras runs use ``TFLiteConverter`` (EfficientNet-Lite0 goes through a Keras 2
SavedModel). YOLO runs use Ultralytics' exporter, whose TFLite keeps the
NCHW input; the evaluator and the app read the layout from the tensor.

Examples::

    python -m pipelines.export_tflite --run yolo11n-cls_v2
    python -m pipelines.export_tflite --run mobilenetv3large_v2 --formats float16
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from pipelines._bootstrap import configure_utf8_console, ensure_local_package, prepare_tensorflow_environment

ensure_local_package()

MIN_TFLITE_BYTES = 500_000
AGREEMENT_GATE_9 = 0.98
AGREEMENT_GATE_6 = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--formats", nargs="+", choices=("float32", "float16"), default=["float32", "float16"])
    parser.add_argument("--parity-per-class", type=int, default=10, help="Validation images per class for parity")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default=None, help="PyTorch device for the YOLO reference")
    return parser.parse_args()


def export_keras(run_dir: Path, config: dict, formats: list[str], output_dir: Path) -> tuple[dict[str, Path], object]:
    prepare_tensorflow_environment(config["model"])
    import tensorflow as tf

    from chemsort.keras_models import load_trained_model

    model = load_trained_model(run_dir)

    def configure(converter, precision: str):
        if precision == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        return converter

    paths: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="chemsort_tflite_") as temp:
        saved_model = None
        if config["model"] == "efficientnet_lite0":
            # Legacy TF Hub layer: the Keras 2 SavedModel keeps its variables.
            saved_model = Path(temp) / "saved_model"
            model.save(str(saved_model), save_format="tf", include_optimizer=False)
        for precision in formats:
            print(f"Converting {precision} ...")
            if saved_model is not None:
                converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
            else:
                # Keras 3 models convert directly; tracing a separate tf.function
                # can drop resource variables and yield a tiny NaN graph.
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
            payload = configure(converter, precision).convert()
            if len(payload) < MIN_TFLITE_BYTES:
                raise SystemExit(f"Refusing implausibly small TFLite export ({len(payload)} bytes)")
            path = output_dir / f"{run_dir.name}_{precision}.tflite"
            path.write_bytes(payload)
            paths[precision] = path
    return paths, model


def export_yolo(run_dir: Path, formats: list[str], output_dir: Path) -> dict[str, Path]:
    from ultralytics import YOLO

    paths: dict[str, Path] = {}
    for precision in formats:
        print(f"Exporting {precision} with Ultralytics ...")
        with tempfile.TemporaryDirectory(prefix="chemsort_yolo_export_") as temp:
            # Export from a copy so Ultralytics' side files stay out of models/.
            weights = Path(temp) / "best.pt"
            shutil.copy2(run_dir / "best.pt", weights)
            try:
                exported = Path(YOLO(str(weights)).export(format="tflite", imgsz=224, half=precision == "float16"))
            except AssertionError as error:
                if "LiteRT" in str(error):
                    raise SystemExit(
                        f"{error}.\nUltralytics exports YOLO to TFLite through LiteRT, which runs only on "
                        "Linux x86-64 and macOS. Run this step in WSL, Linux or Colab "
                        "(training and evaluation work on Windows)."
                    ) from error
                raise
            if exported.is_dir():
                candidates = sorted(exported.glob(f"*{precision}*.tflite")) or sorted(exported.glob("*.tflite"))
                if not candidates:
                    raise SystemExit(f"No .tflite produced in {exported}")
                exported = candidates[0]
            if exported.stat().st_size < MIN_TFLITE_BYTES:
                raise SystemExit(f"Refusing implausibly small TFLite export: {exported}")
            path = output_dir / f"{run_dir.name}_{precision}.tflite"
            shutil.copy2(exported, path)
            paths[precision] = path
    return paths


def main() -> int:
    configure_utf8_console()
    args = parse_args()

    import numpy as np

    from chemsort.contracts import CHEMSORT_CONTRACT
    from chemsort.data import load_manifest, require_files, select_split, stratified_sample
    from chemsort.keras_models import read_run_config
    from chemsort.metrics import aggregate_to_app, evaluate_probabilities
    from chemsort.paths import run_model_dir, run_report_dir
    from chemsort.plots import save_json_ready
    from chemsort.predictors import KerasPredictor, TFLitePredictor, TorchYoloPredictor, predict_paths
    from chemsort.preprocess import InputSpec

    run_dir = run_model_dir(args.run)
    config = read_run_config(run_dir)
    output_dir = run_dir / "tflite"
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = InputSpec.from_dict(config["input_spec"])

    if config["backend"] == "keras":
        paths, model = export_keras(run_dir, config, args.formats, output_dir)
        reference = KerasPredictor(model, spec, {"model_path": config["weights"]})
    else:
        paths = export_yolo(run_dir, args.formats, output_dir)
        reference = TorchYoloPredictor(run_dir / "best.pt", device=args.device)

    manifest = load_manifest(config["manifest"])
    subset = stratified_sample(select_split(manifest, "val"), args.parity_per_class)
    require_files(subset)
    image_paths = subset["abs_path"].tolist()
    true_9 = subset["label_id"].to_numpy()
    print(f"Parity subset: {len(subset)} validation images")
    reference_probs, _ = predict_paths(reference, image_paths, batch_size=32, progress_every=0)
    reference_metrics = evaluate_probabilities(true_9, reference_probs)["summary"]

    report = {
        "run_name": args.run,
        "backend": config["backend"],
        "model": config["model"],
        "parity_subset": {"split": "val", "per_class": args.parity_per_class, "images": len(subset)},
        "reference_metrics": {k: reference_metrics[k] for k in ("accuracy_9class", "macro_f1_9class", "accuracy_6class", "macro_f1_6class")},
        "formats": {},
        "test_set_used": False,
    }
    for precision, path in paths.items():
        tflite = TFLitePredictor(path, threads=args.threads)
        # Pixel range comes from training; the layout from the exported tensor.
        tflite.input_spec = InputSpec(tflite.input_spec.layout, spec.pixel_max, spec.framing, spec.size)
        probabilities, timing = predict_paths(tflite, image_paths, batch_size=32, progress_every=0)
        metrics = evaluate_probabilities(true_9, probabilities)["summary"]
        agreement_9 = float(np.mean(probabilities.argmax(1) == reference_probs.argmax(1)))
        agreement_6 = float(np.mean(
            aggregate_to_app(probabilities).argmax(1) == aggregate_to_app(reference_probs).argmax(1)
        ))
        difference = np.abs(probabilities - reference_probs)
        passed = agreement_9 >= AGREEMENT_GATE_9 and agreement_6 >= AGREEMENT_GATE_6
        report["formats"][precision] = {
            **tflite.description,
            "input_spec": tflite.input_spec.to_dict(),
            "inference_ms_per_image": timing["inference_ms_per_image"],
            "metrics": {k: metrics[k] for k in ("accuracy_9class", "macro_f1_9class", "accuracy_6class", "macro_f1_6class")},
            "parity": {
                "agreement_9class": agreement_9,
                "agreement_6class": agreement_6,
                "max_abs_probability_difference": float(difference.max()),
                "mean_abs_probability_difference": float(difference.mean()),
            },
            "passed_gate": passed,
        }
        status = "PASS" if passed else "FAIL"
        print(f"{precision}: {path.stat().st_size / 2**20:.2f} MB, agreement 9={agreement_9:.4f} "
              f"6={agreement_6:.4f}, max diff={difference.max():.6f} -> {status}")

    passed_formats = [name for name, item in report["formats"].items() if item["passed_gate"]]
    recommended = min(passed_formats, key=lambda name: report["formats"][name]["model_size_mb"]) if passed_formats else None
    report["recommended_format"] = recommended
    report["gate"] = {"agreement_9class": AGREEMENT_GATE_9, "agreement_6class": AGREEMENT_GATE_6}

    layout_spec = InputSpec(**report["formats"][next(iter(paths))]["input_spec"])
    (output_dir / "labels_9.txt").write_text("\n".join(CHEMSORT_CONTRACT.model_class_names) + "\n", encoding="utf-8")
    mapping = {
        "model": {"run_name": args.run, "architecture": config.get("architecture", {}).get("display_name")},
        "input": layout_spec.to_mapping_input(),
        "model_output_9class": {"classes_in_index_order": list(CHEMSORT_CONTRACT.model_class_names)},
        "app_output_6class": {
            "classes_in_index_order": list(CHEMSORT_CONTRACT.app_class_names),
            "groups": {name: list(members) for name, members in CHEMSORT_CONTRACT.app_groups.items()},
            "rule": "Sum probabilities within each group, then take argmax.",
        },
    }
    (output_dir / "chemsort_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_dir = run_report_dir(args.run)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "export_report.json"
    report_path.write_text(json.dumps(save_json_ready(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRecommended format: {recommended or 'none passed the parity gate'}")
    print(f"Artifacts: {output_dir}\nReport:    {report_path}")
    return 0 if passed_formats else 1


if __name__ == "__main__":
    raise SystemExit(main())
