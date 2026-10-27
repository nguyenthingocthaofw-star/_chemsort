"""Evaluate any ChemSort model artifact on val / test / test_ext.

Every artifact is scored with the app's preprocessing and the same metric
code, so YOLO, MobileNet, EfficientNet and their TFLite exports are directly
comparable. Results go to ``reports/experiments/<run>/<artifact>/<split>/``
(metrics.json, predictions.csv, text reports, confusion matrices, per-class
bars and a calibration curve). Compare runs afterwards with
``python -m pipelines.compare_models``.

Examples::

    # A trained run (reference checkpoint or its TFLite exports)
    python -m pipelines.evaluate --run yolo11n-cls_v2 --split test test_ext
    python -m pipelines.evaluate --run mobilenetv3large_v2 --artifact tflite_float16 --split test

    # Any TFLite file, e.g. the model bundled in the Flutter app
    python -m pipelines.evaluate --tflite ../assets/chemsort/yolo11n-cls_data_v2_float32.tflite \
        --name app_yolo11n-cls_v2 --split test test_ext

Use ``val`` for model selection. ``test`` and ``test_ext`` are final
estimates: never tune on them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipelines._bootstrap import configure_utf8_console, ensure_local_package, prepare_tensorflow_environment

ensure_local_package()

from chemsort.data import EVALUATION_SPLITS  # noqa: E402

ARTIFACTS = ("auto", "keras", "pytorch", "tflite_float32", "tflite_float16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", help="Run name under models/")
    target.add_argument("--tflite", type=Path, help="Evaluate an arbitrary TFLite file")
    parser.add_argument("--artifact", choices=ARTIFACTS, default="auto")
    parser.add_argument("--name", help="Report name for --tflite (default: file stem)")
    parser.add_argument("--mapping", type=Path, help="chemsort_mapping.json with the input contract")
    parser.add_argument("--pixel-max", type=float, choices=(1.0, 255.0), help="Input range for --tflite")
    parser.add_argument("--framing", choices=("center_crop", "squash"), default="center_crop",
                        help="Preprocessing framing for --tflite without a mapping")
    parser.add_argument("--split", nargs="+", choices=EVALUATION_SPLITS, default=["val"])
    parser.add_argument("--manifest", default=None, help="Default: the run's manifest, else v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default=None, help="PyTorch device for YOLO checkpoints")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Subsample (smoke tests only)")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if the same model was scored")
    return parser.parse_args()


def build_predictor(args: argparse.Namespace):
    """Return (predictor, run_name, artifact, display_name, run_config)."""
    from chemsort.keras_models import read_run_config
    from chemsort.paths import run_model_dir
    from chemsort.preprocess import FRAMING_CENTER_CROP, FRAMING_SQUASH, InputSpec

    if args.tflite is not None:
        from chemsort.predictors import TFLitePredictor, infer_tflite_spec

        spec = None
        if args.mapping is not None:
            mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
            spec = InputSpec.from_mapping_input(mapping["input"])
        predictor = TFLitePredictor(args.tflite, input_spec=spec, threads=args.threads)
        if spec is None:
            inferred = infer_tflite_spec(predictor.description["input_shape"], args.pixel_max)
            framing = FRAMING_SQUASH if args.framing == "squash" else FRAMING_CENTER_CROP
            predictor.input_spec = InputSpec(inferred.layout, inferred.pixel_max, framing, inferred.size)
        name = args.name or args.tflite.stem
        artifact = "tflite" if args.artifact == "auto" else args.artifact
        return predictor, name, artifact, name, {}

    run_dir = run_model_dir(args.run)
    config = read_run_config(run_dir)
    backend = config["backend"]
    artifact = args.artifact
    if artifact == "auto":
        artifact = "keras" if backend == "keras" else "pytorch"
    display = config.get("architecture", {}).get("display_name", config["model"])

    if artifact.startswith("tflite"):
        from chemsort.predictors import TFLitePredictor

        precision = artifact.split("_", 1)[1]
        model_path = run_dir / "tflite" / f"{args.run}_{precision}.tflite"
        mapping_path = run_dir / "tflite" / "chemsort_mapping.json"
        spec = None
        if mapping_path.is_file():
            spec = InputSpec.from_mapping_input(json.loads(mapping_path.read_text(encoding="utf-8"))["input"])
        if not model_path.is_file():
            raise SystemExit(f"{model_path} not found. Run: python -m pipelines.export_tflite --run {args.run}")
        predictor = TFLitePredictor(model_path, input_spec=spec, threads=args.threads)
        return predictor, args.run, artifact, f"{display} TFLite {precision}", config

    if artifact == "keras":
        if backend != "keras":
            raise SystemExit(f"{args.run} is a {backend} run; use --artifact pytorch or tflite_*")
        prepare_tensorflow_environment(config["model"])
        from chemsort.keras_models import WEIGHTS_FILE, load_trained_model
        from chemsort.predictors import KerasPredictor

        model = load_trained_model(run_dir)
        weights = run_dir / WEIGHTS_FILE
        predictor = KerasPredictor(model, InputSpec.from_dict(config["input_spec"]), {
            "model_path": str(weights),
            "parameters": config.get("total_parameters"),
            "model_size_mb": weights.stat().st_size / (1024 * 1024),
        })
        return predictor, args.run, artifact, f"{display} (Keras)", config

    if backend != "yolo":
        raise SystemExit(f"{args.run} is a {backend} run; use --artifact keras or tflite_*")
    from chemsort.predictors import TorchYoloPredictor

    predictor = TorchYoloPredictor(run_dir / "best.pt", device=args.device)
    return predictor, args.run, artifact, f"{display} (PyTorch)", config


def main() -> int:
    configure_utf8_console()
    args = parse_args()

    from chemsort.data import load_manifest, require_files, select_split, stratified_sample
    from chemsort.evaluation import METRICS_FILE, evaluate_predictor, print_summary
    from chemsort.paths import evaluation_dir

    predictor, run_name, artifact, display, config = build_predictor(args)
    manifest = load_manifest(args.manifest or config.get("manifest") or "v2")

    for split in args.split:
        frame = select_split(manifest, split)
        if args.limit_per_class:
            frame = stratified_sample(frame, args.limit_per_class)
        require_files(frame)
        output_dir = evaluation_dir(run_name, artifact, split)
        existing = output_dir / METRICS_FILE
        model_sha = predictor.description.get("model_sha256")
        if existing.is_file() and not args.overwrite:
            previous = json.loads(existing.read_text(encoding="utf-8"))
            if previous.get("model", {}).get("model_sha256") == model_sha and previous.get("n_images") == len(frame):
                print(f"{split}: already evaluated for this model -> {existing} (use --overwrite to redo)")
                print_summary(previous)
                continue
        if split != "val":
            print(f"NOTE: '{split}' is a held-out split. Report it, do not tune on it.")
        print(f"Evaluating {display} on {split} ({len(frame)} images) ...")
        document = evaluate_predictor(
            predictor, frame, output_dir, f"{display} - {split}",
            {
                "run_name": run_name,
                "artifact": artifact,
                "split": split,
                "dataset": manifest.attrs["dataset"],
                "manifest": manifest.attrs["manifest_path"],
                "backend": config.get("backend", "tflite"),
                "architecture": config.get("architecture", {}).get("display_name", display),
                "total_parameters": config.get("total_parameters"),
                "subsampled_per_class": args.limit_per_class,
                "test_set_used": split != "val",
            },
            batch_size=args.batch_size,
        )
        print_summary(document)
        print(f"Report: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
