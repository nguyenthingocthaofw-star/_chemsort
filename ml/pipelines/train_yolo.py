"""Train an Ultralytics YOLO classification model on the ChemSort manifest.

The defaults reproduce the model shipped in the app
(``assets/chemsort/yolo11n-cls_data_v2_float32.tflite``): YOLO11n-cls,
ImageNet-pretrained, full fine-tune on dataset v2, 224x224, 60 epochs, batch
64, early-stopping patience 15, RandAugment, random erasing 0.4, horizontal
flip 0.5, HSV jitter, seed 42, deterministic.

Ultralytics reads a ``split/class/image`` tree; it is built from the
manifest with hard links under ``data/build/yolo_<dataset>`` (or pass an
existing tree, e.g. the extracted ``data/processed/dataset``). After training
the best checkpoint is scored on the validation split with the shared
evaluator and the app's preprocessing.

Examples::

    python -m pipelines.train_yolo
    python -m pipelines.train_yolo --model yolov8n-cls --epochs 40
    python -m pipelines.train_yolo --model yolo11s-cls --batch 32 --device 0
    python -m pipelines.train_yolo --augment-preset strong --erasing 0.5
    python -m pipelines.train_yolo --manifest data/augmented/v2_medium/manifest_v2_aug_medium.csv

``--augment-preset`` (none | light | medium | strong, see
``chemsort.augment``) sets every augmentation argument; an explicit flag such
as ``--erasing`` overrides the preset. ``medium`` equals the Ultralytics
defaults used for the app model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipelines._bootstrap import configure_utf8_console, ensure_local_package

ensure_local_package()

from chemsort.augment import YOLO_PRESETS, yolo_hyperparameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolo11n-cls", help="Ultralytics weights name or .pt path")
    parser.add_argument("--manifest", default="v2", help="v1, v2 or a CSV path (default v2)")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Existing split/class/image tree")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--run-name", default=None, help="Default: <model>_<dataset>")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--augment-preset", choices=sorted(YOLO_PRESETS), default="medium")
    parser.add_argument("--auto-augment", default=None, help="randaugment | autoaugment | augmix | none")
    parser.add_argument("--erasing", type=float, default=None)
    parser.add_argument("--fliplr", type=float, default=None)
    parser.add_argument("--flipud", type=float, default=None)
    parser.add_argument("--hsv-h", type=float, default=None)
    parser.add_argument("--hsv-s", type=float, default=None)
    parser.add_argument("--hsv-v", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. 0, cpu (default: auto)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of train images (smoke tests)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    augmentation = yolo_hyperparameters(args.augment_preset)
    for key in augmentation:
        value = getattr(args, key)
        if value is not None:
            augmentation[key] = None if value == "none" else value
    args.augmentation = augmentation
    return args


def normalise_results(results_csv: Path):
    import pandas as pd

    frame = pd.read_csv(results_csv)
    frame.columns = [column.strip() for column in frame.columns]
    history = pd.DataFrame({"epoch": frame["epoch"].astype(int)})
    if history["epoch"].min() == 0:
        history["epoch"] += 1
    mapping = {
        "train/loss": "train_loss",
        "val/loss": "val_loss",
        "metrics/accuracy_top1": "val_accuracy",
        "metrics/accuracy_top5": "val_accuracy_top5",
    }
    for source, target in mapping.items():
        if source in frame.columns:
            history[target] = frame[source]
    return history


def main() -> int:
    configure_utf8_console()
    args = parse_args()

    from chemsort.contracts import CHEMSORT_CONTRACT
    from chemsort.data import load_manifest, select_split
    from chemsort.evaluation import evaluate_predictor, print_summary
    from chemsort.keras_models import write_run_config
    from chemsort.paths import BUILD_DIR, evaluation_dir, run_model_dir, run_report_dir
    from chemsort.plots import plot_training_history
    from chemsort.predictors import TorchYoloPredictor, yolo_class_names
    from chemsort.preprocess import YOLO_INPUT
    from chemsort.yolo_data import build_classification_tree, tree_class_order

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Ultralytics is not installed. Run: python -m pip install -r requirements-yolo.txt"
        ) from error

    manifest = load_manifest(args.manifest)
    dataset = manifest.attrs["dataset"]
    model_name = Path(args.model).stem
    display_name = "YOLO" + model_name[4:] if model_name.lower().startswith("yolo") else model_name
    run_name = args.run_name or f"{model_name}_{dataset}"
    model_dir = run_model_dir(run_name)
    report_dir = run_report_dir(run_name)
    if (model_dir / "best.pt").exists() and not args.overwrite:
        raise SystemExit(f"Run {run_name} already exists in {model_dir}. Use --overwrite or --run-name.")
    for directory in (model_dir, report_dir):
        if directory.exists() and args.overwrite:
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir.resolve()
        tree = {"source": "existing", "path": str(dataset_dir)}
    else:
        dataset_dir = BUILD_DIR / f"yolo_{dataset}"
        print(f"Building Ultralytics folder tree in {dataset_dir} ...")
        tree = build_classification_tree(manifest, dataset_dir, mode=args.link_mode, rebuild=args.rebuild_dataset)
        print(json.dumps(tree["counts"], indent=1))
    order = tree_class_order(dataset_dir)
    CHEMSORT_CONTRACT.require_model_order(order)

    train_options = {
        "data": str(dataset_dir),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "deterministic": True,
        "pretrained": True,
        **args.augmentation,
        "workers": args.workers,
        "fraction": args.fraction,
        "project": str(model_dir),
        "name": "ultralytics",
        "exist_ok": True,
        "plots": True,
    }
    if args.device is not None:
        train_options["device"] = args.device
    if args.lr0 is not None:
        train_options["lr0"] = args.lr0

    yolo = YOLO(args.model)
    yolo.train(**train_options)
    best_source = Path(yolo.trainer.best)
    if not best_source.is_file():
        raise SystemExit(f"Ultralytics did not produce a best checkpoint: {best_source}")
    best_path = model_dir / "best.pt"
    shutil.copy2(best_source, best_path)

    trained = YOLO(str(best_path))
    names = yolo_class_names(trained.names)
    CHEMSORT_CONTRACT.require_model_order(names)

    results_csv = best_source.parent.parent / "results.csv"
    history = None
    if results_csv.is_file():
        history = normalise_results(results_csv)
        history.to_csv(report_dir / "history.csv", index=False)
        plot_training_history(history, f"{run_name} training curves", report_dir / "training_curves.png")

    best_epoch = None
    if history is not None and "val_accuracy" in history:
        best_epoch = int(history.loc[history["val_accuracy"].idxmax(), "epoch"])
    config = {
        "run_name": run_name,
        "backend": "yolo",
        "model": model_name,
        "architecture": {"display_name": display_name, "family": "Ultralytics YOLO"},
        "dataset": dataset,
        "manifest": manifest.attrs["manifest_path"],
        "dataset_dir": str(dataset_dir),
        "dataset_tree": tree,
        "input_spec": YOLO_INPUT.to_dict(),
        "class_names": list(names),
        "hyperparameters": {key: value for key, value in train_options.items() if key not in ("project", "name", "exist_ok", "data")},
        "augment_preset": args.augment_preset,
        "epochs_completed": None if history is None else int(len(history)),
        "best_epoch": best_epoch,
        "total_parameters": int(sum(p.numel() for p in trained.model.parameters())),
        "weights": str(best_path),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "test_set_used": False,
    }
    write_run_config(model_dir, config)

    val_df = select_split(manifest, "val")
    predictor = TorchYoloPredictor(best_path, device=args.device)
    document = evaluate_predictor(
        predictor, val_df, evaluation_dir(run_name, "pytorch", "val"),
        f"{model_name} ({run_name}) - val",
        {"run_name": run_name, "artifact": "pytorch", "split": "val", "dataset": dataset,
         "manifest": manifest.attrs["manifest_path"], "test_set_used": False},
        batch_size=64,
    )
    print_summary(document)
    print(f"\nCheckpoint: {best_path}\nReports:    {report_dir}")
    print(json.dumps({"run_name": run_name, "val_macro_f1_6class": document["summary"]["macro_f1_6class"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
