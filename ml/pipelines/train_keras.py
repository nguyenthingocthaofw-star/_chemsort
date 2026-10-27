"""Train a Keras transfer-learning candidate on the ChemSort manifest.

Two stages in one command:

1. head: backbone frozen, classifier head trained (``--epochs``);
2. finetune: last ``--unfreeze-layers`` backbone layers unfrozen, BatchNorm
   kept frozen, low learning rate (``--finetune-epochs``; 0 skips it).

The best weights by validation loss are kept. Afterwards the model is scored
on the validation split with the shared evaluator (app preprocessing), so the
numbers are directly comparable with YOLO and TFLite runs. The test splits
are never read here.

Examples::

    python -m pipelines.train_keras --model mobilenetv3large
    python -m pipelines.train_keras --model efficientnetb0 --manifest v1 --augmentation off
    python -m pipelines.train_keras --model efficientnet_lite0 --finetune-epochs 0
    python -m pipelines.train_keras --model mobilenetv3large --augment-preset strong --mixup 0.2 --cutmix 1.0
    python -m pipelines.train_keras --model efficientnetb0         --manifest data/augmented/v2_medium/manifest_v2_aug_medium.csv

Augmentation (training batches only) comes from ``chemsort.augment``:
``--augment-preset none|light|medium|strong`` builds Keras preprocessing
layers; ``--mixup``/``--cutmix`` (Beta alpha, 0 = off) mix image pairs and
switch the loss to one-hot categorical cross-entropy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipelines._bootstrap import (
    configure_tensorflow_logging,
    configure_utf8_console,
    ensure_local_package,
    prepare_tensorflow_environment,
)

ensure_local_package()

from chemsort.augment import PRESETS  # noqa: E402
from chemsort.keras_models import KERAS_CANDIDATES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(KERAS_CANDIDATES), required=True)
    parser.add_argument("--manifest", default="v2", help="v1, v2 or a CSV path (default v2)")
    parser.add_argument("--run-name", default=None, help="Default: <model>_<dataset>")
    parser.add_argument("--epochs", type=int, default=15, help="Frozen-backbone epochs")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="0 disables fine-tuning")
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--augment-preset", choices=sorted(PRESETS), default="medium")
    parser.add_argument("--augmentation", choices=("on", "off"), default="on", help="off = --augment-preset none")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp Beta alpha, e.g. 0.2 (0 = off)")
    parser.add_argument("--cutmix", type=float, default=0.0, help="CutMix Beta alpha, e.g. 1.0 (0 = off)")
    parser.add_argument("--class-weight", action="store_true", help="Balance classes in the loss")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int, default=None, help="Subsample for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.finetune_epochs < 0:
        parser.error("--finetune-epochs must be >= 0")
    if args.mixup < 0 or args.cutmix < 0:
        parser.error("--mixup and --cutmix must be >= 0")
    if args.augmentation == "off":
        args.augment_preset = "none"
    return args


def main() -> int:
    configure_utf8_console()
    args = parse_args()
    prepare_tensorflow_environment(args.model)

    import numpy as np
    import pandas as pd
    import tensorflow as tf

    configure_tensorflow_logging(tf)

    from chemsort.augment import keras_augmentation, mix_batch
    from chemsort.contracts import MODEL_CLASS_NAMES
    from chemsort.data import load_manifest, require_files, select_split, stratified_sample
    from chemsort.evaluation import evaluate_predictor, print_summary
    from chemsort.keras_models import (
        KERAS_FILE,
        WEIGHTS_FILE,
        build_model,
        unfreeze_top_layers,
        write_run_config,
    )
    from chemsort.paths import evaluation_dir, run_model_dir, run_report_dir
    from chemsort.plots import plot_training_history
    from chemsort.predictors import KerasPredictor
    from chemsort.preprocess import IMAGE_SIZE, KERAS_INPUT

    manifest = load_manifest(args.manifest)
    dataset = manifest.attrs["dataset"]
    run_name = args.run_name or f"{args.model}_{dataset}"
    model_dir = run_model_dir(run_name)
    report_dir = run_report_dir(run_name)
    if (model_dir / WEIGHTS_FILE).exists() and not args.overwrite:
        raise SystemExit(f"Run {run_name} already exists in {model_dir}. Use --overwrite or --run-name.")
    for directory in (model_dir, report_dir):
        if directory.exists() and args.overwrite:
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    train_df = select_split(manifest, "train")
    val_df = select_split(manifest, "val")
    if args.limit_per_class:
        train_df = stratified_sample(train_df, args.limit_per_class, args.seed)
        val_df = stratified_sample(val_df, max(2, args.limit_per_class // 4), args.seed)
    require_files(pd.concat([train_df, val_df]))

    tf.keras.utils.set_random_seed(args.seed)
    augmentation = keras_augmentation(tf, args.augment_preset)
    if augmentation.skipped_layers:
        print(f"Augmentation layers unavailable in this Keras, skipped: {augmentation.skipped_layers}")
    mixing = args.mixup > 0 or args.cutmix > 0

    def load_image(path: tf.Tensor, label: tf.Tensor):
        # Same framing as the app: centre square crop, area resize, raw 0..255.
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        shape = tf.shape(image)
        side = tf.minimum(shape[0], shape[1])
        image = tf.image.crop_to_bounding_box(
            image, (shape[0] - side) // 2, (shape[1] - side) // 2, side, side
        )
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE), method="area")
        return tf.cast(image, tf.float32), label

    def make_dataset(frame: pd.DataFrame, training: bool) -> tf.data.Dataset:
        paths = frame["abs_path"].to_numpy(dtype=str)
        labels = frame["label_id"].to_numpy(dtype=np.int32)
        if mixing:
            labels = np.eye(len(MODEL_CLASS_NAMES), dtype=np.float32)[labels]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            dataset = dataset.shuffle(len(frame), seed=args.seed, reshuffle_each_iteration=True)
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(args.batch_size)
        if training and augmentation.layers:
            dataset = dataset.map(
                lambda images, labels: (augmentation(images, training=True), labels),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        if training and mixing:
            dataset = dataset.map(
                lambda images, labels: mix_batch(tf, images, labels, args.mixup, args.cutmix),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return dataset.prefetch(tf.data.AUTOTUNE)

    train_dataset = make_dataset(train_df, training=True)
    val_dataset = make_dataset(val_df, training=False)

    class_weight = None
    if args.class_weight:
        counts = train_df["label_id"].value_counts().sort_index()
        class_weight = {int(k): float(len(train_df) / (len(counts) * v)) for k, v in counts.items()}

    model, backbone, architecture = build_model(args.model, dropout=args.dropout)
    weights_path = model_dir / WEIGHTS_FILE

    def compile_model(learning_rate: float) -> None:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy() if mixing else tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[(tf.keras.metrics.CategoricalAccuracy if mixing else tf.keras.metrics.SparseCategoricalAccuracy)(name="accuracy")],
        )

    def callbacks(stage: str, best_so_far: float | None):
        checkpoint_options = {}
        if best_so_far is not None:
            checkpoint_options["initial_value_threshold"] = best_so_far
        return [
            tf.keras.callbacks.ModelCheckpoint(
                str(weights_path), monitor="val_loss", save_best_only=True,
                save_weights_only=True, verbose=1, **checkpoint_options,
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.CSVLogger(str(report_dir / f"history_{stage}.csv")),
        ]

    print(f"Run {run_name}: {architecture['display_name']} on dataset {dataset}")
    print(f"  train={len(train_df)}  val={len(val_df)}  augment={args.augment_preset}  "
          f"mixup={args.mixup}  cutmix={args.cutmix}  class_weight={args.class_weight}")

    compile_model(args.learning_rate)
    head_history = model.fit(
        train_dataset, validation_data=val_dataset, epochs=args.epochs,
        callbacks=callbacks("head", None), class_weight=class_weight,
    )
    histories = [("head", head_history.history)]
    best_val_loss = float(min(head_history.history["val_loss"]))
    unfrozen: list[str] = []

    fine_tune = args.finetune_epochs > 0 and architecture["fine_tune_supported"]
    if args.finetune_epochs > 0 and not fine_tune:
        print(f"Fine-tuning skipped: {architecture['display_name']} backbone cannot be unfrozen.")
    if fine_tune:
        model.load_weights(str(weights_path))
        unfrozen = unfreeze_top_layers(tf, backbone, args.unfreeze_layers)
        compile_model(args.finetune_learning_rate)
        start = len(head_history.history["loss"])
        tune_history = model.fit(
            train_dataset, validation_data=val_dataset, initial_epoch=start,
            epochs=start + args.finetune_epochs, callbacks=callbacks("finetune", best_val_loss),
            class_weight=class_weight,
        )
        histories.append(("finetune", tune_history.history))
        best_val_loss = min(best_val_loss, float(min(tune_history.history["val_loss"])))

    model.load_weights(str(weights_path))
    keras_error = None
    try:
        model.save(str(model_dir / KERAS_FILE))
    except Exception as error:  # TF Hub layers may not serialise; weights remain.
        keras_error = f"{type(error).__name__}: {error}"
        print(f"WARNING: full-model save failed ({keras_error}); weights are kept.")

    rows = []
    epoch = 0
    for stage, values in histories:
        for index in range(len(values["loss"])):
            epoch += 1
            rows.append({
                "epoch": epoch, "stage": stage,
                "train_loss": values["loss"][index], "val_loss": values["val_loss"][index],
                "train_accuracy": values["accuracy"][index], "val_accuracy": values["val_accuracy"][index],
            })
    history = pd.DataFrame(rows)
    history.to_csv(report_dir / "history.csv", index=False)
    plot_training_history(history, f"{run_name} training curves", report_dir / "training_curves.png")

    trainable = int(sum(np.prod(variable.shape) for variable in model.trainable_weights))
    config = {
        "run_name": run_name,
        "backend": "keras",
        "model": args.model,
        "architecture": architecture,
        "dataset": dataset,
        "manifest": manifest.attrs["manifest_path"],
        "input_spec": KERAS_INPUT.to_dict(),
        "augmentation": {**PRESETS[args.augment_preset].to_dict(), "skipped_keras_layers": augmentation.skipped_layers},
        "class_names": list(manifest["label"].drop_duplicates().sort_values()),
        "hyperparameters": {
            key: getattr(args, key) for key in (
                "epochs", "finetune_epochs", "unfreeze_layers", "learning_rate",
                "finetune_learning_rate", "batch_size", "dropout", "augment_preset", "mixup", "cutmix",
                "class_weight", "patience", "seed", "limit_per_class",
            )
        },
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "epochs_completed": int(len(history)),
        "best_epoch": int(history.loc[history["val_loss"].idxmin(), "epoch"]),
        "best_val_loss": best_val_loss,
        "total_parameters": int(model.count_params()),
        "trainable_parameters_last_stage": trainable,
        "unfrozen_backbone_layers": len(unfrozen),
        "weights": str(weights_path),
        "keras_model": None if keras_error else str(model_dir / KERAS_FILE),
        "keras_save_error": keras_error,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "test_set_used": False,
    }
    write_run_config(model_dir, config)

    predictor = KerasPredictor(model, KERAS_INPUT, {
        "model_path": str(weights_path),
        "parameters": config["total_parameters"],
        "model_size_mb": weights_path.stat().st_size / (1024 * 1024),
    })
    document = evaluate_predictor(
        predictor, val_df, evaluation_dir(run_name, "keras", "val"),
        f"{architecture['display_name']} ({run_name}) - val",
        {"run_name": run_name, "artifact": "keras", "split": "val", "dataset": dataset,
         "manifest": manifest.attrs["manifest_path"], "test_set_used": False},
        batch_size=args.batch_size,
    )
    print_summary(document)
    print(f"\nWeights:  {weights_path}\nReports:  {report_dir}")
    print(json.dumps({"run_name": run_name, "val_macro_f1_6class": document["summary"]["macro_f1_6class"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
