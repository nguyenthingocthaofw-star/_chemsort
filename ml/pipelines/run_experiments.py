"""Run the whole benchmark: train, export, evaluate and compare many models.

Each step runs as its own subprocess, so TensorFlow and PyTorch never share
a process, and the two stacks may live in different virtual environments
(``--tf-python`` / ``--yolo-python``).

Examples::

    # Full benchmark on dataset v2 (default model list)
    python -m pipelines.run_experiments

    # Only some models, skip training (already trained), final test splits
    python -m pipelines.run_experiments --models yolo11n-cls mobilenetv3large \
        --steps export evaluate compare --splits val test test_ext

    # Quick end-to-end smoke test on a tiny subset
    python -m pipelines.run_experiments --models yolo11n-cls mobilenetv3small --smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pipelines._bootstrap import configure_utf8_console, ensure_local_package

ensure_local_package()

from chemsort.data import dataset_tag, resolve_manifest  # noqa: E402
from chemsort.keras_models import KERAS_CANDIDATES  # noqa: E402
from chemsort.paths import ML_ROOT  # noqa: E402

DEFAULT_MODELS = [
    "yolo11n-cls",
    "yolov8n-cls",
    "mobilenetv2",
    "mobilenetv3large",
    "efficientnetb0",
    "efficientnet_lite0",
]
STEPS = ("train", "export", "evaluate", "compare")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Keras keys {sorted(KERAS_CANDIDATES)} or Ultralytics *-cls names")
    parser.add_argument("--manifest", default="v2")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--splits", nargs="+", default=["val", "test", "test_ext"])
    parser.add_argument("--artifacts", nargs="+", default=["reference", "tflite_float32", "tflite_float16"],
                        help="reference = Keras/PyTorch checkpoint")
    parser.add_argument("--tf-python", default=sys.executable, help="Python with TensorFlow")
    parser.add_argument("--yolo-python", default=sys.executable, help="Python with Ultralytics")
    parser.add_argument("--device", default=None, help="YOLO device, e.g. 0 or cpu")
    parser.add_argument("--smoke", action="store_true", help="Tiny subset, 1 epoch: checks the plumbing only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue with other models after a failure")
    return parser.parse_args()


def is_yolo(model: str) -> bool:
    return model.endswith("-cls") or model.endswith(".pt")


def main() -> int:
    configure_utf8_console()
    args = parse_args()
    unknown = [m for m in args.models if not is_yolo(m) and m not in KERAS_CANDIDATES]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")

    dataset = dataset_tag(resolve_manifest(args.manifest))
    suffix = "_smoke" if args.smoke else ""
    failures: list[str] = []

    def run(python: str, module: str, *extra: str) -> bool:
        command = [python, "-m", module, *extra]
        print("\n$ " + " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ML_ROOT)
        if completed.returncode != 0:
            failures.append(" ".join(command))
            if not args.keep_going:
                raise SystemExit(f"Step failed ({completed.returncode}): {' '.join(command)}")
            return False
        return True

    for model in args.models:
        yolo = is_yolo(model)
        python = args.yolo_python if yolo else args.tf_python
        run_name = f"{Path(model).stem}_{dataset}{suffix}"
        common = ["--run-name", run_name, "--manifest", args.manifest]
        if args.overwrite:
            common.append("--overwrite")

        if "train" in args.steps:
            if yolo:
                extra = ["--model", model, *common]
                if args.device:
                    extra += ["--device", args.device]
                if args.smoke:
                    extra += ["--epochs", "1", "--batch", "16", "--fraction", "0.02", "--workers", "0"]
                ok = run(python, "pipelines.train_yolo", *extra)
            else:
                extra = ["--model", model, *common]
                if args.smoke:
                    extra += ["--epochs", "1", "--finetune-epochs", "1", "--limit-per-class", "8"]
                ok = run(python, "pipelines.train_keras", *extra)
            if not ok:
                continue

        if "export" in args.steps and any(a.startswith("tflite") for a in args.artifacts):
            formats = [a.split("_", 1)[1] for a in args.artifacts if a.startswith("tflite")]
            extra = ["--run", run_name, "--formats", *formats]
            if args.smoke:
                extra += ["--parity-per-class", "2"]
            if yolo and args.device:
                extra += ["--device", args.device]
            run(python, "pipelines.export_tflite", *extra)

        if "evaluate" in args.steps:
            for artifact in args.artifacts:
                name = ("pytorch" if yolo else "keras") if artifact == "reference" else artifact
                extra = ["--run", run_name, "--artifact", name, "--split", *args.splits]
                if args.smoke:
                    extra += ["--limit-per-class", "3"]
                if args.overwrite:
                    extra.append("--overwrite")
                if yolo and args.device:
                    extra += ["--device", args.device]
                run(python, "pipelines.evaluate", *extra)

    if "compare" in args.steps:
        for split in args.splits:
            extra = ["--split", split]
            if args.smoke:
                extra += ["--runs", "_smoke", "--include-subsampled"]
            run(sys.executable, "pipelines.compare_models", *extra)

    if failures:
        print("\nFailed steps:\n  " + "\n  ".join(failures))
        return 1
    print("\nAll steps finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
