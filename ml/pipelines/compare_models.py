"""Compare every evaluated model on one split.

Reads ``reports/experiments/*/*/<split>/metrics.json`` and writes
``reports/comparison/<split>/``::

    comparison.csv / comparison.md        ranking with all headline metrics
    metrics_6class.png, metrics_9class.png accuracy / precision / recall / F1 bars (95% CI)
    per_class_f1_6class.png, per_class_f1_9class.png  model x class F1 heatmaps
    confusion_matrices_6class.png, confusion_matrices_9class.png  side by side
    mcnemar_6class.csv/.png, mcnemar_9class.csv  paired significance tests
    efficiency.png                        macro-F1 vs model size

Examples::

    python -m pipelines.compare_models --split test
    python -m pipelines.compare_models --split test_ext --runs yolo mobilenet
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from pipelines._bootstrap import configure_utf8_console, ensure_local_package

ensure_local_package()

from chemsort.data import EVALUATION_SPLITS  # noqa: E402

TABLE_COLUMNS = [
    ("accuracy_6class", "Acc 6"),
    ("macro_precision_6class", "Prec 6"),
    ("macro_recall_6class", "Rec 6"),
    ("macro_f1_6class", "F1 6"),
    ("weighted_f1_6class", "wF1 6"),
    ("cohen_kappa_6class", "Kappa 6"),
    ("mcc_6class", "MCC 6"),
    ("accuracy_9class", "Acc 9"),
    ("macro_precision_9class", "Prec 9"),
    ("macro_recall_9class", "Rec 9"),
    ("macro_f1_9class", "F1 9"),
    ("top3_accuracy_9class", "Top3 9"),
    ("ece_6class", "ECE 6"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=EVALUATION_SPLITS, default="test")
    parser.add_argument("--runs", nargs="*", default=None, help="Keep runs whose name contains any of these")
    parser.add_argument("--artifacts", nargs="*", default=None, help="Keep only these artifacts")
    parser.add_argument("--sort", default="macro_f1_6class")
    parser.add_argument("--include-subsampled", action="store_true", help="Also include --limit-per-class runs")
    return parser.parse_args()


def collect(args: argparse.Namespace) -> list[dict]:
    from chemsort.evaluation import METRICS_FILE
    from chemsort.paths import EXPERIMENTS_DIR

    documents = []
    for path in sorted(EXPERIMENTS_DIR.glob(f"*/*/{args.split}/{METRICS_FILE}")):
        document = json.loads(path.read_text(encoding="utf-8"))
        run, artifact = path.parts[-4], path.parts[-3]
        if args.runs and not any(token in run for token in args.runs):
            continue
        if args.artifacts and artifact not in args.artifacts:
            continue
        if document.get("subsampled_per_class") and not args.include_subsampled:
            continue
        document["_label"] = f"{run} [{artifact}]"
        document["_dir"] = path.parent
        documents.append(document)
    return documents


def markdown_table(frame, columns: list[tuple[str, str]]) -> str:
    header = ["#", "Model", "Images", *[label for _, label in columns], "Size MB", "ms/img"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * 2 + ["---:"] * (len(header) - 2)) + "|"]
    for rank, (_, row) in enumerate(frame.iterrows(), start=1):
        cells = [str(rank), row["model"], str(int(row["n_images"]))]
        for key, _ in columns:
            value = row.get(key)
            cells.append("" if value is None or value != value else f"{value:.4f}")
        size = row.get("model_size_mb")
        latency = row.get("inference_ms_per_image")
        cells.append("" if size != size or size is None else f"{size:.2f}")
        cells.append("" if latency != latency or latency is None else f"{latency:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    configure_utf8_console()
    args = parse_args()

    import numpy as np
    import pandas as pd

    from chemsort.contracts import CHEMSORT_CONTRACT
    from chemsort.evaluation import PREDICTIONS_FILE
    from chemsort.metrics import mcnemar_test
    from chemsort.paths import COMPARISON_DIR
    from chemsort.plots import (
        plot_confusion_grid,
        plot_efficiency,
        plot_metric_comparison,
        plot_per_class_heatmap,
        plot_pvalue_matrix,
    )

    documents = collect(args)
    if not documents:
        raise SystemExit(
            f"No evaluations found for split '{args.split}'. "
            f"Run: python -m pipelines.evaluate --run <run> --split {args.split}"
        )
    output_dir = COMPARISON_DIR / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    intervals: dict[str, dict[str, tuple[float, float]]] = {}
    for document in documents:
        model = document.get("model", {})
        summary = document["summary"]
        row = {
            "model": document["_label"],
            "run_name": document.get("run_name"),
            "artifact": document.get("artifact"),
            "architecture": document.get("architecture"),
            "backend": document.get("backend") or model.get("backend"),
            "train_dataset": document.get("dataset"),
            "n_images": document["n_images"],
            "model_size_mb": model.get("model_size_mb"),
            "parameters": model.get("parameters") or document.get("total_parameters"),
            "inference_ms_per_image": document.get("timing", {}).get("inference_ms_per_image"),
            "model_sha256": model.get("model_sha256"),
            **summary,
        }
        ci = document.get("confidence_intervals_95", {})
        for level in ("6class", "9class"):
            for metric in ("accuracy", "macro_f1"):
                bounds = ci.get(level, {}).get(metric)
                if bounds:
                    row[f"{metric}_{level}_ci_low"] = bounds["lower"]
                    row[f"{metric}_{level}_ci_high"] = bounds["upper"]
                    intervals.setdefault(document["_label"], {})[f"{metric}_{level}"] = (bounds["lower"], bounds["upper"])
        rows.append(row)

    table = pd.DataFrame(rows)
    if args.sort not in table.columns:
        raise SystemExit(f"Unknown sort metric {args.sort!r}")
    ascending = args.sort.startswith(("ece", "log_loss"))
    table = table.sort_values(args.sort, ascending=ascending).reset_index(drop=True)
    order = table["model"].tolist()
    by_label = {document["_label"]: document for document in documents}
    table.to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8")

    sizes = sorted({int(n) for n in table["n_images"]})
    note = "" if len(sizes) == 1 else (
        f"\n> Warning: models were scored on different image counts {sizes}; "
        "compare only rows with the same count.\n"
    )
    markdown = [
        f"# Model comparison - split `{args.split}`",
        "",
        f"Sorted by `{args.sort}`. 6 = app groups (probabilities summed per group), 9 = RealWaste classes.",
        "Macro averages weight every class equally; CI = 95% bootstrap interval (2000 resamples).",
        note,
        markdown_table(table, TABLE_COLUMNS),
        "",
        "## Macro-F1 (6 groups) with 95% CI",
        "",
    ]
    for _, row in table.iterrows():
        low, high = row.get("macro_f1_6class_ci_low"), row.get("macro_f1_6class_ci_high")
        markdown.append(f"- **{row['model']}**: {row['macro_f1_6class']:.4f} [{low:.4f}, {high:.4f}]")

    for level, names in (("6class", CHEMSORT_CONTRACT.app_class_names), ("9class", CHEMSORT_CONTRACT.model_class_names)):
        metrics = [
            (f"accuracy_{level}", "Accuracy"),
            (f"balanced_accuracy_{level}", "Balanced acc."),
            (f"macro_precision_{level}", "Macro precision"),
            (f"macro_recall_{level}", "Macro recall"),
            (f"macro_f1_{level}", "Macro F1"),
            (f"weighted_f1_{level}", "Weighted F1"),
            (f"cohen_kappa_{level}", "Cohen kappa"),
            (f"mcc_{level}", "MCC"),
        ]
        label = "6 app groups" if level == "6class" else "9 RealWaste classes"
        plot_metric_comparison(table, metrics, f"{label} - split {args.split}", output_dir / f"metrics_{level}.png", intervals)

        f1_table = pd.DataFrame(
            {name: [by_label[model][f"metrics_{level}"]["per_class"][name]["f1"] for model in order] for name in names},
            index=order,
        )
        f1_table.to_csv(output_dir / f"per_class_f1_{level}.csv", encoding="utf-8")
        plot_per_class_heatmap(f1_table, f"Per-class F1, {label} - split {args.split}", output_dir / f"per_class_f1_{level}.png")
        markdown += ["", f"## Per-class F1 - {label}", "", "| Model | " + " | ".join(names) + " |",
                     "|---|" + "---:|" * len(names)]
        for model in order:
            markdown.append(f"| {model} | " + " | ".join(f"{f1_table.loc[model, name]:.3f}" for name in names) + " |")

        plot_confusion_grid(
            {model: by_label[model][f"metrics_{level}"]["confusion_matrix"] for model in order},
            names,
            f"Confusion matrices, {label} - split {args.split}",
            output_dir / f"confusion_matrices_{level}.png",
        )

    # Paired McNemar tests on images scored by both models.
    predictions = {}
    for model in order:
        path = by_label[model]["_dir"] / PREDICTIONS_FILE
        if path.is_file():
            predictions[model] = pd.read_csv(path, usecols=["rel_path", "correct_9", "correct_6"]).set_index("rel_path")
    image_sets = {model: frozenset(frame.index) for model, frame in predictions.items()}
    if len(set(image_sets.values())) > 1:
        reference = max(image_sets.values(), key=len)
        lines = [
            f"- {model}: {len(images & reference)}/{len(reference)} images shared with the largest set"
            for model, images in image_sets.items() if images != reference
        ]
        markdown[4:4] = [
            "",
            "> Warning: not every model was scored on the same images (e.g. a report made from an",
            "> older manifest). Headline metrics are not strictly comparable; McNemar tests use",
            "> only the shared images.",
            ">",
            *[f"> {line}" for line in lines],
            "",
        ]
        print("WARNING: models were scored on different image sets:\n" + "\n".join(lines))

    for level in ("6class", "9class"):
        column = f"correct_{level[0]}"
        records = []
        matrix = pd.DataFrame(np.nan, index=list(predictions), columns=list(predictions))
        for first, second in itertools.combinations(predictions, 2):
            common = predictions[first].index.intersection(predictions[second].index)
            if len(common) == 0:
                continue
            test = mcnemar_test(predictions[first].loc[common, column].to_numpy(),
                                predictions[second].loc[common, column].to_numpy())
            records.append({"model_a": first, "model_b": second, "images": len(common), **test})
            matrix.loc[first, second] = matrix.loc[second, first] = test["p_value"]
        if records:
            pd.DataFrame(records).to_csv(output_dir / f"mcnemar_{level}.csv", index=False, encoding="utf-8")
            if level == "6class" and len(predictions) > 1:
                plot_pvalue_matrix(matrix, f"McNemar p-values (6 groups) - split {args.split}", output_dir / "mcnemar_6class.png")
                markdown += ["", "## McNemar tests (6 groups)", "",
                             "p < 0.05 means the two models' error rates differ significantly on the same images.", "",
                             "| Model A | Model B | Only A correct | Only B correct | p-value |", "|---|---|---:|---:|---:|"]
                for record in records:
                    markdown.append(f"| {record['model_a']} | {record['model_b']} | {record['only_a_correct']} | "
                                    f"{record['only_b_correct']} | {record['p_value']:.4g} |")

    plot_efficiency(table, "model_size_mb", "Model file size (MB)", "macro_f1_6class", "Macro F1 (6 groups)",
                    f"Accuracy vs size - split {args.split}", output_dir / "efficiency.png")
    (output_dir / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(table[["model", "n_images", "accuracy_6class", "macro_precision_6class", "macro_recall_6class",
                 "macro_f1_6class", "accuracy_9class", "macro_f1_9class"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest by {args.sort}: {table.loc[0, 'model']}")
    print(f"Comparison written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
