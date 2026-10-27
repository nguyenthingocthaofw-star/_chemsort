"""Matplotlib figures for evaluation and model comparison reports."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_COLORS = {
    "precision": "#4C78A8",
    "recall": "#F58518",
    "f1": "#54A24B",
    "specificity": "#B279A2",
}


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(totals > 0, matrix / totals, 0.0)


def _draw_matrix(
    axis: plt.Axes,
    values: np.ndarray,
    labels: Sequence[str],
    cell_text: np.ndarray,
    title: str,
    vmax: float | None = None,
    font_size: float = 9,
) -> Any:
    image = axis.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    threshold = (vmax if vmax is not None else values.max() or 1) / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                cell_text[row, column],
                ha="center",
                va="center",
                fontsize=font_size,
                color="white" if values[row, column] > threshold else "black",
            )
    return image


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    """Counts on the left, row-normalised recall percentages on the right."""
    matrix = np.asarray(matrix, dtype=np.int64)
    normalized = _row_normalize(matrix)
    size = 6.5 if len(labels) <= 6 else 8.5
    figure, axes = plt.subplots(1, 2, figsize=(size * 2, size * 0.92))
    _draw_matrix(
        axes[0], matrix, labels, matrix.astype(str), "Counts"
    )
    percent_text = np.vectorize(lambda value: f"{value * 100:.1f}")(normalized)
    _draw_matrix(
        axes[1], normalized, labels, percent_text, "Row-normalised (% of true class)", vmax=1.0
    )
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_per_class_metrics(
    per_class: dict[str, dict[str, float]],
    title: str,
    output_path: Path,
) -> None:
    names = list(per_class)
    keys = ("precision", "recall", "f1")
    positions = np.arange(len(names))
    width = 0.26
    figure, axis = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    for offset, key in enumerate(keys):
        values = [per_class[name][key] for name in names]
        bars = axis.bar(
            positions + (offset - 1) * width,
            values,
            width,
            label=key.capitalize() if key != "f1" else "F1",
            color=METRIC_COLORS[key],
        )
        axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    supports = [per_class[name]["support"] for name in names]
    axis.set_xticks(
        positions,
        [f"{name}\n(n={support})" for name, support in zip(names, supports)],
        rotation=30,
        ha="right",
    )
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_reliability_diagram(
    curve: dict[str, list[float]],
    ece: float,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Perfect calibration")
    axis.plot(curve["confidence"], curve["accuracy"], marker="o", color="#4C78A8", label="Model")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title(f"{title}\nECE = {ece:.4f}")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_training_history(history: pd.DataFrame, title: str, output_path: Path) -> None:
    """Expects columns epoch, train_loss, val_loss and optionally accuracies."""
    has_accuracy = "val_accuracy" in history.columns
    figure, axes = plt.subplots(1, 2 if has_accuracy else 1, figsize=(12 if has_accuracy else 6, 4.5))
    axes = np.atleast_1d(axes)
    loss_axis = axes[0]
    for column, label in (("train_loss", "Train"), ("val_loss", "Validation")):
        if column in history.columns:
            loss_axis.plot(history["epoch"], history[column], label=label)
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.legend()
    loss_axis.grid(alpha=0.3)
    if has_accuracy:
        for column, label in (("train_accuracy", "Train"), ("val_accuracy", "Validation")):
            if column in history.columns:
                axes[1].plot(history["epoch"], history[column], label=label)
        axes[1].set_title("Accuracy (top-1)")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    if "stage" in history.columns and history["stage"].nunique() > 1:
        boundary = history.loc[history["stage"] != history["stage"].iloc[0], "epoch"].min()
        for axis in axes:
            axis.axvline(boundary - 0.5, color="#888888", linestyle=":", label="Fine-tune start")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Cross-model comparison figures
# ---------------------------------------------------------------------------


def plot_metric_comparison(
    table: pd.DataFrame,
    metrics: Sequence[tuple[str, str]],
    title: str,
    output_path: Path,
    intervals: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> None:
    """Grouped bars: one group per metric, one bar per model."""
    models = table["model"].tolist()
    positions = np.arange(len(metrics))
    width = 0.8 / max(1, len(models))
    colors = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(max(9, len(metrics) * 1.6), 5.5))
    for index, (_, row) in enumerate(table.iterrows()):
        values = [row.get(key, np.nan) for key, _ in metrics]
        errors = None
        if intervals and row["model"] in intervals:
            lower, upper = [], []
            for (key, _), value in zip(metrics, values):
                bounds = intervals[row["model"]].get(key)
                lower.append(value - bounds[0] if bounds else 0)
                upper.append(bounds[1] - value if bounds else 0)
            errors = [lower, upper]
        bars = axis.bar(
            positions + (index - (len(models) - 1) / 2) * width,
            values,
            width,
            yerr=errors,
            capsize=2 if errors else 0,
            label=row["model"],
            color=colors(index % 10),
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=6, rotation=90, padding=2)
    axis.set_xticks(positions, [label for _, label in metrics], rotation=20, ha="right")
    finite = table[[key for key, _ in metrics]].to_numpy(dtype=float)
    low = np.nanmin(finite) if np.isfinite(finite).any() else 0
    axis.set_ylim(max(0.0, math.floor((low - 0.08) * 10) / 10), 1.05)
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.legend(fontsize=8, loc="upper left", ncol=min(3, len(models)))
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_per_class_heatmap(
    table: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """Rows are models, columns are classes, cells are F1."""
    values = table.to_numpy(dtype=float)
    figure, axis = plt.subplots(
        figsize=(max(7, values.shape[1] * 1.25), max(2.5, values.shape[0] * 0.55 + 1.8))
    )
    image = axis.imshow(values, cmap="RdYlGn", vmin=max(0.0, np.nanmin(values) - 0.05), vmax=1.0, aspect="auto")
    axis.set_xticks(range(values.shape[1]), table.columns, rotation=30, ha="right")
    axis.set_yticks(range(values.shape[0]), table.index)
    best = np.nanmax(values, axis=0)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            weight = "bold" if np.isclose(value, best[column]) else "normal"
            axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=8, fontweight=weight)
    figure.colorbar(image, ax=axis, fraction=0.03)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_confusion_grid(
    matrices: dict[str, Sequence[Sequence[int]]],
    labels: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    """Row-normalised confusion matrices of several models side by side."""
    count = len(matrices)
    columns = min(3, count)
    rows = math.ceil(count / columns)
    cell = 4.6 if len(labels) <= 6 else 5.6
    figure, axes = plt.subplots(rows, columns, figsize=(cell * columns, cell * rows), squeeze=False)
    for axis in axes.ravel()[count:]:
        axis.axis("off")
    short = [label if len(label) <= 12 else label[:11] + "." for label in labels]
    for axis, (name, matrix) in zip(axes.ravel(), matrices.items()):
        matrix = np.asarray(matrix, dtype=np.int64)
        normalized = _row_normalize(matrix)
        text = np.vectorize(lambda value: f"{value * 100:.0f}")(normalized)
        _draw_matrix(axis, normalized, short, text, name, vmax=1.0, font_size=7 if len(labels) > 6 else 8)
    figure.suptitle(title + "  (% of true class)", fontsize=13)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_efficiency(
    table: pd.DataFrame,
    x_column: str,
    x_label: str,
    y_column: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    subset = table.dropna(subset=[x_column, y_column])
    if subset.empty:
        return
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(subset[x_column], subset[y_column], s=60, color="#4C78A8")
    for _, row in subset.iterrows():
        axis.annotate(row["model"], (row[x_column], row[y_column]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_pvalue_matrix(frame: pd.DataFrame, title: str, output_path: Path) -> None:
    values = frame.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(max(5, len(frame) * 1.1), max(4, len(frame) * 0.9)))
    axis.imshow(values < 0.05, cmap="Greens", vmin=0, vmax=1.6)
    axis.set_xticks(range(len(frame.columns)), frame.columns, rotation=30, ha="right")
    axis.set_yticks(range(len(frame.index)), frame.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if row == column or np.isnan(values[row, column]):
                continue
            axis.text(column, row, f"{values[row, column]:.3g}", ha="center", va="center", fontsize=8)
    axis.set_title(title + "\n(green: p < 0.05)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_json_ready(value: Any) -> Any:
    """Convert numpy scalars recursively so json.dump accepts them."""
    if isinstance(value, dict):
        return {str(key): save_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [save_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return save_json_ready(value.tolist())
    return value
