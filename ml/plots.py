"""Presentation charts from measured metrics/history only."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})


def confusion_plot(metrics: dict, path: Path, title: str):
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        axis.text(col, row, str(value), ha="center", va="center", fontsize=18,
                  color="white" if value > matrix.max() / 2 else "black")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["VACANT", "OCCUPIED"],
             yticklabels=["VACANT", "OCCUPIED"], xlabel="Predicted", ylabel="True", title=title)
    figure.tight_layout(); figure.savefig(path); plt.close(figure)


def training_plot(history_path: Path, path: Path, title: str):
    with history_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    epochs = [int(row["epoch"]) for row in rows]
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="Training")
    axes[0].plot(epochs, [float(row["validation_loss"]) for row in rows], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss"); axes[0].legend()
    axes[1].plot(epochs, [float(row["validation_accuracy"]) * 100 for row in rows], color="#1976a3")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)", ylim=(0, 105))
    figure.suptitle(title); figure.tight_layout(); figure.savefig(path); plt.close(figure)


def comparison_plots(results: dict, directory: Path):
    names = [name for name in ("baseline", "mobilenet") if name in results]
    names += [name for name in results if name not in names]
    if not names:
        return
    figure, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names)); width = 0.18
    for index, metric in enumerate(("accuracy", "precision", "recall", "f1")):
        axis.bar(x + (index - 1.5) * width, [results[name]["in_domain"][metric] * 100 for name in names],
                 width, label=metric.capitalize())
    axis.set(xticks=x, xticklabels=names, ylim=(0, 110), ylabel="Percent", title="Dataset A held-out test")
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12)); figure.tight_layout()
    figure.savefig(directory / "model_comparison.png"); plt.close(figure)
    if all("external" in results[name] for name in names):
        figure, axis = plt.subplots(figsize=(8, 4))
        for index, (split, label) in enumerate((("in_domain", "Dataset A test"), ("external", "Dataset B external"))):
            values = [results[name][split]["f1"] * 100 for name in names]
            bars = axis.bar(x + (index - 0.5) * 0.32, values, 0.32, label=label)
            axis.bar_label(bars, fmt="%.1f%%", padding=3)
        axis.set(xticks=x, xticklabels=names, ylim=(0, 115), ylabel="Occupied-class F1 (%)",
                 title="Cross-dataset generalization (no Dataset B training)")
        axis.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        figure.tight_layout(); figure.savefig(directory / "cross_dataset_comparison.png"); plt.close(figure)


def condition_plot(rows: list[dict], path: Path, title: str):
    if not rows:
        return  # Never manufacture missing analysis categories.
    labels = [f"{row['model']} / {row['condition']}\n(n={row['sample_count']})" for row in rows]
    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 1.1), 4.4))
    bars = axis.bar(labels, [float(row["accuracy"]) * 100 for row in rows], color="#207b9e")
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.set(ylim=(0, 115), ylabel="Accuracy (%)", title=title)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout(); figure.savefig(path); plt.close(figure)
