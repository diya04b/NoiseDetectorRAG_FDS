"""Charts for detector scores. The numbers come from the saved predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rag_noise.data.labels import LABELS

SHORT_LABELS: dict[str, str] = {
    "Relevant": "Relevant",
    "Semantic Noise": "Semantic",
    "Datatype Noise": "Datatype",
    "Illegal-Sentence Noise": "Illegal",
    "Counterfactual Noise": "Counterfactual",
    "Supportive Noise": "Supportive",
    "Orthographic Noise": "Orthographic",
    "Prior Noise": "Prior",
}


def confusion_matrix_figure(matrix: list[list[int]], title: str, path: Path) -> None:
    if len(matrix) != len(LABELS) or any(len(row) != len(LABELS) for row in matrix):
        raise ValueError("Confusion matrix must be 8 by 8 in the fixed label order")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    tick_labels = [SHORT_LABELS[label] for label in LABELS]
    axis.set_xticks(range(len(LABELS)), tick_labels, rotation=30, ha="right")
    axis.set_yticks(range(len(LABELS)), tick_labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)
    highest = max(value for row in matrix for value in row)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if highest and value > highest / 2 else "black"
            axis.text(column_index, row_index, str(value), ha="center", va="center", color=color, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def per_class_f1_figure(per_class: list[dict], title: str, path: Path) -> None:
    by_label = {item["label"]: item["f1"] for item in per_class}
    missing = [label for label in LABELS if label not in by_label]
    if missing:
        raise KeyError(f"Per-class scores are missing {missing}")
    values = [by_label[label] for label in LABELS]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(range(len(LABELS)), values, color="#3C6E71")
    axis.set_xticks(range(len(LABELS)), list(LABELS), rotation=30, ha="right")
    axis.set_ylabel("F1")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
