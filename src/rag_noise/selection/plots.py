"""Chart of how many passages of each class the remove step kept."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rag_noise.data.labels import LABELS
from rag_noise.detector.plots import SHORT_LABELS


def keep_remove_figure(
    kept_by_label: dict[str, int],
    removed_by_label: dict[str, int],
    title: str,
    path: Path,
) -> None:
    missing = [label for label in LABELS if label not in kept_by_label or label not in removed_by_label]
    if missing:
        raise KeyError(f"Keep/remove counts are missing {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [kept_by_label[label] for label in LABELS]
    removed = [removed_by_label[label] for label in LABELS]
    positions = range(len(LABELS))
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(positions, kept, color="#3C6E71", label="Kept")
    axis.bar(positions, removed, bottom=kept, color="#C44536", label="Removed")
    axis.set_xticks(list(positions), [SHORT_LABELS[label] for label in LABELS], rotation=30, ha="right")
    axis.set_ylabel("Passages")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
