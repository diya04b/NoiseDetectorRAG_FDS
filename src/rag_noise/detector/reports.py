"""Write detector scores, charts, and prediction rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_noise.data.labels import LABELS
from rag_noise.detector.metrics import build_error_analysis, score_predictions
from rag_noise.detector.plots import confusion_matrix_figure, per_class_f1_figure


def round_numbers(value: Any, places: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {key: round_numbers(item, places) for key, item in value.items()}
    if isinstance(value, list):
        return [round_numbers(item, places) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(round_numbers(payload), indent=2) + "\n", encoding="utf-8")


def prediction_rows(
    examples_by_id: dict[str, dict[str, Any]],
    evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    ids = evaluation["example_ids"]
    width = len(ids)
    if len(evaluation["true_ids"]) != width or len(evaluation["predicted_ids"]) != width:
        raise ValueError("Prediction fields do not have the same length")
    if len(evaluation["probabilities"]) != width:
        raise ValueError("Prediction probabilities do not match the example ids")
    rows: list[dict[str, Any]] = []
    for index, example_id in enumerate(ids):
        source = examples_by_id.get(example_id)
        if source is None:
            raise KeyError(f"Prediction refers to unknown example_id {example_id}")
        true_id = int(evaluation["true_ids"][index])
        predicted_id = int(evaluation["predicted_ids"][index])
        passage = str(source["passage_text"])
        rows.append({
            "example_id": example_id,
            "parent_id": source.get("parent_id"),
            "source_dataset": source.get("source_dataset"),
            "split": source.get("split"),
            "question": source["question"],
            "passage_text": passage,
            "true_label": LABELS[true_id],
            "predicted_label": LABELS[predicted_id],
            "predicted_probability": float(evaluation["probabilities"][index]),
            "correct": true_id == predicted_id,
        })
    return rows


def write_split_report(
    output_dir: Path,
    split: str,
    evaluation: dict[str, Any],
    examples_by_id: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Write the metrics, confusion matrix, error counts, and row-level predictions."""
    scores = score_predictions(evaluation["true_ids"], evaluation["predicted_ids"])
    matrix = scores.pop("confusion_matrix")
    errors = build_error_analysis(evaluation["true_ids"], evaluation["predicted_ids"])
    rows = prediction_rows(examples_by_id, evaluation)
    metrics_path = output_dir / f"{split}_metrics.json"
    per_class_path = output_dir / f"{split}_per_class.json"
    matrix_path = output_dir / f"{split}_confusion_matrix.json"
    matrix_figure = output_dir / f"{split}_confusion_matrix.png"
    f1_figure = output_dir / f"{split}_per_class_f1.png"
    error_path = output_dir / f"{split}_error_analysis.json"
    prediction_path = output_dir / f"{split}_predictions.jsonl"

    payload = {
        "split": split,
        "loss": evaluation["loss"],
        **scores,
    }
    write_json(metrics_path, payload)
    write_json(per_class_path, scores["per_class"])
    write_json(matrix_path, {
        "split": split,
        "labels": list(LABELS),
        "rows": "true class",
        "columns": "predicted class",
        "matrix": matrix,
    })
    write_json(error_path, {"split": split, **errors})
    confusion_matrix_figure(matrix, f"{split} confusion matrix", matrix_figure)
    per_class_f1_figure(scores["per_class"], f"{split} F1 by class", f1_figure)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(round_numbers(row), ensure_ascii=False) + "\n")
    return {
        "metrics": str(metrics_path),
        "per_class": str(per_class_path),
        "confusion_matrix": str(matrix_path),
        "confusion_matrix_figure": str(matrix_figure),
        "f1_figure": str(f1_figure),
        "error_analysis": str(error_path),
        "predictions": str(prediction_path),
    }
