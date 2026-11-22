"""Accuracy, precision, recall, F1, and the confusion counts for eight classes."""

from __future__ import annotations

from rag_noise.data.labels import LABEL_TO_ID, LABELS

# The report asks for these mix-ups in particular.
NAMED_CONFUSIONS: tuple[tuple[str, str], ...] = (
    ("Semantic Noise", "Supportive Noise"),
    ("Supportive Noise", "Semantic Noise"),
    ("Counterfactual Noise", "Prior Noise"),
    ("Prior Noise", "Counterfactual Noise"),
)


def confusion_counts(true_ids: list[int], predicted_ids: list[int], n_classes: int | None = None) -> list[list[int]]:
    """Rows are true classes. Columns are predicted classes."""
    size = len(LABELS) if n_classes is None else n_classes
    if len(true_ids) != len(predicted_ids):
        raise ValueError("Prediction count does not match label count")
    if not true_ids:
        raise ValueError("Cannot score an empty set of predictions")
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for true_id, predicted_id in zip(true_ids, predicted_ids, strict=True):
        if not 0 <= int(true_id) < size or not 0 <= int(predicted_id) < size:
            raise ValueError(f"Class id is outside 0..{size - 1}: {true_id}, {predicted_id}")
        matrix[int(true_id)][int(predicted_id)] += 1
    return matrix


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def score_predictions(true_ids: list[int], predicted_ids: list[int]) -> dict:
    """Score one split. Macro averages treat all eight classes equally."""
    matrix = confusion_counts(true_ids, predicted_ids)
    per_class: list[dict] = []
    correct = 0
    total = 0
    for class_id, label in enumerate(LABELS):
        support = sum(matrix[class_id])
        predicted_count = sum(matrix[row][class_id] for row in range(len(LABELS)))
        true_positive = matrix[class_id][class_id]
        precision = _safe_divide(true_positive, predicted_count)
        recall = _safe_divide(true_positive, support)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class.append({
            "label": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
        correct += true_positive
        total += support
    macro_precision = sum(item["precision"] for item in per_class) / len(LABELS)
    macro_recall = sum(item["recall"] for item in per_class) / len(LABELS)
    macro_f1 = sum(item["f1"] for item in per_class) / len(LABELS)
    weighted_precision = _safe_divide(
        sum(item["precision"] * item["support"] for item in per_class), total
    )
    weighted_recall = _safe_divide(
        sum(item["recall"] * item["support"] for item in per_class), total
    )
    weighted_f1 = _safe_divide(sum(item["f1"] * item["support"] for item in per_class), total)
    return {
        "n_examples": total,
        "n_correct": correct,
        "accuracy": _safe_divide(correct, total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def build_error_analysis(true_ids: list[int], predicted_ids: list[int]) -> dict:
    """Count the class mix-ups the report asks us to inspect."""
    matrix = confusion_counts(true_ids, predicted_ids)
    named: list[dict] = []
    for true_label, predicted_label in NAMED_CONFUSIONS:
        true_id = LABEL_TO_ID[true_label]
        predicted_id = LABEL_TO_ID[predicted_label]
        support = sum(matrix[true_id])
        count = matrix[true_id][predicted_id]
        named.append({
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": count,
            "support": support,
            "rate": _safe_divide(count, support),
        })
    off_diagonal: list[dict] = []
    for true_id, row in enumerate(matrix):
        for predicted_id, count in enumerate(row):
            if true_id == predicted_id or count == 0:
                continue
            off_diagonal.append({
                "true_label": LABELS[true_id],
                "predicted_label": LABELS[predicted_id],
                "count": count,
            })
    off_diagonal.sort(key=lambda item: (-item["count"], item["true_label"], item["predicted_label"]))
    per_class = []
    for class_id, label in enumerate(LABELS):
        support = sum(matrix[class_id])
        n_correct = matrix[class_id][class_id]
        per_class.append({
            "label": label,
            "support": support,
            "n_correct": n_correct,
            "n_wrong": support - n_correct,
        })
    return {
        "n_examples": len(true_ids),
        "n_correct": sum(matrix[index][index] for index in range(len(LABELS))),
        "named_confusions": named,
        "top_confusions": off_diagonal[:10],
        "per_class": per_class,
    }
