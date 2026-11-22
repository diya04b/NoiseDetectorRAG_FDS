"""Read labelled question-passage rows and encode them for the detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_noise.data.labels import LABEL_TO_ID, LABELS

SPLITS: tuple[str, ...] = ("train", "validation", "test")


def load_examples(path: Path) -> list[dict[str, Any]]:
    """Read classification.jsonl and check each label against the fixed order."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing labelled data at {path}. Run python scripts/build_classification_set.py first."
        )
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(_checked_example(row, line_no))
    if not examples:
        raise ValueError(f"{path} has no examples")
    return examples


def _checked_example(row: dict[str, Any], line_no: int) -> dict[str, Any]:
    label = row.get("label")
    if label not in LABEL_TO_ID:
        raise ValueError(f"Line {line_no}: unknown label {label!r}")
    expected_id = LABEL_TO_ID[label]
    if int(row.get("label_id", -1)) != expected_id:
        raise ValueError(
            f"Line {line_no}: label_id {row.get('label_id')!r} does not match {label} ({expected_id})"
        )
    if row.get("split") not in SPLITS:
        raise ValueError(f"Line {line_no}: split must be one of {SPLITS}")
    if not str(row.get("question", "")).strip():
        raise ValueError(f"Line {line_no}: question is empty")
    if not str(row.get("passage_text", "")).strip():
        raise ValueError(f"Line {line_no}: passage_text is empty")
    if not str(row.get("example_id", "")).strip():
        raise ValueError(f"Line {line_no}: example_id is empty")
    return row


def split_examples(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in examples:
        grouped[row["split"]].append(row)
    return grouped


def limit_examples(examples: list[dict[str, Any]], limit_per_class: int | None) -> list[dict[str, Any]]:
    """Keep the first N rows of each class inside each split. None keeps every row."""
    if limit_per_class is None:
        return list(examples)
    if limit_per_class < 1:
        raise ValueError("limit_per_class must be a positive integer or None")
    kept: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], int] = {}
    for row in examples:
        key = (row["split"], row["label"])
        count = seen.get(key, 0)
        if count >= limit_per_class:
            continue
        seen[key] = count + 1
        kept.append(row)
    return kept


def require_all_labels(examples: list[dict[str, Any]], split: str) -> None:
    present = {row["label"] for row in examples if row["split"] == split}
    missing = [label for label in LABELS if label not in present]
    if missing:
        raise ValueError(f"{split} is missing classes: {missing}")


def encode_pair(tokenizer: Any, question: str, passage: str, max_length: int) -> dict[str, list[int]]:
    """Tokenize the question and passage as one pair.

    The passage is the second sequence, so truncation removes passage tokens
    before it removes question tokens. The passage text itself is not rewritten.
    """
    if max_length < 32:
        raise ValueError("max_length must be at least 32")
    encoded = tokenizer(
        question,
        passage,
        truncation="only_second",
        max_length=max_length,
        add_special_tokens=True,
    )
    features: dict[str, list[int]] = {}
    for key, value in encoded.items():
        if isinstance(value, list):
            features[key] = value
    if "input_ids" not in features or "attention_mask" not in features:
        raise ValueError("Tokenizer did not return input_ids and attention_mask")
    return features


def assert_label_order(id2label: dict[Any, str]) -> None:
    """Check a saved model still uses this project's class order."""
    for index, expected in enumerate(LABELS):
        found = id2label.get(index, id2label.get(str(index)))
        if found != expected:
            raise ValueError(f"Checkpoint label {index} is {found!r}, expected {expected!r}")
