"""Build one retrieved list per question from the labelled dataset or a file.

The labelled file stores one passage per class for a parent question. That
group is the candidate pool. BM25 then chooses the top-k order. Gold labels
are carried along for the summary only. The remove and rerank rules do not
read them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_noise.data.labels import LABELS
from rag_noise.selection.retrieve import rank_passages


def _passage_from_row(row: dict[str, Any], line_no: int) -> dict[str, Any]:
    passage_id = str(row.get("example_id") or row.get("passage_id") or "").strip()
    text = str(row.get("passage_text") or "").strip()
    if not passage_id:
        raise ValueError(f"Line {line_no}: example_id is empty")
    if not text:
        raise ValueError(f"Line {line_no}: passage_text is empty")
    gold = row.get("label", row.get("gold_label", row.get("true_label")))
    if gold is not None and gold not in LABELS:
        raise ValueError(f"Line {line_no}: unknown gold label {gold!r}")
    return {
        "passage_id": passage_id,
        "passage_text": text,
        "passage_title": row.get("passage_title") or "",
        "gold_label": gold,
    }


def questions_from_rows(
    rows: list[dict[str, Any]],
    retrieve_k: int,
    k1: float,
    b: float,
) -> list[dict[str, Any]]:
    """Group rows that share a parent question, then rank each group."""
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line_no, row in enumerate(rows, start=1):
        question_id = str(row.get("parent_id") or row.get("question_id") or "").strip()
        question = str(row.get("question") or "").strip()
        if not question_id:
            raise ValueError(f"Line {line_no}: parent_id is empty")
        if not question:
            raise ValueError(f"Line {line_no}: question is empty")
        if question_id not in grouped:
            grouped[question_id] = {
                "question_id": question_id,
                "question": question,
                "split": row.get("split"),
                "source_dataset": row.get("source_dataset"),
                "gold_answers": list(row.get("gold_answers") or []),
                "passages": [],
            }
            order.append(question_id)
        bucket = grouped[question_id]
        if bucket["question"] != question:
            raise ValueError(f"Line {line_no}: {question_id} has two different questions")
        bucket["passages"].append(_passage_from_row(row, line_no))
    questions: list[dict[str, Any]] = []
    for question_id in order:
        bucket = grouped[question_id]
        ranked = rank_passages(bucket["question"], bucket["passages"], retrieve_k, k1=k1, b=b)
        questions.append({
            "question_id": question_id,
            "question": bucket["question"],
            "split": bucket["split"],
            "source_dataset": bucket["source_dataset"],
            "gold_answers": bucket["gold_answers"],
            "passages": ranked,
        })
    if not questions:
        raise ValueError("No questions to retrieve")
    return questions


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_no} must be a JSON object")
            rows.append(row)
    return rows


def load_split_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [row for row in load_jsonl(path) if row.get("split") == split]
    if not rows:
        raise ValueError(f"{path} has no rows in the {split} split")
    return rows


def questions_from_retrieval_file(
    path: Path,
    retrieve_k: int,
    k1: float,
    b: float,
) -> list[dict[str, Any]]:
    """Read a file where each line is one question and its candidate passages.

    If every passage already has retrieval_rank, that order is kept. Otherwise
    the passages are ranked with BM25.
    """
    questions: list[dict[str, Any]] = []
    for line_no, row in enumerate(load_jsonl(path), start=1):
        question_id = str(row.get("question_id") or row.get("parent_id") or "").strip()
        question = str(row.get("question") or "").strip()
        raw_passages = row.get("passages")
        if not question_id or not question:
            raise ValueError(f"Line {line_no}: question_id and question are required")
        if not isinstance(raw_passages, list) or not raw_passages:
            raise ValueError(f"Line {line_no}: passages must be a non-empty list")
        passages = [_passage_from_row(item, line_no) for item in raw_passages]
        ranks = [item.get("retrieval_rank") for item in raw_passages]
        if all(isinstance(rank, int) and not isinstance(rank, bool) and rank >= 1 for rank in ranks):
            ordered = sorted(zip(ranks, passages, raw_passages), key=lambda item: int(item[0]))
            ranked = []
            for rank, passage, source in ordered[:retrieve_k]:
                row_passage = dict(passage)
                row_passage["retrieval_rank"] = int(rank)
                score = source.get("retrieval_score", 0.0)
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError(f"Line {line_no}: retrieval_score must be a number")
                row_passage["retrieval_score"] = float(score)
                ranked.append(row_passage)
        else:
            ranked = rank_passages(question, passages, retrieve_k, k1=k1, b=b)
        questions.append({
            "question_id": question_id,
            "question": question,
            "split": row.get("split"),
            "source_dataset": row.get("source_dataset"),
            "gold_answers": list(row.get("gold_answers") or []),
            "passages": ranked,
        })
    if not questions:
        raise ValueError(f"{path} has no questions")
    return questions


def attach_saved_predictions(
    questions: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    """Copy detector labels onto the retrieved passages. Match on passage id."""
    by_id: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(predictions, start=1):
        passage_id = str(row.get("example_id") or row.get("passage_id") or "").strip()
        label = row.get("predicted_label")
        probability = row.get("predicted_probability")
        if not passage_id:
            raise ValueError(f"Prediction line {line_no}: example_id is empty")
        if label not in LABELS:
            raise ValueError(f"Prediction line {line_no}: unknown predicted_label {label!r}")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"Prediction line {line_no}: predicted_probability must be a number")
        by_id[passage_id] = row
    missing: list[str] = []
    for question in questions:
        for passage in question["passages"]:
            found = by_id.get(passage["passage_id"])
            if found is None:
                missing.append(passage["passage_id"])
                continue
            passage["predicted_label"] = found["predicted_label"]
            passage["predicted_probability"] = float(found["predicted_probability"])
            # The saved detector file stores only the winning class probability,
            # not P(Relevant) for a passage predicted as noise.
            if found["predicted_label"] == "Relevant":
                passage["relevant_probability"] = float(found["predicted_probability"])
            else:
                passage["relevant_probability"] = found.get("relevant_probability")
                if passage["relevant_probability"] is not None:
                    passage["relevant_probability"] = float(passage["relevant_probability"])
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"{len(missing)} retrieved passages have no detector prediction ({preview}). "
            "Score that split first, or run with --rescore."
        )


def attach_probabilities(
    questions: list[dict[str, Any]],
    probabilities: list[dict[str, float]],
) -> None:
    """Attach a full eight-class distribution, in retrieved-passage order."""
    flat = [passage for question in questions for passage in question["passages"]]
    if len(probabilities) != len(flat):
        raise ValueError(
            f"Got {len(probabilities)} probability rows for {len(flat)} passages"
        )
    for passage, distribution in zip(flat, probabilities):
        missing = [label for label in LABELS if label not in distribution]
        if missing:
            raise ValueError(f"Class probabilities are missing {missing}")
        winner = max(LABELS, key=lambda label: float(distribution[label]))
        passage["predicted_label"] = winner
        passage["predicted_probability"] = float(distribution[winner])
        passage["relevant_probability"] = float(distribution["Relevant"])
        passage["class_probabilities"] = {label: float(distribution[label]) for label in LABELS}
