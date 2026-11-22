"""Write the retrieved list, the two routes, and the final passage list."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_noise.detector.reports import round_numbers, write_json
from rag_noise.selection.metrics import policy_metrics
from rag_noise.selection.plots import keep_remove_figure
from rag_noise.selection.policies import pack_context, remove_docs, remove_then_rerank, rerank_docs


POLICIES: tuple[str, ...] = ("remove", "rerank", "final")


def _public_passage(passage: dict[str, Any]) -> dict[str, Any]:
    row = {
        "passage_id": passage["passage_id"],
        "passage_title": passage.get("passage_title") or "",
        "passage_text": passage["passage_text"],
        "retrieval_rank": int(passage["retrieval_rank"]),
        "retrieval_score": float(passage["retrieval_score"]),
        "predicted_label": passage["predicted_label"],
        "predicted_probability": float(passage["predicted_probability"]),
        "relevant_probability": passage.get("relevant_probability"),
        "gold_label": passage.get("gold_label"),
    }
    if "final_rank" in passage:
        row["final_rank"] = int(passage["final_rank"])
    if "class_probabilities" in passage:
        row["class_probabilities"] = {
            label: float(value) for label, value in passage["class_probabilities"].items()
        }
    return row


def _decision_record(question: dict[str, Any], policy: str) -> dict[str, Any]:
    decision = question[policy]
    kept = [_public_passage(passage) for passage in decision["kept"]]
    removed = [_public_passage(passage) for passage in decision["removed"]]
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "split": question.get("split"),
        "source_dataset": question.get("source_dataset"),
        "gold_answers": question.get("gold_answers") or [],
        "policy": policy,
        "used_fallback": bool(decision["used_fallback"]),
        "n_retrieved": len(question["passages"]),
        "n_kept": len(kept),
        "n_removed": len(removed),
        "context": pack_context(decision["kept"]),
        "passages": kept,
        "removed": removed,
    }


def apply_policies(question: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    passages = question["passages"]
    keep_labels = list(config["keep_labels"])
    keep_at_least = int(config["keep_at_least"])
    final_k = config["final_k"]
    removed = remove_docs(passages, keep_labels, keep_at_least)
    reranked = rerank_docs(passages, final_k=final_k)
    reranked_ids = {str(passage["passage_id"]) for passage in reranked}
    rerank_removed = [
        passage for passage in passages if str(passage["passage_id"]) not in reranked_ids
    ]
    rerank_removed.sort(key=lambda passage: int(passage["retrieval_rank"]))
    final = remove_then_rerank(passages, keep_labels, keep_at_least, final_k=final_k)
    return {
        **question,
        "remove": {
            "kept": removed["kept"],
            "removed": removed["removed"],
            "used_fallback": removed["used_fallback"],
        },
        "rerank": {"kept": reranked, "removed": rerank_removed, "used_fallback": False},
        "final": {
            "kept": final["kept"],
            "removed": final["removed"],
            "used_fallback": final["used_fallback"],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(round_numbers(row), ensure_ascii=False) + "\n")


def write_selection(output_dir: Path, questions: list[dict[str, Any]]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieved = []
    for question in questions:
        retrieved.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "split": question.get("split"),
            "source_dataset": question.get("source_dataset"),
            "gold_answers": question.get("gold_answers") or [],
            "passages": [_public_passage(passage) for passage in question["passages"]],
        })
    paths = {
        "retrieved": output_dir / "retrieved_passages.jsonl",
        "remove": output_dir / "removed_passages.jsonl",
        "rerank": output_dir / "reranked_passages.jsonl",
        "final": output_dir / "final_retrieved_passages.jsonl",
        "summary": output_dir / "selection_summary.json",
        "keep_remove_figure": output_dir / "keep_remove_by_gold_label.png",
    }
    _write_jsonl(paths["retrieved"], retrieved)
    for policy, key in (("remove", "remove"), ("rerank", "rerank"), ("final", "final")):
        _write_jsonl(paths[key], [_decision_record(question, policy) for question in questions])

    summary = {policy: policy_metrics(questions, policy) for policy in POLICIES}
    write_json(paths["summary"], summary)
    final_counts = summary["final"]
    if any(final_counts["kept_by_gold_label"].values()) or any(final_counts["removed_by_gold_label"].values()):
        keep_remove_figure(
            final_counts["kept_by_gold_label"],
            final_counts["removed_by_gold_label"],
            "Final list: gold class kept or removed",
            paths["keep_remove_figure"],
        )
    else:
        keep_remove_figure(
            final_counts["kept_by_predicted_label"],
            final_counts["removed_by_predicted_label"],
            "Final list: predicted class kept or removed",
            paths["keep_remove_figure"],
        )
    return {key: str(path) for key, path in paths.items()}
