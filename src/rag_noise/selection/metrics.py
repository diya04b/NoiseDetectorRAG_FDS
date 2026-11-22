"""Counts for the remove and rerank decisions.

Gold labels are used only here. A passage with no gold label is skipped by
the gold rates and still counts in the predicted-label chart.
"""

from __future__ import annotations

from typing import Any

from rag_noise.data.labels import LABELS


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _gold_counts(passages: list[dict[str, Any]]) -> tuple[int, int]:
    relevant = sum(1 for passage in passages if passage.get("gold_label") == "Relevant")
    noise = sum(
        1
        for passage in passages
        if passage.get("gold_label") not in {None, "Relevant"}
    )
    return relevant, noise


def policy_metrics(questions: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    """Micro rates over passages, plus how often rank 1 is gold Relevant."""
    retrieved_relevant = 0
    retrieved_noise = 0
    kept_relevant = 0
    removed_noise = 0
    kept_passages = 0
    fallback = 0
    ranked_questions = 0
    relevant_at_1 = 0
    by_gold_kept = {label: 0 for label in LABELS}
    by_gold_removed = {label: 0 for label in LABELS}
    by_predicted_kept = {label: 0 for label in LABELS}
    by_predicted_removed = {label: 0 for label in LABELS}

    for question in questions:
        decision = question[policy]
        kept = decision["kept"]
        removed = decision["removed"]
        kept_passages += len(kept)
        if decision["used_fallback"]:
            fallback += 1
        gold_relevant, gold_noise = _gold_counts(question["passages"])
        retrieved_relevant += gold_relevant
        retrieved_noise += gold_noise
        kept_relevant += sum(1 for passage in kept if passage.get("gold_label") == "Relevant")
        removed_noise += sum(
            1 for passage in removed if passage.get("gold_label") not in {None, "Relevant"}
        )
        if gold_relevant:
            ranked_questions += 1
            if kept and kept[0].get("gold_label") == "Relevant":
                relevant_at_1 += 1
        for passage in kept:
            predicted = passage["predicted_label"]
            by_predicted_kept[predicted] = by_predicted_kept.get(predicted, 0) + 1
            gold = passage.get("gold_label")
            if gold in by_gold_kept:
                by_gold_kept[gold] += 1
        for passage in removed:
            predicted = passage["predicted_label"]
            by_predicted_removed[predicted] = by_predicted_removed.get(predicted, 0) + 1
            gold = passage.get("gold_label")
            if gold in by_gold_removed:
                by_gold_removed[gold] += 1

    n_questions = len(questions)
    return {
        "n_questions": n_questions,
        "n_kept": kept_passages,
        "mean_kept": (kept_passages / n_questions) if n_questions else None,
        "fallback_questions": fallback,
        "relevant_kept": kept_relevant,
        "relevant_retrieved": retrieved_relevant,
        "relevant_retention": _rate(kept_relevant, retrieved_relevant),
        "noise_removed": removed_noise,
        "noise_retrieved": retrieved_noise,
        "noise_rejection": _rate(removed_noise, retrieved_noise),
        "questions_with_gold_relevant": ranked_questions,
        "relevant_at_rank_1": relevant_at_1,
        "relevant_at_rank_1_rate": _rate(relevant_at_1, ranked_questions),
        "kept_by_gold_label": by_gold_kept,
        "removed_by_gold_label": by_gold_removed,
        "kept_by_predicted_label": by_predicted_kept,
        "removed_by_predicted_label": by_predicted_removed,
    }
