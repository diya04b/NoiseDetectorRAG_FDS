"""Build the clean context and one noisy context per noise class.

Section 3.8 of the report keeps the question fixed and changes only the
context. The clean context is the Relevant passage. A noise condition appends
or prepends the passage of that noise class. Questions that have no Relevant
passage, such as PriorQA, are not given a clean baseline.

The detector_final condition is the architecture path: BM25 top-k, then the
saved detector labels, then remove and rerank.
"""

from __future__ import annotations

from typing import Any

from rag_noise.data.labels import LABELS
from rag_noise.selection.policies import pack_context, remove_then_rerank
from rag_noise.selection.retrieve import rank_passages


NOISE_LABELS: tuple[str, ...] = tuple(label for label in LABELS if label != "Relevant")
DETECTOR_CONDITION = "detector_final"
RETRIEVED_CONDITION = "retrieved"
SELECTIVE_CONDITION = "detector_selective"
PAIR_JOIN = " + "

# Harmful noise is dropped. Beneficial noise stays, but below the relevant passage.
BENEFICIAL_NOISE: tuple[str, ...] = (
    "Semantic Noise",
    "Datatype Noise",
    "Illegal-Sentence Noise",
)
HARMFUL_NOISE: tuple[str, ...] = (
    "Counterfactual Noise",
    "Supportive Noise",
    "Orthographic Noise",
)
SELECTIVE_KEEP_LABELS: tuple[str, ...] = ("Relevant", *BENEFICIAL_NOISE)


def _copy_passage(passage: dict[str, Any]) -> dict[str, Any]:
    return {
        "passage_id": passage["passage_id"],
        "passage_text": passage["passage_text"],
        "passage_title": passage.get("passage_title") or "",
        "gold_label": passage.get("gold_label"),
    }


def passages_by_gold_label(passages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for passage in passages:
        label = passage.get("gold_label")
        if label in LABELS and label not in chosen:
            chosen[str(label)] = passage
    return chosen


def _condition(name: str, passages: list[dict[str, Any]], role: str) -> dict[str, Any]:
    copied = [_copy_passage(passage) for passage in passages]
    return {
        "condition": name,
        "role": role,
        "passages": copied,
        "context": pack_context(copied),
    }


def controlled_conditions(
    passages: list[dict[str, Any]],
    noise_position: str,
) -> list[dict[str, Any]]:
    """Clean passage alone, then clean plus one noise class at a time."""
    if noise_position not in {"after", "before"}:
        raise ValueError("noise_position must be after or before")
    by_label = passages_by_gold_label(passages)
    relevant = by_label.get("Relevant")
    if relevant is None:
        return []
    conditions = [_condition("clean", [relevant], "clean")]
    for label in NOISE_LABELS:
        noisy = by_label.get(label)
        if noisy is None:
            continue
        ordered = [noisy, relevant] if noise_position == "before" else [relevant, noisy]
        conditions.append(_condition(label, ordered, "individual_noise"))
    return conditions


def pairwise_conditions(
    passages: list[dict[str, Any]],
    noise_position: str,
) -> list[dict[str, Any]]:
    """Clean passage plus two noise classes at once.

    Pairs use the fixed label order, so Semantic Noise + Datatype Noise is
    the same condition whichever passage was stored first. A pair is skipped
    when the question does not have both classes.
    """
    if noise_position not in {"after", "before"}:
        raise ValueError("noise_position must be after or before")
    by_label = passages_by_gold_label(passages)
    relevant = by_label.get("Relevant")
    if relevant is None:
        return []
    present = [label for label in NOISE_LABELS if label in by_label]
    conditions: list[dict[str, Any]] = []
    for index, left in enumerate(present):
        for right in present[index + 1 :]:
            noises = [by_label[left], by_label[right]]
            ordered = [*noises, relevant] if noise_position == "before" else [relevant, *noises]
            conditions.append(_condition(f"{left}{PAIR_JOIN}{right}", ordered, "pairwise_noise"))
    return conditions


def detector_condition(
    question: str,
    passages: list[dict[str, Any]],
    retrieve_k: int,
    k1: float,
    b: float,
    keep_labels: list[str],
    keep_at_least: int,
    final_k: int | None,
) -> dict[str, Any] | None:
    """Rank, then keep the passages the detector did not call noise.

    Returns None when a retrieved passage has no detector label yet.
    """
    if not passages:
        return None
    if any(passage.get("predicted_label") not in LABELS for passage in passages):
        return None
    ranked = rank_passages(question, passages, retrieve_k, k1=k1, b=b)
    decision = remove_then_rerank(ranked, keep_labels, keep_at_least, final_k=final_k)
    return _condition(DETECTOR_CONDITION, decision["kept"], "detector_final")


def retrieved_condition(
    question: str,
    passages: list[dict[str, Any]],
    retrieve_k: int,
    k1: float,
    b: float,
) -> dict[str, Any]:
    """BM25 top-k in retrieval order. Harmful noise is left in place."""
    ranked = rank_passages(question, passages, retrieve_k, k1=k1, b=b)
    return _condition(RETRIEVED_CONDITION, ranked, "retrieved")


def selective_detector_condition(
    question: str,
    passages: list[dict[str, Any]],
    retrieve_k: int,
    k1: float,
    b: float,
    keep_labels: list[str],
    bottom_labels: list[str],
    keep_at_least: int,
    final_k: int | None,
) -> dict[str, Any] | None:
    """Drop harmful classes, put Relevant first, and leave beneficial noise below it.

    Returns None when a retrieved passage has no detector label yet.
    """
    if not passages:
        return None
    if any(passage.get("predicted_label") not in LABELS for passage in passages):
        return None
    ranked = rank_passages(question, passages, retrieve_k, k1=k1, b=b)
    decision = remove_then_rerank(
        ranked,
        keep_labels,
        keep_at_least,
        final_k=final_k,
        bottom_labels=bottom_labels,
    )
    return _condition(SELECTIVE_CONDITION, decision["kept"], "detector_selective")


def question_conditions(question: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    position = str(config["noise_position"])
    conditions = controlled_conditions(question["passages"], position)
    if config.get("pairwise"):
        conditions.extend(pairwise_conditions(question["passages"], position))
    has_clean = any(item["condition"] == "clean" for item in conditions)
    if config["use_detector"] and has_clean:
        detected = detector_condition(
            question["question"],
            question["passages"],
            int(config["retrieve_k"]),
            float(config["bm25_k1"]),
            float(config["bm25_b"]),
            list(config["keep_labels"]),
            int(config["keep_at_least"]),
            config["final_k"],
        )
        if detected is not None:
            conditions.append(detected)
    return conditions


def clip_context(context: str, max_chars: int) -> tuple[str, bool]:
    if max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if len(context) <= max_chars:
        return context, False
    return context[:max_chars], True
