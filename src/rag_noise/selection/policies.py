"""Remove noisy passages and rerank the ones that remain.

Figure 1 draws Remove docs and Rerank docs as two routes from the detector
labels into Final Retrieved Passages. Both routes are applied here. The final
list is the remove route followed by the rerank route: drop predicted noise,
then order what is left. Decisions use the detector label, not the gold label.
"""

from __future__ import annotations

from typing import Any

from rag_noise.data.labels import LABELS


def _checked_passage(passage: dict[str, Any]) -> None:
    label = passage.get("predicted_label")
    if label not in LABELS:
        raise ValueError(f"predicted_label must be one of the eight classes, got {label!r}")
    probability = passage.get("predicted_probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError("predicted_probability must be a number")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("predicted_probability must be between 0 and 1")
    relevant = passage.get("relevant_probability")
    if relevant is not None:
        if isinstance(relevant, bool) or not isinstance(relevant, (int, float)):
            raise ValueError("relevant_probability must be a number or null")
        if not 0.0 <= float(relevant) <= 1.0:
            raise ValueError("relevant_probability must be between 0 and 1")
    rank = passage.get("retrieval_rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("retrieval_rank must be a positive integer")
    if not str(passage.get("passage_id", "")).strip():
        raise ValueError("passage_id is empty")
    if not str(passage.get("passage_text", "")).strip():
        raise ValueError("passage_text is empty")


def check_passages(passages: list[dict[str, Any]]) -> None:
    if not passages:
        raise ValueError("A question needs at least one retrieved passage")
    seen: set[str] = set()
    ranks: set[int] = set()
    for passage in passages:
        _checked_passage(passage)
        passage_id = str(passage["passage_id"])
        if passage_id in seen:
            raise ValueError(f"Duplicate passage_id {passage_id}")
        seen.add(passage_id)
        rank = int(passage["retrieval_rank"])
        if rank in ranks:
            raise ValueError(f"Duplicate retrieval_rank {rank}")
        ranks.add(rank)


def _relevance_key(
    passage: dict[str, Any],
    bottom_labels: set[str] | None = None,
) -> tuple[int, float, int]:
    """Predicted Relevant first. Then the passage most likely to be relevant.

    When bottom_labels is set, those classes stay below every predicted Relevant
    passage, and any other kept class stays below them. When the full class
    distribution is missing, a noise label's own confidence is used in reverse:
    a passage the detector is sure is noise sinks below one it is unsure about.
    """
    predicted = passage["predicted_label"]
    if bottom_labels is None:
        noise_band = 0 if predicted == "Relevant" else 1
    elif predicted == "Relevant":
        noise_band = 0
    elif predicted in bottom_labels:
        noise_band = 1
    else:
        noise_band = 2
    relevant = passage.get("relevant_probability")
    if relevant is not None:
        score = -float(relevant)
    elif predicted == "Relevant":
        score = -float(passage["predicted_probability"])
    else:
        score = float(passage["predicted_probability"])
    return (noise_band, score, int(passage["retrieval_rank"]))


def _with_final_rank(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for index, passage in enumerate(passages, start=1):
        row = dict(passage)
        row["final_rank"] = index
        ranked.append(row)
    return ranked


def rerank_docs(
    passages: list[dict[str, Any]],
    final_k: int | None = None,
    bottom_labels: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Reorder every retrieved passage. Nothing is dropped unless final_k is set.

    bottom_labels are placed after predicted Relevant passages. Other classes
    go last. This is how beneficial noise stays in the context without sitting
    above the relevant passage.
    """
    check_passages(passages)
    if final_k is not None and final_k < 1:
        raise ValueError("final_k must be a positive integer or None")
    bottom = None if bottom_labels is None else set(bottom_labels)
    if bottom is not None:
        unknown = [label for label in bottom if label not in LABELS]
        if unknown:
            raise ValueError(f"bottom_labels has unknown classes: {unknown}")
    ordered = sorted(passages, key=lambda passage: _relevance_key(passage, bottom))
    if final_k is not None:
        ordered = ordered[:final_k]
    return _with_final_rank(ordered)


def remove_docs(
    passages: list[dict[str, Any]],
    keep_labels: list[str] | tuple[str, ...],
    keep_at_least: int = 1,
) -> dict[str, Any]:
    """Drop passages whose predicted class is not in keep_labels.

    Kept passages stay in retrieval order. If that would leave fewer than
    keep_at_least passages, the best dropped passage is put back. That
    fallback is recorded so an empty final list is not silent.
    """
    check_passages(passages)
    if keep_at_least < 0:
        raise ValueError("keep_at_least must be >= 0")
    allowed = set(keep_labels)
    unknown = [label for label in allowed if label not in LABELS]
    if unknown or not allowed:
        raise ValueError(f"keep_labels must be non-empty class names, got {list(keep_labels)!r}")
    kept = [passage for passage in passages if passage["predicted_label"] in allowed]
    kept.sort(key=lambda passage: int(passage["retrieval_rank"]))
    used_fallback = False
    if len(kept) < keep_at_least:
        chosen = {str(passage["passage_id"]) for passage in kept}
        for candidate in sorted(passages, key=_relevance_key):
            if str(candidate["passage_id"]) in chosen:
                continue
            kept.append(candidate)
            chosen.add(str(candidate["passage_id"]))
            used_fallback = True
            if len(kept) >= keep_at_least:
                break
        kept.sort(key=lambda passage: int(passage["retrieval_rank"]))
    kept_ids = {str(passage["passage_id"]) for passage in kept}
    removed = [passage for passage in passages if str(passage["passage_id"]) not in kept_ids]
    removed.sort(key=lambda passage: int(passage["retrieval_rank"]))
    return {
        "kept": _with_final_rank(kept),
        "removed": removed,
        "used_fallback": used_fallback,
    }


def remove_then_rerank(
    passages: list[dict[str, Any]],
    keep_labels: list[str] | tuple[str, ...],
    keep_at_least: int = 1,
    final_k: int | None = None,
    bottom_labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """The Final Retrieved Passages list: remove noise, then rerank the rest."""
    removed = remove_docs(passages, keep_labels, keep_at_least)
    if not removed["kept"]:
        return {
            "kept": [],
            "removed": sorted(passages, key=lambda passage: int(passage["retrieval_rank"])),
            "used_fallback": False,
        }
    # Rerank only the passages remove kept. final_k limits that shorter list.
    ordered = rerank_docs(removed["kept"], final_k=final_k, bottom_labels=bottom_labels)
    kept_ids = {str(passage["passage_id"]) for passage in ordered}
    still_removed = [passage for passage in passages if str(passage["passage_id"]) not in kept_ids]
    still_removed.sort(key=lambda passage: int(passage["retrieval_rank"]))
    return {
        "kept": ordered,
        "removed": still_removed,
        "used_fallback": bool(removed["used_fallback"]),
    }


def pack_context(passages: list[dict[str, Any]]) -> str:
    """Join the final passages in rank order. This is the generator's context."""
    blocks: list[str] = []
    for passage in passages:
        title = str(passage.get("passage_title") or "").strip()
        text = str(passage["passage_text"]).strip()
        blocks.append(f"{title}\n{text}" if title else text)
    return "\n\n".join(blocks)
