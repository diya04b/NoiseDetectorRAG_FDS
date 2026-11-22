"""Exact Match and token F1, plus the change against the clean baseline.

Normalization follows the SQuAD scoring rules: lower case, no punctuation,
no articles, and collapsed whitespace. A prediction is scored against every
gold answer and the best score is kept.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from rag_noise.rag.contexts import PAIR_JOIN


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans({character: " " for character in string.punctuation})


def normalize_answer(text: str) -> str:
    lowered = str(text).lower().translate(_PUNCTUATION)
    without_articles = _ARTICLES.sub(" ", lowered)
    return " ".join(without_articles.split())


def _token_f1(prediction: str, gold: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not predicted_tokens and not gold_tokens:
        return 1.0
    if not predicted_tokens or not gold_tokens:
        return 0.0
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    predicted = normalize_answer(prediction)
    return 1.0 if any(predicted == normalize_answer(gold) for gold in gold_answers) else 0.0


def token_f1(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(_token_f1(prediction, gold) for gold in gold_answers)


def whitespace_tokens(text: str) -> int:
    return len(str(text).split())


def score_answer(prediction: str, gold_answers: list[str]) -> dict[str, float]:
    return {
        "exact_match": exact_match(prediction, gold_answers),
        "token_f1": token_f1(prediction, gold_answers),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_conditions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean scores per condition, and the change from the clean baseline.

    The change is the report's delta: score on the noisy context minus the
    score on the clean context, averaged over questions that have both.
    """
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    clean_by_question = {
        str(row["question_id"]): row for row in by_condition.get("clean", [])
    }
    summary: dict[str, Any] = {}
    for condition, condition_rows in by_condition.items():
        paired_em: list[float] = []
        paired_f1: list[float] = []
        for row in condition_rows:
            clean = clean_by_question.get(str(row["question_id"]))
            if clean is None or condition == "clean":
                continue
            paired_em.append(float(row["exact_match"]) - float(clean["exact_match"]))
            paired_f1.append(float(row["token_f1"]) - float(clean["token_f1"]))
        block: dict[str, Any] = {
            "n": len(condition_rows),
            "exact_match": _mean([float(row["exact_match"]) for row in condition_rows]),
            "token_f1": _mean([float(row["token_f1"]) for row in condition_rows]),
            "mean_context_tokens": _mean([float(row["context_tokens"]) for row in condition_rows]),
            "mean_answer_tokens": _mean([float(row["answer_tokens"]) for row in condition_rows]),
            "mean_latency_seconds": _mean([float(row["latency_seconds"]) for row in condition_rows]),
        }
        if condition != "clean":
            block["n_paired"] = len(paired_f1)
            block["delta_exact_match"] = _mean(paired_em)
            block["delta_token_f1"] = _mean(paired_f1)
        summary[condition] = block
    return summary


def _pair_parts(condition: str) -> tuple[str, str] | None:
    if PAIR_JOIN not in condition:
        return None
    left, right = condition.split(PAIR_JOIN, 1)
    if not left or not right or PAIR_JOIN in right:
        return None
    return left, right


def summarize_interactions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare a two-noise context with the sum of the two single-noise changes.

    Observed delta is the combined score minus the clean score. Expected delta
    is the sum of the two individual deltas. Interaction is observed minus
    expected. A value near zero means the two noises add. A negative value
    means the combination hurt more than adding the separate effects.
    """
    by_key = {(str(row["question_id"]), str(row["condition"])): row for row in rows}
    collected: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("role") != "pairwise_noise":
            continue
        parts = _pair_parts(str(row["condition"]))
        if parts is None:
            continue
        left, right = parts
        question_id = str(row["question_id"])
        clean = by_key.get((question_id, "clean"))
        left_row = by_key.get((question_id, left))
        right_row = by_key.get((question_id, right))
        if clean is None or left_row is None or right_row is None:
            continue
        bucket = collected.setdefault(
            str(row["condition"]),
            {"observed_em": [], "expected_em": [], "interaction_em": [],
             "observed_f1": [], "expected_f1": [], "interaction_f1": []},
        )
        for metric, observed_key, expected_key, interaction_key in (
            ("exact_match", "observed_em", "expected_em", "interaction_em"),
            ("token_f1", "observed_f1", "expected_f1", "interaction_f1"),
        ):
            observed = float(row[metric]) - float(clean[metric])
            expected = (float(left_row[metric]) - float(clean[metric])) + (
                float(right_row[metric]) - float(clean[metric])
            )
            bucket[observed_key].append(observed)
            bucket[expected_key].append(expected)
            bucket[interaction_key].append(observed - expected)
    summary: dict[str, Any] = {}
    for name, bucket in collected.items():
        summary[name] = {
            "n": len(bucket["observed_f1"]),
            "observed_delta_exact_match": _mean(bucket["observed_em"]),
            "expected_delta_exact_match": _mean(bucket["expected_em"]),
            "interaction_exact_match": _mean(bucket["interaction_em"]),
            "observed_delta_token_f1": _mean(bucket["observed_f1"]),
            "expected_delta_token_f1": _mean(bucket["expected_f1"]),
            "interaction_token_f1": _mean(bucket["interaction_f1"]),
        }
    return summary
