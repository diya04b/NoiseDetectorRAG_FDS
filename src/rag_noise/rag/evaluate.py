"""Run every context condition through one reader and score the answers."""

from __future__ import annotations

import time
from typing import Any

from rag_noise.rag.contexts import clip_context, question_conditions
from rag_noise.rag.metrics import score_answer, whitespace_tokens
from rag_noise.rag.reader import AnswerReader


def _prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the context. "
        "If the context does not contain the answer, say that it is not in the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def answer_question(
    question: dict[str, Any],
    config: dict[str, Any],
    reader: AnswerReader,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gold_answers = [str(answer) for answer in (question.get("gold_answers") or [])]
    for condition in question_conditions(question, config):
        context, truncated = clip_context(condition["context"], int(config["max_context_chars"]))
        started = time.perf_counter()
        answer = reader.answer(question["question"], context)
        latency = time.perf_counter() - started
        scores = score_answer(answer, gold_answers)
        rows.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "split": question.get("split"),
            "source_dataset": question.get("source_dataset"),
            "condition": condition["condition"],
            "role": condition["role"],
            "gold_answers": gold_answers,
            "answer": answer,
            "exact_match": scores["exact_match"],
            "token_f1": scores["token_f1"],
            "context": context,
            "context_truncated": truncated,
            "context_tokens": whitespace_tokens(context),
            "answer_tokens": whitespace_tokens(answer),
            "latency_seconds": latency,
            "n_passages": len(condition["passages"]),
            "passage_ids": [passage["passage_id"] for passage in condition["passages"]],
            "passage_labels": [passage.get("gold_label") for passage in condition["passages"]],
            "prompt": _prompt(question["question"], context),
        })
    return rows
