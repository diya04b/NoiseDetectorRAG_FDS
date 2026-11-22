"""Rank a question's candidate passages before the detector filters them.

The ranker is Okapi BM25 over the passages already stored for that question.
It supplies the Top-k Retrieved Passages box in Figure 1. It does not search
the web, and it does not change the passage text.
"""

from __future__ import annotations

import math
import re
from typing import Any


_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25. A query term missing from every passage adds nothing."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or b < 0:
            raise ValueError("bm25_k1 and bm25_b must be >= 0")
        self.k1 = float(k1)
        self.b = float(b)
        self.documents = documents
        self.n_docs = len(documents)
        self.doc_lengths = [len(tokens) for tokens in documents]
        self.avg_length = (sum(self.doc_lengths) / self.n_docs) if self.n_docs else 0.0
        self.term_freqs: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for tokens in documents:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.term_freqs.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: math.log(1.0 + (self.n_docs - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def score(self, query_tokens: list[str], index: int) -> float:
        if self.avg_length == 0:
            return 0.0
        counts = self.term_freqs[index]
        length_norm = 1.0 - self.b + self.b * (self.doc_lengths[index] / self.avg_length)
        total = 0.0
        seen: set[str] = set()
        for token in query_tokens:
            if token in seen:
                continue
            seen.add(token)
            frequency = counts.get(token, 0)
            if frequency == 0:
                continue
            numerator = frequency * (self.k1 + 1.0)
            denominator = frequency + self.k1 * length_norm
            total += self.idf.get(token, 0.0) * numerator / denominator
        return total


def rank_passages(
    question: str,
    passages: list[dict[str, Any]],
    retrieve_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Return up to retrieve_k passages, best BM25 score first.

    Equal scores keep the order of `passages`. retrieval_rank starts at 1.
    """
    if retrieve_k < 1:
        raise ValueError("retrieve_k must be a positive integer")
    if not passages:
        raise ValueError("Cannot retrieve from an empty passage list")
    tokenized = [tokenize(str(passage.get("passage_text", ""))) for passage in passages]
    ranker = BM25(tokenized, k1=k1, b=b)
    query = tokenize(question)
    scored = [(ranker.score(query, index), index, passages[index]) for index in range(len(passages))]
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = scored[:retrieve_k]
    ranked: list[dict[str, Any]] = []
    for rank, (score, _index, passage) in enumerate(chosen, start=1):
        row = dict(passage)
        row["retrieval_score"] = float(score)
        row["retrieval_rank"] = rank
        ranked.append(row)
    return ranked
