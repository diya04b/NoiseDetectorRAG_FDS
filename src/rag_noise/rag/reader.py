"""Turn a question and a context into an answer.

extractive is a vanilla reading-comprehension RAG reader. It returns a span
copied from the context, which is what Exact Match and token F1 expect for
these short gold answers.

overlap needs no model download. It returns the context sentence with the
most question words. Use it to check the files. The report numbers come from
the extractive reader.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from rag_noise.rag.metrics import normalize_answer


_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


class AnswerReader(Protocol):
    def answer(self, question: str, context: str) -> str:
        """Return the answer string. An empty context returns an empty answer."""


class OverlapReader:
    """Sentence picker. No weights and no network."""

    name = "overlap"

    def answer(self, question: str, context: str) -> str:
        if not context.strip():
            return ""
        sentences = [part.strip() for part in _SENTENCE.findall(context) if part.strip()]
        if not sentences:
            return context.strip()
        question_tokens = set(normalize_answer(question).split())
        if not question_tokens:
            return sentences[0]

        def score(sentence: str) -> tuple[int, int]:
            tokens = normalize_answer(sentence).split()
            overlap = sum(1 for token in tokens if token in question_tokens)
            return (overlap, -len(tokens))

        return max(sentences, key=score)


class ExtractiveReader:
    """Span reader. The answer is copied from the context.

    Transformers 5 no longer registers a question-answering pipeline, so this
    loads the span model directly and slides a window over long contexts.
    The highest-scoring span is returned. The SQuAD v2 null score is not used
    to blank the answer, because an empty answer on both the clean and noisy
    contexts would hide the comparison in section 3.8.
    """

    def __init__(self, model_name: str, device_name: str) -> None:
        if device_name not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        try:
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The extractive reader needs transformers and torch. "
                "Install them with the commands in docs/RAG_GUIDE.md, "
                "and run the script with Python 3.12."
            ) from exc
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("device is cuda, but this machine has no CUDA GPU")
        self.name = model_name
        self._torch = torch
        self.device = torch.device(device_name)
        self.max_length = 384
        self.stride = 128
        self.max_answer_length = 64
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def answer(self, question: str, context: str) -> str:
        if not context.strip() or not question.strip():
            return ""
        encoding = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        window_count = len(encoding["input_ids"])
        if window_count == 0:
            return ""
        torch = self._torch
        input_ids = torch.tensor(encoding["input_ids"], device=self.device)
        attention_mask = torch.tensor(encoding["attention_mask"], device=self.device)
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits.detach().cpu()
        end_logits = outputs.end_logits.detach().cpu()

        best_span: tuple[float, str] | None = None
        for index in range(window_count):
            offsets = encoding["offset_mapping"][index]
            sequence_ids = encoding.sequence_ids(index)
            start = start_logits[index]
            end = end_logits[index]
            context_indexes = [token for token, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
            if not context_indexes:
                continue
            start_candidates = sorted(
                ((float(start[token]), token) for token in context_indexes),
                reverse=True,
            )[:20]
            end_candidates = sorted(
                ((float(end[token]), token) for token in context_indexes),
                reverse=True,
            )[:20]
            for start_score, start_token in start_candidates:
                for end_score, end_token in end_candidates:
                    if end_token < start_token or end_token - start_token + 1 > self.max_answer_length:
                        continue
                    start_char, _ = offsets[start_token]
                    _, end_char = offsets[end_token]
                    if start_char is None or end_char is None or end_char <= start_char:
                        continue
                    text = context[start_char:end_char].strip()
                    if not text:
                        continue
                    score = start_score + end_score
                    if best_span is None or score > best_span[0]:
                        best_span = (score, text)
        if best_span is None:
            return ""
        return best_span[1]


def build_reader(config: dict[str, Any]) -> AnswerReader:
    if config["reader"] == "overlap":
        return OverlapReader()
    if config["reader"] == "extractive":
        return ExtractiveReader(str(config["model_name"]), str(config["device"]))
    raise ValueError(f"reader must be extractive or overlap, got {config['reader']!r}")
