"""Score retrieved passages with the saved detector.

The training evaluator needs a gold label on every row. Retrieved passages do
not, so this scorer only returns the eight class probabilities.
"""

from __future__ import annotations

from typing import Any

from rag_noise.data.labels import LABELS
from rag_noise.detector.data import encode_pair


def class_probabilities(
    model: Any,
    tokenizer: Any,
    passages: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    device: Any,
) -> list[dict[str, float]]:
    """Return one probability map per passage, in the same order."""
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not passages:
        raise ValueError("Cannot score an empty passage list")
    import torch

    model.eval()
    distributions: list[dict[str, float]] = []
    with torch.no_grad():
        for start in range(0, len(passages), batch_size):
            chunk = passages[start:start + batch_size]
            encoded = [
                encode_pair(
                    tokenizer,
                    str(passage["question"]),
                    str(passage["passage_text"]),
                    max_length,
                )
                for passage in chunk
            ]
            batch = tokenizer.pad(encoded, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            logits = output.logits if hasattr(output, "logits") else output
            if logits.shape[-1] != len(LABELS):
                raise ValueError(
                    f"Detector returned {logits.shape[-1]} scores, expected {len(LABELS)}"
                )
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            for row in probs:
                distributions.append({label: float(row[index]) for index, label in enumerate(LABELS)})
    if len(distributions) != len(passages):
        raise RuntimeError(f"Scored {len(distributions)} passages, expected {len(passages)}")
    return distributions
