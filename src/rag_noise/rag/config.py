"""Load and check the RAG experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rag_noise.data.labels import LABELS


def default_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "split": "test",
        "classification_path": "data/processed/classification.jsonl",
        "predictions_path": "outputs/detector/test_predictions.jsonl",
        "output_dir": "outputs/rag",
        "reader": "extractive",
        "model_name": "deepset/roberta-base-squad2",
        "device": "cpu",
        "noise_position": "after",
        "pairwise": False,
        "max_context_chars": 6000,
        "use_detector": True,
        "retrieve_k": 7,
        "keep_labels": ["Relevant"],
        "keep_at_least": 1,
        "final_k": None,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    }


def _require_int(config: dict[str, Any], key: str, minimum: int) -> None:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    _require_int(config, "seed", 0)
    _require_int(config, "retrieve_k", 1)
    _require_int(config, "keep_at_least", 0)
    _require_int(config, "max_context_chars", 200)
    if config["split"] not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if config["reader"] not in {"extractive", "overlap"}:
        raise ValueError("reader must be extractive or overlap")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config["noise_position"] not in {"after", "before"}:
        raise ValueError("noise_position must be after or before")
    if not isinstance(config["use_detector"], bool):
        raise ValueError("use_detector must be true or false")
    if not isinstance(config["pairwise"], bool):
        raise ValueError("pairwise must be true or false")
    final_k = config["final_k"]
    if final_k is not None and (isinstance(final_k, bool) or not isinstance(final_k, int) or final_k < 1):
        raise ValueError("final_k must be a positive integer or null")
    for key in ("bm25_k1", "bm25_b"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
            raise ValueError(f"{key} must be a number >= 0")
    keep_labels = config["keep_labels"]
    if not isinstance(keep_labels, list) or not keep_labels:
        raise ValueError("keep_labels must be a non-empty list of class names")
    unknown = [label for label in keep_labels if label not in LABELS]
    if unknown:
        raise ValueError(f"keep_labels has unknown classes: {unknown}")
    if not isinstance(config.get("model_name"), str) or not config["model_name"].strip():
        raise ValueError("model_name must be a non-empty string")
    for key in ("classification_path", "predictions_path", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty path")
    return config


def load_config(path: Path | None) -> dict[str, Any]:
    config = default_config()
    if path is not None:
        with path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config at {path} must be a mapping")
        config.update(loaded)
    return validate_config(config)
