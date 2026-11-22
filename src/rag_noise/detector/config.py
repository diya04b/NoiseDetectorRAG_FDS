"""Load and check the detector training configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def default_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "model_name": "roberta-base",
        "max_length": 512,
        "train_batch_size": 4,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "epochs": 3,
        "learning_rate": 2.0e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 2,
        "limit_per_class": None,
        "classification_path": "data/processed/classification.jsonl",
        "output_dir": "outputs/detector",
        "checkpoint_dir": "checkpoints/detector",
    }


def _require_int(config: dict[str, Any], key: str, minimum: int) -> None:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")


def _require_unit_interval(config: dict[str, Any], key: str) -> None:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    if not 0 <= float(value) < 1:
        raise ValueError(f"{key} must be at least 0 and less than 1")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config.get("model_name"), str) or not config["model_name"].strip():
        raise ValueError("model_name must be a non-empty string")
    _require_int(config, "seed", 0)
    _require_int(config, "max_length", 32)
    _require_int(config, "train_batch_size", 1)
    _require_int(config, "eval_batch_size", 1)
    _require_int(config, "gradient_accumulation_steps", 1)
    _require_int(config, "epochs", 1)
    learning_rate = config["learning_rate"]
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("learning_rate must be a positive number")
    weight_decay = config["weight_decay"]
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise ValueError("weight_decay must be a number >= 0")
    _require_unit_interval(config, "warmup_ratio")
    max_grad_norm = config["max_grad_norm"]
    if isinstance(max_grad_norm, bool) or not isinstance(max_grad_norm, (int, float)) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be a positive number")
    patience = config["early_stopping_patience"]
    if patience is not None and (isinstance(patience, bool) or not isinstance(patience, int) or patience < 1):
        raise ValueError("early_stopping_patience must be a positive integer or null")
    limit = config["limit_per_class"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit_per_class must be a positive integer or null")
    for key in ("classification_path", "output_dir", "checkpoint_dir"):
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
