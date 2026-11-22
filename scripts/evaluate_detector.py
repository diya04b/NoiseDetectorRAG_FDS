#!/usr/bin/env python3
"""Score a saved noise detector on one split.

Training already scores the test split. Use this script to score the saved
checkpoint again, or to score validation without training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.detector.config import load_config, validate_config  # noqa: E402
from rag_noise.detector.data import (  # noqa: E402
    assert_label_order,
    limit_examples,
    load_examples,
    require_all_labels,
    split_examples,
)
from rag_noise.detector.reports import write_split_report  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved noise detector.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "detector.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


def _load_run_config(checkpoint: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    path = checkpoint / "run_config.json"
    if not path.is_file():
        return dict(fallback)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise ValueError(f"{path} must contain a JSON object")
    merged = dict(fallback)
    # Keep the token length and model name from training. Do not reuse a
    # practice-run row limit, or a later full test would silently score a subset.
    for key in ("model_name", "max_length"):
        if key in saved:
            merged[key] = saved[key]
    return merged


def main() -> int:
    args = _parse_args()
    fallback = load_config(args.config)
    checkpoint = args.checkpoint or (ROOT / fallback["checkpoint_dir"])
    if not checkpoint.is_dir():
        print(f"No checkpoint at {checkpoint}. Train first with python scripts/train_detector.py")
        return 1
    config = validate_config(_load_run_config(checkpoint, fallback))
    if args.batch_size is not None:
        config["eval_batch_size"] = args.batch_size
    if args.limit_per_class is not None:
        config["limit_per_class"] = args.limit_per_class
    config = validate_config(config)

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("The detector libraries are not installed. From the project folder, run:")
        print("python -m pip install matplotlib pyyaml torch transformers")
        return 1

    from rag_noise.detector.engine import evaluate_examples, training_device

    data_path = ROOT / config["classification_path"]
    try:
        examples = limit_examples(load_examples(data_path), config["limit_per_class"])
    except FileNotFoundError as exc:
        print(exc)
        return 1
    grouped = split_examples(examples)
    require_all_labels(grouped[args.split], args.split)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    assert_label_order(model.config.id2label)
    device = training_device(args.device)
    model.to(device)
    print(f"Device: {device}", flush=True)
    if device.type == "cpu":
        print("Running on CPU. No GPU is required for this command.", flush=True)
    print(f"Checkpoint: {checkpoint}")
    print(f"Split: {args.split} ({len(grouped[args.split])} rows)")

    evaluation = evaluate_examples(
        model,
        tokenizer,
        grouped[args.split],
        config,
        device,
        log=lambda message: print(message, flush=True),
        progress_label=args.split,
    )
    by_id = {row["example_id"]: row for row in examples}
    written = write_split_report(ROOT / config["output_dir"], args.split, evaluation, by_id)
    metrics = json.loads(Path(written["metrics"]).read_text(encoding="utf-8"))
    print(
        f"{args.split} accuracy: {metrics['accuracy']:.4f}  "
        f"macro-F1: {metrics['macro_f1']:.4f}  "
        f"weighted-F1: {metrics['weighted_f1']:.4f}"
    )
    print("Wrote", written["metrics"])
    print("Wrote", written["confusion_matrix_figure"])
    print("Wrote", written["error_analysis"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
