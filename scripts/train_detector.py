#!/usr/bin/env python3
"""Fine-tune the noise detector and then score the untouched test split.

The epoch is chosen by validation macro-F1. The test split is scored once,
after that choice. Run this from the project folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.labels import LABELS, LABEL_TO_ID  # noqa: E402
from rag_noise.detector.config import load_config  # noqa: E402
from rag_noise.detector.data import (  # noqa: E402
    limit_examples,
    load_examples,
    require_all_labels,
    split_examples,
)
from rag_noise.detector.reports import write_json, write_split_report  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the eight-class noise detector.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "detector.yaml")
    parser.add_argument("--model", help="Override model_name, for example roberta-base")
    parser.add_argument("--epochs", type=int, help="Override the number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override the training batch size")
    parser.add_argument("--max-length", type=int, help="Override the maximum token length")
    parser.add_argument("--limit-per-class", type=int, help="Keep only this many rows per class in each split")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.model:
        config["model_name"] = args.model
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train_batch_size"] = args.batch_size
    if args.max_length is not None:
        config["max_length"] = args.max_length
    if args.limit_per_class is not None:
        config["limit_per_class"] = args.limit_per_class
    return config


def _import_training_libraries() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("The detector libraries are not installed. From the project folder, run:")
        print("python -m pip install matplotlib pyyaml torch transformers")
        raise SystemExit(1) from None
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def main() -> int:
    args = _parse_args()
    config = _apply_overrides(load_config(args.config), args)
    from rag_noise.detector.config import validate_config

    config = validate_config(config)
    data_path = ROOT / config["classification_path"]
    try:
        examples = limit_examples(load_examples(data_path), config["limit_per_class"])
    except FileNotFoundError as exc:
        print(exc)
        return 1
    grouped = split_examples(examples)
    for split in ("train", "validation", "test"):
        require_all_labels(grouped[split], split)

    _torch, model_class, tokenizer_class = _import_training_libraries()
    from rag_noise.detector.engine import evaluate_examples, train_model, training_device

    device = training_device(args.device)
    print(f"Device: {device}", flush=True)
    if device.type == "cpu":
        print("Running on CPU. No GPU is required for this command.", flush=True)
    print(f"Model: {config['model_name']}")
    print(
        "Rows:",
        {split: len(grouped[split]) for split in ("train", "validation", "test")},
    )
    tokenizer = tokenizer_class.from_pretrained(config["model_name"])
    model = model_class.from_pretrained(
        config["model_name"],
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=dict(LABEL_TO_ID),
    )
    print(
        "Model loaded. A new 8-class layer was added on top of the pretrained weights. "
        "Messages about missing classifier weights are that new layer, not a failed download.",
        flush=True,
    )

    output_dir = ROOT / config["output_dir"]
    checkpoint_dir = ROOT / config["checkpoint_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / "detector_history.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_config = {
        **config,
        "label_order": list(LABELS),
        "device": str(device),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_metric": "validation_macro_f1",
    }
    write_json(output_dir / "run_config.json", run_config)

    def on_epoch(record: dict[str, Any]) -> None:
        line = (
            f"epoch {record['epoch']}: train_loss={record['train_loss']:.4f} "
            f"val_loss={record['validation_loss']:.4f} "
            f"val_accuracy={record['validation_accuracy']:.4f} "
            f"val_macro_f1={record['validation_macro_f1']:.4f}"
        )
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def on_improvement(epoch: int, _scores: dict[str, Any]) -> None:
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        saved = {**run_config, "best_epoch": epoch}
        write_json(checkpoint_dir / "run_config.json", saved)
        print(f"Saved checkpoint for epoch {epoch}", flush=True)

    result = train_model(
        model,
        tokenizer,
        grouped["train"],
        grouped["validation"],
        config,
        device,
        on_epoch=on_epoch,
        on_improvement=on_improvement,
        log=lambda message: print(message, flush=True),
    )
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    finished = {
        **run_config,
        "best_epoch": result["best_epoch"],
        "best_validation_macro_f1": result["best_validation_macro_f1"],
        "optimizer_steps": result["optimizer_steps"],
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(output_dir / "run_config.json", finished)
    write_json(checkpoint_dir / "run_config.json", finished)
    write_json(output_dir / "training_history.json", result["history"])

    by_id = {row["example_id"]: row for row in examples}
    validation_report = write_split_report(output_dir, "validation", result["best_validation"], by_id)
    test_evaluation = evaluate_examples(
        model,
        tokenizer,
        grouped["test"],
        config,
        device,
        log=lambda message: print(message, flush=True),
        progress_label="test",
    )
    test_report = write_split_report(output_dir, "test", test_evaluation, by_id)

    test_metrics = json.loads(Path(test_report["metrics"]).read_text(encoding="utf-8"))
    print(f"Best epoch: {result['best_epoch']}")
    print(
        "Test accuracy: {accuracy:.4f}  Test macro-F1: {macro_f1:.4f}  Test weighted-F1: {weighted_f1:.4f}".format(
            **test_metrics
        )
    )
    print("Wrote", validation_report["metrics"])
    print("Wrote", test_report["metrics"])
    print("Wrote", test_report["confusion_matrix_figure"])
    print("Wrote", test_report["error_analysis"])
    print("Wrote checkpoint to", checkpoint_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
