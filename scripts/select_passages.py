#!/usr/bin/env python3
"""Build the final retrieved passage list from detector labels.

The noise detector is already trained. This command ranks each question's
candidate passages, reads the detector label for each one, then writes the
remove route, the rerank route, and the final list (remove, then rerank).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.detector.reports import write_json  # noqa: E402
from rag_noise.selection.candidates import (  # noqa: E402
    attach_probabilities,
    attach_saved_predictions,
    load_jsonl,
    load_split_rows,
    questions_from_retrieval_file,
    questions_from_rows,
)
from rag_noise.selection.config import load_config, validate_config  # noqa: E402
from rag_noise.selection.reports import apply_policies, write_selection  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove and rerank retrieved passages.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "selection.yaml")
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--retrieval-file", type=Path)
    parser.add_argument("--retrieve-k", type=int)
    parser.add_argument("--final-k", type=int)
    parser.add_argument("--limit", type=int, help="Score only the first N questions.")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Run the saved detector again and keep P(Relevant) for every passage.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


def _rescore(questions: list[dict[str, Any]], config: dict[str, Any], checkpoint: Path, device_name: str) -> None:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("The detector libraries are not installed. From the project folder, run:")
        print("python -m pip install matplotlib pyyaml torch transformers")
        raise SystemExit(1)
    from rag_noise.detector.data import assert_label_order
    from rag_noise.detector.engine import training_device
    from rag_noise.selection.scoring import class_probabilities

    if not checkpoint.is_dir():
        print(f"No checkpoint at {checkpoint}. Train first with python scripts/train_detector.py")
        raise SystemExit(1)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    assert_label_order(model.config.id2label)
    device = training_device(device_name)
    model.to(device)
    print(f"Device: {device}", flush=True)
    flat: list[dict[str, Any]] = []
    for question in questions:
        for passage in question["passages"]:
            flat.append({
                "question": question["question"],
                "passage_text": passage["passage_text"],
            })
    print(f"Scoring {len(flat)} passages with {checkpoint}", flush=True)
    probabilities = class_probabilities(
        model,
        tokenizer,
        flat,
        int(config["max_length"]),
        int(config["eval_batch_size"]),
        device,
    )
    attach_probabilities(questions, probabilities)


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    if args.split is not None:
        config["split"] = args.split
    if args.retrieve_k is not None:
        config["retrieve_k"] = args.retrieve_k
    if args.final_k is not None:
        config["final_k"] = args.final_k
    config = validate_config(config)
    if args.limit is not None and args.limit < 1:
        print("--limit must be a positive integer")
        return 1

    try:
        if args.retrieval_file is not None:
            questions = questions_from_retrieval_file(
                args.retrieval_file,
                int(config["retrieve_k"]),
                float(config["bm25_k1"]),
                float(config["bm25_b"]),
            )
        else:
            data_path = ROOT / config["classification_path"]
            rows = load_split_rows(data_path, config["split"])
            questions = questions_from_rows(
                rows,
                int(config["retrieve_k"]),
                float(config["bm25_k1"]),
                float(config["bm25_b"]),
            )
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    if args.limit is not None:
        questions = questions[:args.limit]

    try:
        if args.rescore:
            checkpoint = args.checkpoint or (ROOT / config["checkpoint_dir"])
            _rescore(questions, config, checkpoint, args.device)
        else:
            predictions_path = args.predictions or (ROOT / config["predictions_path"])
            if not predictions_path.is_file():
                print(
                    f"No detector predictions at {predictions_path}. "
                    "Run python scripts/evaluate_detector.py first, or pass --rescore."
                )
                return 1
            attach_saved_predictions(questions, load_jsonl(predictions_path))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(exc)
        return 1

    decided = [apply_policies(question, config) for question in questions]
    output_dir = ROOT / config["output_dir"]
    written = write_selection(output_dir, decided)
    write_json(output_dir / "run_config.json", {key: config[key] for key in config})
    summary = json.loads(Path(written["summary"]).read_text(encoding="utf-8"))
    final = summary["final"]
    print(
        f"questions: {final['n_questions']}  "
        f"mean kept: {final['mean_kept']:.2f}  "
        f"relevant kept: {final['relevant_retention']}  "
        f"noise removed: {final['noise_rejection']}  "
        f"relevant at rank 1: {final['relevant_at_rank_1_rate']}"
    )
    print("Wrote", written["final"])
    print("Wrote", written["summary"])
    print("Wrote", written["keep_remove_figure"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
