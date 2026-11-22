#!/usr/bin/env python3
"""Answer the test questions from clean, noisy, and detector-filtered contexts.

This is the RAG Generator box in Figure 1. It does not train the noise
detector. The clean condition uses only the Relevant passage. Each noise
condition adds one noise class. detector_final uses the passages kept by
remove-then-rerank. Answers are scored with Exact Match and token F1.
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
from rag_noise.rag.config import load_config, validate_config  # noqa: E402
from rag_noise.rag.evaluate import answer_question  # noqa: E402
from rag_noise.rag.reader import build_reader  # noqa: E402
from rag_noise.rag.reports import write_rag  # noqa: E402
from rag_noise.selection.candidates import (  # noqa: E402
    attach_saved_predictions,
    load_jsonl,
    load_split_rows,
    questions_from_rows,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vanilla RAG answerer and score it.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "rag.yaml")
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--reader", choices=["extractive", "overlap"])
    parser.add_argument("--model", type=str)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--limit", type=int, help="Answer only the first N questions that have a Relevant passage.")
    parser.add_argument("--no-detector", action="store_true", help="Skip the detector_final condition.")
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Also score every pair of noise classes and the interaction against the additive expectation.",
    )
    return parser.parse_args()


def _prepare_questions(config: dict[str, Any], predictions_path: Path) -> list[dict[str, Any]]:
    rows = load_split_rows(ROOT / config["classification_path"], config["split"])
    # Keep every stored passage. The noise conditions need each class, not a truncated top-k.
    questions = questions_from_rows(rows, retrieve_k=10_000, k1=float(config["bm25_k1"]), b=float(config["bm25_b"]))
    if not config["use_detector"]:
        return questions
    if not predictions_path.is_file():
        print(
            f"No detector predictions at {predictions_path}. "
            "The clean and noise conditions will still run. "
            "detector_final is skipped. Pass --no-detector to hide this note."
        )
        config["use_detector"] = False
        return questions
    attach_saved_predictions(questions, load_jsonl(predictions_path))
    return questions


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    if args.split is not None:
        config["split"] = args.split
    if args.reader is not None:
        config["reader"] = args.reader
    if args.model is not None:
        config["model_name"] = args.model
    if args.device is not None:
        config["device"] = args.device
    if args.no_detector:
        config["use_detector"] = False
    if args.pairwise:
        config["pairwise"] = True
    config = validate_config(config)
    if args.limit is not None and args.limit < 1:
        print("--limit must be a positive integer")
        return 1

    try:
        predictions_path = args.predictions or (ROOT / config["predictions_path"])
        questions = _prepare_questions(config, predictions_path)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(exc)
        return 1

    # PriorQA questions have no Relevant passage, so they have no paired clean baseline.
    skipped_prior = 0
    paired: list[dict[str, Any]] = []
    for question in questions:
        if any(passage.get("gold_label") == "Relevant" for passage in question["passages"]):
            paired.append(question)
        else:
            skipped_prior += 1
    questions = paired
    if args.limit is not None:
        questions = questions[:args.limit]

    try:
        reader = build_reader(config)
    except (ImportError, OSError, ValueError) as exc:
        print(exc)
        return 1

    print(
        f"Reader: {config['reader']}  questions: {len(questions)}  "
        f"detector_final: {config['use_detector']}  pairwise: {config['pairwise']}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    clean_seen = 0
    for index, question in enumerate(questions, start=1):
        rows.extend(answer_question(question, config, reader))
        if any(passage.get("gold_label") == "Relevant" for passage in question["passages"]):
            clean_seen += 1
        if index == 1 or index % 10 == 0 or index == len(questions):
            print(f"answered {index}/{len(questions)}", flush=True)

    summary = {
        "reader": config["reader"],
        "model_name": getattr(reader, "name", config["model_name"]),
        "split": config["split"],
        "noise_position": config["noise_position"],
        "n_questions": len(questions),
        "n_questions_with_clean": clean_seen,
        "n_questions_without_clean_skipped": skipped_prior,
        "n_answers": len(rows),
    }
    output_dir = ROOT / config["output_dir"]
    written = write_rag(output_dir, rows, summary)
    write_json(output_dir / "run_config.json", {key: config[key] for key in config})
    saved = json.loads(Path(written["summary"]).read_text(encoding="utf-8"))
    clean_scores = saved["conditions"].get("clean", {})
    print(
        "clean exact match: {em:.2%}  clean token F1: {f1:.2%}".format(
            em=float(clean_scores.get("exact_match", 0.0)),
            f1=float(clean_scores.get("token_f1", 0.0)),
        )
    )
    print("Wrote", written["results"])
    print("Wrote", written["answers"])
    print("Wrote", written["figure"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
