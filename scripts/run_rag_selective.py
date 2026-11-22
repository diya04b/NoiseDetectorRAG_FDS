#!/usr/bin/env python3
"""Score unfiltered retrieval against the detector keep/remove rule.

retrieved is BM25 top-k with harmful noise still in the context.
detector_selective drops counterfactual, supportive, and orthographic
predictions, then places predicted Relevant passages first and beneficial
noise (semantic, datatype, illegal sentence) below them.
clean is the relevant passage alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.labels import LABELS  # noqa: E402
from rag_noise.detector.reports import round_numbers, write_json  # noqa: E402
from rag_noise.rag.config import load_config, validate_config  # noqa: E402
from rag_noise.rag.contexts import (  # noqa: E402
    BENEFICIAL_NOISE,
    HARMFUL_NOISE,
    RETRIEVED_CONDITION,
    SELECTIVE_CONDITION,
    controlled_conditions,
    retrieved_condition,
    selective_detector_condition,
)
from rag_noise.rag.metrics import score_answer, whitespace_tokens  # noqa: E402
from rag_noise.rag.reader import build_reader  # noqa: E402
from rag_noise.rag.reports import score_figure  # noqa: E402
from rag_noise.selection.candidates import (  # noqa: E402
    attach_saved_predictions,
    load_jsonl,
    load_split_rows,
    questions_from_rows,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score detector selection against unfiltered BM25.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "rag_selection.yaml")
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--reader", choices=["extractive", "overlap"])
    parser.add_argument("--model", type=str)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--limit", type=int, help="Answer only the first N questions that have a Relevant passage.")
    return parser.parse_args()


def _label_list(config: dict[str, Any], key: str) -> list[str]:
    labels = config.get(key)
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{key} must be a non-empty list of class names")
    unknown = [label for label in labels if label not in LABELS]
    if unknown:
        raise ValueError(f"{key} has unknown classes: {unknown}")
    return [str(label) for label in labels]


def _prepare_questions(config: dict[str, Any], predictions_path: Path) -> list[dict[str, Any]]:
    rows = load_split_rows(ROOT / config["classification_path"], config["split"])
    questions = questions_from_rows(rows, retrieve_k=10_000, k1=float(config["bm25_k1"]), b=float(config["bm25_b"]))
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"No detector predictions at {predictions_path}. Train or evaluate the detector first."
        )
    attach_saved_predictions(questions, load_jsonl(predictions_path))
    paired = [
        question
        for question in questions
        if any(passage.get("gold_label") == "Relevant" for passage in question["passages"])
    ]
    if not paired:
        raise ValueError(f"No {config['split']} questions have a Relevant passage")
    return paired


def _conditions(question: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    passages = question["passages"]
    question_text = question["question"]
    retrieved = retrieved_condition(
        question_text,
        passages,
        int(config["retrieve_k"]),
        float(config["bm25_k1"]),
        float(config["bm25_b"]),
    )
    selected = selective_detector_condition(
        question_text,
        passages,
        int(config["retrieve_k"]),
        float(config["bm25_k1"]),
        float(config["bm25_b"]),
        list(config["keep_labels"]),
        list(config["bottom_labels"]),
        int(config["keep_at_least"]),
        config["final_k"],
    )
    if selected is None:
        raise ValueError(f"{question['question_id']} is missing a detector label")
    clean = next(item for item in controlled_conditions(passages, "after") if item["condition"] == "clean")
    return [retrieved, selected, clean]


def _prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the context. "
        "If the context does not contain the answer, say that it is not in the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def _answer_question(question: dict[str, Any], config: dict[str, Any], reader: Any) -> list[dict[str, Any]]:
    from rag_noise.rag.contexts import clip_context

    rows: list[dict[str, Any]] = []
    gold_answers = [str(answer) for answer in (question.get("gold_answers") or [])]
    for condition in _conditions(question, config):
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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _paired_delta(rows: list[dict[str, Any]], condition: str, baseline: str, metric: str) -> float:
    by_key = {(str(row["question_id"]), str(row["condition"])): row for row in rows}
    deltas: list[float] = []
    for row in rows:
        if row["condition"] != condition:
            continue
        base = by_key.get((str(row["question_id"]), baseline))
        if base is None:
            continue
        deltas.append(float(row[metric]) - float(base[metric]))
    return _mean(deltas)


def _condition_block(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    chosen = [row for row in rows if row["condition"] == condition]
    harmful = list(HARMFUL_NOISE)
    beneficial = list(BENEFICIAL_NOISE)
    relevant_first = 0
    harmful_kept = 0
    beneficial_kept = 0
    for row in chosen:
        labels = [str(label) for label in row["passage_labels"]]
        if labels and labels[0] == "Relevant":
            relevant_first += 1
        harmful_kept += sum(label in harmful for label in labels)
        beneficial_kept += sum(label in beneficial for label in labels)
    n = len(chosen)
    return {
        "n": n,
        "exact_match": _mean([float(row["exact_match"]) for row in chosen]),
        "token_f1": _mean([float(row["token_f1"]) for row in chosen]),
        "mean_context_tokens": _mean([float(row["context_tokens"]) for row in chosen]),
        "mean_latency_seconds": _mean([float(row["latency_seconds"]) for row in chosen]),
        "mean_passages": _mean([float(row["n_passages"]) for row in chosen]),
        "relevant_first_rate": (relevant_first / n) if n else 0.0,
        "mean_harmful_kept": (harmful_kept / n) if n else 0.0,
        "mean_beneficial_kept": (beneficial_kept / n) if n else 0.0,
        "delta_exact_match_vs_retrieved": _paired_delta(rows, condition, RETRIEVED_CONDITION, "exact_match"),
        "delta_token_f1_vs_retrieved": _paired_delta(rows, condition, RETRIEVED_CONDITION, "token_f1"),
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _signed_percent(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def _results_markdown(summary: dict[str, Any]) -> str:
    conditions = summary["conditions"]
    lines = [
        "# Detector selection versus unfiltered retrieval",
        "",
        f"Reader: `{summary['reader']}`. Split: `{summary['split']}`. Questions: {summary['n_questions']}.",
        "",
        "`retrieved` is BM25 top-k with every stored passage, including harmful noise.",
        "`detector_selective` drops passages predicted as counterfactual, supportive, or orthographic noise,",
        "then places predicted Relevant passages first and beneficial noise (semantic, datatype, illegal sentence) below them.",
        "`clean` is the relevant passage alone.",
        "",
        "Delta is the score minus the unfiltered `retrieved` score on the same questions.",
        "A positive delta means the detector context scored higher than leaving harmful noise in.",
        "",
        "| Condition | N | Exact Match | Token F1 | Delta EM | Delta F1 | Mean passages | Harmful kept | Beneficial kept | Relevant first |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in (RETRIEVED_CONDITION, SELECTIVE_CONDITION, "clean"):
        block = conditions[name]
        lines.append(
            "| {name} | {n} | {em} | {f1} | {dem} | {df1} | {passages:.2f} | {harmful:.2f} | {beneficial:.2f} | {first} |".format(
                name=name,
                n=block["n"],
                em=_percent(float(block["exact_match"])),
                f1=_percent(float(block["token_f1"])),
                dem=_signed_percent(float(block["delta_exact_match_vs_retrieved"])),
                df1=_signed_percent(float(block["delta_token_f1_vs_retrieved"])),
                passages=float(block["mean_passages"]),
                harmful=float(block["mean_harmful_kept"]),
                beneficial=float(block["mean_beneficial_kept"]),
                first=_percent(float(block["relevant_first_rate"])),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(round_numbers(row), ensure_ascii=False) + "\n")


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
    config = validate_config(config)
    config["keep_labels"] = _label_list(config, "keep_labels")
    config["bottom_labels"] = _label_list(config, "bottom_labels")
    outside = [label for label in config["bottom_labels"] if label not in config["keep_labels"]]
    if outside:
        print(f"bottom_labels must also be in keep_labels, got {outside}")
        return 1
    if "Relevant" not in config["keep_labels"]:
        print("keep_labels must include Relevant")
        return 1
    if args.limit is not None and args.limit < 1:
        print("--limit must be a positive integer")
        return 1

    try:
        predictions_path = args.predictions or (ROOT / config["predictions_path"])
        questions = _prepare_questions(config, predictions_path)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(exc)
        return 1
    if args.limit is not None:
        questions = questions[: args.limit]

    try:
        reader = build_reader(config)
    except (ImportError, OSError, ValueError) as exc:
        print(exc)
        return 1

    print(f"Reader: {config['reader']}  questions: {len(questions)}", flush=True)
    rows: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        rows.extend(_answer_question(question, config, reader))
        if index == 1 or index % 10 == 0 or index == len(questions):
            print(f"answered {index}/{len(questions)}", flush=True)

    conditions = {
        RETRIEVED_CONDITION: _condition_block(rows, RETRIEVED_CONDITION),
        SELECTIVE_CONDITION: _condition_block(rows, SELECTIVE_CONDITION),
        "clean": _condition_block(rows, "clean"),
    }
    summary = {
        "reader": config["reader"],
        "model_name": getattr(reader, "name", config["model_name"]),
        "split": config["split"],
        "n_questions": len(questions),
        "n_answers": len(rows),
        "keep_labels": config["keep_labels"],
        "bottom_labels": config["bottom_labels"],
        "harmful_labels": list(HARMFUL_NOISE),
        "conditions": conditions,
    }
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "answers.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "run_config.json", {key: config[key] for key in config})
    results_path = output_dir / "results.md"
    results_path.write_text(_results_markdown(summary), encoding="utf-8")
    score_figure(
        {
            "retrieved": {"exact_match": conditions[RETRIEVED_CONDITION]["exact_match"], "token_f1": conditions[RETRIEVED_CONDITION]["token_f1"]},
            "detector_selective": {
                "exact_match": conditions[SELECTIVE_CONDITION]["exact_match"],
                "token_f1": conditions[SELECTIVE_CONDITION]["token_f1"],
            },
            "clean": {"exact_match": conditions["clean"]["exact_match"], "token_f1": conditions["clean"]["token_f1"]},
        },
        output_dir / "scores_by_condition.png",
    )
    selected = conditions[SELECTIVE_CONDITION]
    print(
        "detector minus retrieved  exact match: {em:+.2%}  token F1: {f1:+.2%}".format(
            em=float(selected["delta_exact_match_vs_retrieved"]),
            f1=float(selected["delta_token_f1_vs_retrieved"]),
        )
    )
    print("Wrote", results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
