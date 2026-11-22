"""Write RAG answers, the score table, and the comparison chart."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rag_noise.data.labels import LABELS
from rag_noise.detector.plots import SHORT_LABELS
from rag_noise.detector.reports import round_numbers, write_json
from rag_noise.rag.contexts import DETECTOR_CONDITION, PAIR_JOIN
from rag_noise.rag.metrics import summarize_conditions, summarize_interactions


def condition_order(names: list[str]) -> list[str]:
    preferred = ["clean", *[label for label in LABELS if label != "Relevant"]]
    ordered = [name for name in preferred if name in names]
    pairs = [name for name in names if PAIR_JOIN in name]
    ordered.extend(pairs)
    if DETECTOR_CONDITION in names:
        ordered.append(DETECTOR_CONDITION)
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(round_numbers(row), ensure_ascii=False) + "\n")


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _signed_percent(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def results_markdown(summary: dict[str, Any], conditions: dict[str, Any]) -> str:
    lines = [
        "# RAG answer scores",
        "",
        f"Reader: `{summary['reader']}`. Split: `{summary['split']}`. "
        f"Questions with a clean passage: {summary['n_questions_with_clean']}.",
        "",
        "Delta is the noisy score minus the clean score on the same questions. "
        "A negative delta means the noise lowered the score.",
        "",
        "| Condition | N | Exact Match | Token F1 | Delta EM | Delta F1 | Mean context tokens | Mean seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in condition_order(list(conditions)):
        block = conditions[name]
        if name == "clean":
            delta_em = ""
            delta_f1 = ""
        else:
            delta_em = _signed_percent(float(block["delta_exact_match"]))
            delta_f1 = _signed_percent(float(block["delta_token_f1"]))
        lines.append(
            "| {name} | {n} | {em} | {f1} | {dem} | {df1} | {tokens:.1f} | {seconds:.3f} |".format(
                name=name,
                n=block["n"],
                em=_percent(float(block["exact_match"])),
                f1=_percent(float(block["token_f1"])),
                dem=delta_em,
                df1=delta_f1,
                tokens=float(block["mean_context_tokens"]),
                seconds=float(block["mean_latency_seconds"]),
            )
        )
    lines.append("")
    interactions = summary.get("interactions") or {}
    if interactions:
        lines.extend([
            "Pairwise interaction is the combined delta minus the sum of the two individual deltas. "
            "Near zero means the noises add. Negative means the pair hurt more than that sum.",
            "",
            "| Pair | N | Observed delta F1 | Expected delta F1 | Interaction F1 | Interaction EM |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for name in condition_order(list(interactions)):
            block = interactions[name]
            lines.append(
                "| {name} | {n} | {obs} | {exp} | {inter} | {em} |".format(
                    name=name,
                    n=block["n"],
                    obs=_signed_percent(float(block["observed_delta_token_f1"])),
                    exp=_signed_percent(float(block["expected_delta_token_f1"])),
                    inter=_signed_percent(float(block["interaction_token_f1"])),
                    em=_signed_percent(float(block["interaction_exact_match"])),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _short_condition(name: str) -> str:
    if name == DETECTOR_CONDITION:
        return "Detector"
    if PAIR_JOIN in name:
        return PAIR_JOIN.join(SHORT_LABELS.get(part, part) for part in name.split(PAIR_JOIN))
    return SHORT_LABELS.get(name, name)


def score_figure(conditions: dict[str, Any], path: Path) -> None:
    names = condition_order(list(conditions))
    if not names:
        raise ValueError("No conditions to plot")
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [_short_condition(name) for name in names]
    exact = [100.0 * float(conditions[name]["exact_match"]) for name in names]
    f1 = [100.0 * float(conditions[name]["token_f1"]) for name in names]
    positions = list(range(len(names)))
    figure, axis = plt.subplots(figsize=(11, 5))
    width = 0.38
    axis.bar([pos - width / 2 for pos in positions], exact, width=width, color="#3C6E71", label="Exact Match")
    axis.bar([pos + width / 2 for pos in positions], f1, width=width, color="#D9A05B", label="Token F1")
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 100)
    axis.set_title("RAG answer scores by context")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def interaction_figure(interactions: dict[str, Any], path: Path) -> None:
    names = condition_order(list(interactions))
    if not names:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [_short_condition(name) for name in names]
    values = [100.0 * float(interactions[name]["interaction_token_f1"]) for name in names]
    figure, axis = plt.subplots(figsize=(10, max(4.0, 0.38 * len(names) + 1.2)))
    positions = list(range(len(names)))
    colors = ["#A33B3B" if value < 0 else "#3C6E71" for value in values]
    axis.barh(positions, values, color=colors)
    axis.set_yticks(positions, labels)
    axis.axvline(0, color="#222222", linewidth=0.8)
    axis.set_xlabel("Interaction on token F1 (percentage points)")
    axis.set_title("Pairwise noise interaction versus the additive expectation")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_rag(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = summarize_conditions(rows)
    interactions = summarize_interactions(rows)
    summary = {**summary, "conditions": conditions, "interactions": interactions}
    answers_path = output_dir / "answers.jsonl"
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "results.md"
    figure_path = output_dir / "scores_by_condition.png"
    interaction_path = output_dir / "pairwise_interaction.png"
    _write_jsonl(answers_path, rows)
    write_json(summary_path, summary)
    markdown_path.write_text(results_markdown(summary, conditions), encoding="utf-8")
    single = {name: block for name, block in conditions.items() if PAIR_JOIN not in name}
    if single:
        score_figure(single, figure_path)
    elif conditions:
        score_figure(conditions, figure_path)
    if interactions:
        interaction_figure(interactions, interaction_path)
    return {
        "answers": str(answers_path),
        "summary": str(summary_path),
        "results": str(markdown_path),
        "figure": str(figure_path),
        "interactions": str(interaction_path) if interactions else "",
    }
