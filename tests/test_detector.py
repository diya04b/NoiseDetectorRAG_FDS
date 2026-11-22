"""Tests for detector scoring, data checks, and the training step."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.labels import LABEL_TO_ID, LABELS  # noqa: E402
from rag_noise.detector.config import load_config, validate_config  # noqa: E402
from rag_noise.detector.data import (  # noqa: E402
    assert_label_order,
    encode_pair,
    limit_examples,
    load_examples,
    require_all_labels,
    split_examples,
)
from rag_noise.detector.metrics import (  # noqa: E402
    NAMED_CONFUSIONS,
    build_error_analysis,
    confusion_counts,
    score_predictions,
)
from rag_noise.detector.plots import confusion_matrix_figure, per_class_f1_figure  # noqa: E402
from rag_noise.detector.reports import prediction_rows, write_split_report  # noqa: E402
from rag_noise.detector.schedule import learning_rate_multiplier, optimizer_step_count  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
REAL_DATA = ROOT / "data" / "processed" / "classification.jsonl"


def _row(example_id: str, label: str, split: str, passage: str = "A passage about the prize.") -> dict:
    return {
        "example_id": example_id,
        "question": "Who won the prize?",
        "passage_text": passage,
        "label": label,
        "label_id": LABEL_TO_ID[label],
        "split": split,
        "parent_id": "hotpotqa:1",
        "source_dataset": "hotpotqa",
    }


class MetricTests(unittest.TestCase):
    def test_confusion_rows_are_true_and_columns_are_predicted(self) -> None:
        matrix = confusion_counts([1, 1, 5], [5, 1, 1])
        self.assertEqual(len(matrix), 8)
        self.assertEqual(matrix[1][5], 1)
        self.assertEqual(matrix[1][1], 1)
        self.assertEqual(matrix[5][1], 1)
        self.assertEqual(sum(sum(row) for row in matrix), 3)

    def test_scores_match_a_hand_count(self) -> None:
        scores = score_predictions([1, 1, 5], [5, 1, 1])
        self.assertEqual(scores["n_examples"], 3)
        self.assertEqual(scores["n_correct"], 1)
        self.assertAlmostEqual(scores["accuracy"], 1 / 3)
        semantic = scores["per_class"][1]
        self.assertEqual(semantic["label"], "Semantic Noise")
        self.assertAlmostEqual(semantic["precision"], 0.5)
        self.assertAlmostEqual(semantic["recall"], 0.5)
        self.assertAlmostEqual(semantic["f1"], 0.5)
        self.assertAlmostEqual(scores["macro_f1"], 0.5 / 8)
        self.assertAlmostEqual(scores["weighted_f1"], 1 / 3)

    def test_named_confusions_follow_the_report(self) -> None:
        self.assertEqual(
            NAMED_CONFUSIONS,
            (
                ("Semantic Noise", "Supportive Noise"),
                ("Supportive Noise", "Semantic Noise"),
                ("Counterfactual Noise", "Prior Noise"),
                ("Prior Noise", "Counterfactual Noise"),
            ),
        )
        analysis = build_error_analysis([1, 1, 5], [5, 1, 1])
        by_pair = {
            (item["true_label"], item["predicted_label"]): item
            for item in analysis["named_confusions"]
        }
        semantic = by_pair[("Semantic Noise", "Supportive Noise")]
        self.assertEqual(semantic["count"], 1)
        self.assertEqual(semantic["support"], 2)
        self.assertAlmostEqual(semantic["rate"], 0.5)
        prior = by_pair[("Counterfactual Noise", "Prior Noise")]
        self.assertEqual(prior["count"], 0)
        self.assertEqual(analysis["top_confusions"][0]["count"], 1)

    def test_empty_predictions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            score_predictions([], [])


class DataTests(unittest.TestCase):
    def test_label_id_must_match_the_fixed_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            row = _row("a", "Relevant", "train")
            row["label_id"] = 3
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_examples(path)

    def test_limit_keeps_the_first_row_of_each_class(self) -> None:
        rows = [
            _row("first", "Relevant", "train", "first passage"),
            _row("second", "Relevant", "train", "second passage"),
            _row("other-split", "Relevant", "test", "test passage"),
        ]
        kept = limit_examples(rows, 1)
        self.assertEqual([row["example_id"] for row in kept], ["first", "other-split"])

    def test_missing_class_is_reported(self) -> None:
        rows = [_row("a", "Relevant", "train")]
        with self.assertRaises(ValueError):
            require_all_labels(rows, "train")

    def test_tokenizer_receives_the_passage_unchanged_and_truncates_it_first(self) -> None:
        seen = {}

        def tokenizer(question, passage, truncation, max_length, add_special_tokens):
            seen["question"] = question
            seen["passage"] = passage
            seen["truncation"] = truncation
            seen["max_length"] = max_length
            seen["add_special_tokens"] = add_special_tokens
            return {"input_ids": [0, 1, 2], "attention_mask": [1, 1, 1]}

        passage = "Ada wn the pize."
        features = encode_pair(tokenizer, "Who won?", passage, 32)
        self.assertEqual(seen["question"], "Who won?")
        self.assertEqual(seen["passage"], passage)
        self.assertEqual(seen["truncation"], "only_second")
        self.assertEqual(seen["max_length"], 32)
        self.assertTrue(seen["add_special_tokens"])
        self.assertEqual(features["input_ids"], [0, 1, 2])

    def test_checkpoint_labels_must_stay_in_project_order(self) -> None:
        assert_label_order({index: label for index, label in enumerate(LABELS)})
        assert_label_order({str(index): label for index, label in enumerate(LABELS)})
        swapped = {index: label for index, label in enumerate(LABELS)}
        swapped[0] = "Prior Noise"
        with self.assertRaises(ValueError):
            assert_label_order(swapped)

    def test_saved_detector_config_uses_roberta_and_the_model_maximum(self) -> None:
        config = load_config(ROOT / "configs" / "detector.yaml")
        self.assertEqual(config["model_name"], "roberta-base")
        self.assertEqual(config["max_length"], 512)
        self.assertEqual(config["epochs"], 3)
        self.assertEqual(config["seed"], 42)
        self.assertIsNone(config["limit_per_class"])
        broken = dict(config)
        broken["max_length"] = 8
        with self.assertRaises(ValueError):
            validate_config(broken)


class ScheduleTests(unittest.TestCase):
    def test_optimizer_step_count_includes_a_short_last_window(self) -> None:
        self.assertEqual(optimizer_step_count(5, 2, 3), 9)
        self.assertEqual(optimizer_step_count(4, 1, 2), 8)

    def test_linear_warmup_then_decay(self) -> None:
        self.assertAlmostEqual(learning_rate_multiplier(0, 10, 0.2), 0.5)
        self.assertAlmostEqual(learning_rate_multiplier(1, 10, 0.2), 1.0)
        self.assertAlmostEqual(learning_rate_multiplier(2, 10, 0.2), 1.0)
        self.assertAlmostEqual(learning_rate_multiplier(9, 10, 0.2), 0.125)
        self.assertAlmostEqual(learning_rate_multiplier(0, 10, 0.0), 1.0)
        self.assertAlmostEqual(learning_rate_multiplier(9, 10, 0.0), 0.1)


class ReportTests(unittest.TestCase):
    def test_split_report_writes_metrics_and_charts(self) -> None:
        examples = [
            _row("a", "Semantic Noise", "test", "A ferry crossed the harbour."),
            _row("b", "Semantic Noise", "test", "The final was played in Paris."),
            _row("c", "Supportive Noise", "test", "Ada wn the pize."),
        ]
        evaluation = {
            "loss": 0.4,
            "true_ids": [1, 1, 5],
            "predicted_ids": [5, 1, 1],
            "example_ids": ["a", "b", "c"],
            "probabilities": [0.7, 0.8, 0.6],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            written = write_split_report(
                output,
                "test",
                evaluation,
                {row["example_id"]: row for row in examples},
            )
            metrics = json.loads(Path(written["metrics"]).read_text(encoding="utf-8"))
            matrix = json.loads(Path(written["confusion_matrix"]).read_text(encoding="utf-8"))
            self.assertEqual(matrix["rows"], "true class")
            self.assertEqual(matrix["columns"], "predicted class")
            self.assertEqual(matrix["matrix"][1][5], 1)
            self.assertAlmostEqual(metrics["accuracy"], 1 / 3, places=6)
            self.assertGreater(Path(written["confusion_matrix_figure"]).stat().st_size, 0)
            self.assertGreater(Path(written["f1_figure"]).stat().st_size, 0)
            predictions = [
                json.loads(line)
                for line in Path(written["predictions"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(predictions[0]["passage_text"], "A ferry crossed the harbour.")
            self.assertEqual(predictions[0]["true_label"], "Semantic Noise")
            self.assertEqual(predictions[0]["predicted_label"], "Supportive Noise")
            self.assertFalse(predictions[0]["correct"])
            self.assertTrue(predictions[1]["correct"])

    def test_prediction_rows_reject_a_short_probability_list(self) -> None:
        evaluation = {
            "true_ids": [0],
            "predicted_ids": [0],
            "example_ids": ["a"],
            "probabilities": [],
        }
        with self.assertRaises(ValueError):
            prediction_rows({"a": _row("a", "Relevant", "test")}, evaluation)


class RealDataTests(unittest.TestCase):
    @unittest.skipUnless(REAL_DATA.is_file(), "labelled dataset is not on this machine")
    def test_processed_file_has_the_expected_split_sizes(self) -> None:
        examples = load_examples(REAL_DATA)
        grouped = split_examples(examples)
        self.assertEqual(len(examples), 4000)
        self.assertEqual({split: len(rows) for split, rows in grouped.items()}, {
            "train": 3200,
            "validation": 400,
            "test": 400,
        })
        for split, rows in grouped.items():
            require_all_labels(rows, split)
            counts = {label: 0 for label in LABELS}
            for row in rows:
                counts[row["label"]] += 1
                self.assertEqual(row["label_id"], LABEL_TO_ID[row["label"]])
            self.assertTrue(all(count == len(rows) // 8 for count in counts.values()))


def _training_config() -> dict:
    return {
        "seed": 0,
        "max_length": 32,
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "epochs": 1,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "max_grad_norm": 1.0,
        "early_stopping_patience": None,
    }


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TrainingLoopTests(unittest.TestCase):
    def test_cross_entropy_matches_negative_log_probability(self) -> None:
        import torch

        from rag_noise.detector.engine import sequence_classification_loss

        logits = torch.tensor([[4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
        labels = torch.tensor([0])
        manual = -math.log(torch.softmax(logits, dim=-1)[0, 0].item())
        self.assertAlmostEqual(sequence_classification_loss(logits, labels).item(), manual, places=5)

    def test_one_epoch_updates_and_keeps_every_validation_row(self) -> None:
        import torch

        from rag_noise.detector.engine import evaluate_examples, train_model

        class RecordingTokenizer:
            def __init__(self) -> None:
                self.passages: list[str] = []

            def __call__(self, question, passage, truncation, max_length, add_special_tokens):
                self.passages.append(passage)
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

            def pad(self, features, return_tensors="pt"):
                max_len = max(len(feature["input_ids"]) for feature in features)
                input_ids = []
                mask = []
                for feature in features:
                    extra = max_len - len(feature["input_ids"])
                    input_ids.append(feature["input_ids"] + [0] * extra)
                    mask.append(feature["attention_mask"] + [0] * extra)
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(mask, dtype=torch.long),
                }

        class TinyClassifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = torch.nn.Embedding(20, 8)
                self.score = torch.nn.Linear(8, 8)

            def forward(self, input_ids, attention_mask):
                hidden = self.embed(input_ids)
                weights = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
                return self.score(pooled)

        def rows_for(split: str, copies: int) -> list[dict]:
            built = []
            for label in LABELS:
                for copy in range(copies):
                    built.append(_row(
                        f"{split}-{LABEL_TO_ID[label]}-{copy}",
                        label,
                        split,
                        f"{label} passage {copy}.",
                    ))
            return built

        train_rows = rows_for("train", 2)
        validation_rows = rows_for("validation", 1)
        tokenizer = RecordingTokenizer()
        model = TinyClassifier()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        from rag_noise.detector.engine import training_device

        self.assertEqual(training_device("cpu").type, "cpu")
        notes: list[str] = []
        result = train_model(
            model,
            tokenizer,
            train_rows,
            validation_rows,
            _training_config(),
            torch.device("cpu"),
            log=notes.append,
        )
        self.assertTrue(any("batch 1/" in note and "starting" in note for note in notes))
        self.assertTrue(any(note.startswith("epoch 1/1 validation") for note in notes))
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(len(result["history"]), 1)
        self.assertTrue(math.isfinite(result["history"][0]["train_loss"]))
        self.assertEqual(
            result["best_validation"]["example_ids"],
            [row["example_id"] for row in validation_rows],
        )
        self.assertTrue(any(torch.any(old != parameter.detach()) for old, parameter in zip(before, model.parameters(), strict=True)))
        self.assertIn("Relevant passage 0.", tokenizer.passages)
        scored = evaluate_examples(
            model,
            tokenizer,
            validation_rows,
            _training_config(),
            torch.device("cpu"),
        )
        self.assertEqual(len(scored["predicted_ids"]), len(validation_rows))
        self.assertTrue(all(0 <= value <= 1 for value in scored["probabilities"]))


class PlotTests(unittest.TestCase):
    def test_figures_are_written(self) -> None:
        matrix = [[1 if row == column else 0 for column in range(8)] for row in range(8)]
        per_class = [{"label": label, "f1": 0.25} for label in LABELS]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            confusion_path = folder / "confusion.png"
            f1_path = folder / "f1.png"
            confusion_matrix_figure(matrix, "test confusion matrix", confusion_path)
            per_class_f1_figure(per_class, "test F1 by class", f1_path)
            self.assertGreater(confusion_path.stat().st_size, 0)
            self.assertGreater(f1_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
