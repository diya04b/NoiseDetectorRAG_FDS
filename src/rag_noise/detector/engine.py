"""Train and score the detector. The classification loss is cross-entropy."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from rag_noise.detector.data import encode_pair
from rag_noise.detector.schedule import learning_rate_multiplier, optimizer_step_count


def sequence_classification_loss(logits: Any, labels: Any) -> Any:
    """Cross-entropy between the eight class scores and the true class."""
    return torch.nn.functional.cross_entropy(logits, labels)


def _logits_of(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    return output


def _collate(tokenizer: Any, items: list[dict[str, Any]]) -> tuple[dict[str, Any], Any, list[str]]:
    example_ids = [item["example_id"] for item in items]
    labels = torch.tensor([item["label_id"] for item in items], dtype=torch.long)
    feature_keys = [key for key in items[0] if key not in {"example_id", "label_id"}]
    features = [{key: item[key] for key in feature_keys} for item in items]
    batch = tokenizer.pad(features, return_tensors="pt")
    return batch, labels, example_ids


class _PairDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.examples[index]
        features = encode_pair(self.tokenizer, row["question"], row["passage_text"], self.max_length)
        features["label_id"] = int(row["label_id"])
        features["example_id"] = str(row["example_id"])
        return features


def _loader(examples: list[dict[str, Any]], tokenizer: Any, config: dict[str, Any], shuffle: bool, seed: int) -> DataLoader:
    dataset = _PairDataset(examples, tokenizer, int(config["max_length"]))
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    batch_size = int(config["train_batch_size"] if shuffle else config["eval_batch_size"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=lambda items: _collate(tokenizer, items),
    )


def set_seed(seed: int, device: torch.device | None = None) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    # A CPU run must not touch CUDA. On a machine whose GPU library is
    # installed but unusable, that probe can fail before the first batch.
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def training_device(requested: str = "cpu") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this machine has no usable GPU. Run again with --device cpu."
            )
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError("device must be auto, cpu, or cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _emit(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)


def evaluate_examples(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    device: Any,
    log: Callable[[str], None] | None = None,
    progress_label: str = "scoring",
) -> dict[str, Any]:
    """Score examples without updating the model."""
    if not examples:
        raise ValueError("Cannot evaluate an empty split")
    loader = _loader(examples, tokenizer, config, shuffle=False, seed=int(config["seed"]))
    n_batches = len(loader)
    _emit(log, f"{progress_label}: {len(examples)} rows, {n_batches} batches on {device}")
    model.eval()
    total_loss = 0.0
    seen = 0
    true_ids: list[int] = []
    predicted_ids: list[int] = []
    example_ids: list[str] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch_index, (batch, labels, batch_ids) in enumerate(loader):
            if batch_index == 0:
                _emit(log, f"{progress_label} batch 1/{n_batches} starting")
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = labels.to(device)
            logits = _logits_of(model(**batch))
            loss = sequence_classification_loss(logits, labels)
            if not math.isfinite(float(loss.item())):
                raise RuntimeError("Evaluation loss is not finite")
            total_loss += float(loss.item()) * int(labels.size(0))
            seen += int(labels.size(0))
            probs = torch.softmax(logits, dim=-1)
            predicted = probs.argmax(dim=-1)
            confidence = probs.gather(1, predicted.unsqueeze(1)).squeeze(1)
            true_ids.extend(int(value) for value in labels.detach().cpu().tolist())
            predicted_ids.extend(int(value) for value in predicted.detach().cpu().tolist())
            probabilities.extend(float(value) for value in confidence.detach().cpu().tolist())
            example_ids.extend(batch_ids)
            if (batch_index + 1) % 10 == 0 or (batch_index + 1) == n_batches:
                _emit(log, f"{progress_label} batch {batch_index + 1}/{n_batches}")
    if seen != len(examples):
        raise RuntimeError(f"Evaluated {seen} rows, expected {len(examples)}")
    return {
        "loss": total_loss / seen,
        "true_ids": true_ids,
        "predicted_ids": predicted_ids,
        "example_ids": example_ids,
        "probabilities": probabilities,
    }


def train_model(
    model: Any,
    tokenizer: Any,
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    config: dict[str, Any],
    device: Any,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
    on_improvement: Callable[[int, dict[str, Any]], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fine-tune with cross-entropy. Keep the epoch with the best validation macro-F1."""
    from rag_noise.detector.metrics import score_predictions

    if not train_examples or not validation_examples:
        raise ValueError("Training needs both a train split and a validation split")
    epochs = int(config["epochs"])
    set_seed(int(config["seed"]), device)
    _emit(
        log,
        f"Training on {device}: {len(train_examples)} rows, "
        f"batch size {int(config['train_batch_size'])}, {epochs} epochs. "
        "The first batch is slow on CPU because the model runs there.",
    )
    model.to(device)
    base_lr = float(config["learning_rate"])
    accumulation = int(config["gradient_accumulation_steps"])
    patience = config["early_stopping_patience"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(config["weight_decay"]),
    )
    loader = _loader(train_examples, tokenizer, config, shuffle=True, seed=int(config["seed"]))
    n_batches = len(loader)
    planned_steps = optimizer_step_count(n_batches, accumulation, epochs)
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    best_validation: dict[str, Any] | None = None
    waits = 0
    completed_steps = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        _emit(log, f"epoch {epoch}/{epochs}: training, {n_batches} batches")
        for batch_index, (batch, labels, _example_ids) in enumerate(loader):
            if batch_index == 0:
                _emit(log, f"epoch {epoch}/{epochs} batch 1/{n_batches} starting")
            if batch_index % accumulation == 0:
                multiplier = learning_rate_multiplier(completed_steps, planned_steps, float(config["warmup_ratio"]))
                for group in optimizer.param_groups:
                    group["lr"] = base_lr * multiplier
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = labels.to(device)
            logits = _logits_of(model(**batch))
            raw_loss = sequence_classification_loss(logits, labels)
            if not math.isfinite(float(raw_loss.item())):
                raise RuntimeError(
                    "Training loss is not finite. Lower learning_rate in configs/detector.yaml and run again."
                )
            (raw_loss / accumulation).backward()
            running_loss += float(raw_loss.item()) * int(labels.size(0))
            seen += int(labels.size(0))
            window_done = (batch_index + 1) % accumulation == 0 or (batch_index + 1) == n_batches
            if window_done:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
            if (batch_index + 1) % 10 == 0 or (batch_index + 1) == n_batches:
                _emit(
                    log,
                    f"epoch {epoch}/{epochs} batch {batch_index + 1}/{n_batches} "
                    f"train_loss={running_loss / seen:.4f}",
                )
        if seen != len(train_examples):
            raise RuntimeError(f"Trained on {seen} rows, expected {len(train_examples)}")

        validation = evaluate_examples(
            model,
            tokenizer,
            validation_examples,
            config,
            device,
            log=log,
            progress_label=f"epoch {epoch}/{epochs} validation",
        )
        scores = score_predictions(validation["true_ids"], validation["predicted_ids"])
        record = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "validation_loss": validation["loss"],
            "validation_accuracy": scores["accuracy"],
            "validation_macro_f1": scores["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "n_train": len(train_examples),
            "n_validation": len(validation_examples),
        }
        history.append(record)
        if on_epoch is not None:
            on_epoch(record)
        if scores["macro_f1"] > best_score:
            best_score = scores["macro_f1"]
            best_epoch = epoch
            waits = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_validation = validation
            if on_improvement is not None:
                on_improvement(epoch, scores)
        else:
            waits += 1
            if patience is not None and waits >= int(patience):
                break

    if best_state is None or best_validation is None:
        raise RuntimeError("Training finished without a saved epoch")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "best_validation": best_validation,
        "optimizer_steps": completed_steps,
    }
