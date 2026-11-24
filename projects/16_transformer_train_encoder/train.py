from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_load_prep import tokenize_batch


def get_device() -> torch.device:
    """check if MPS (Apple Silicon GPU) is available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _iterate_batches(
    sequences: list[str], 
    labels: torch.Tensor, 
    batch_size: int, 
    shuffle: bool
):
    n = len(sequences)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield [sequences[i] for i in idx.tolist()], labels[idx]


def _run_epoch(
    model: nn.Module,
    sequences: list[str],
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
    accumulation_steps: int = 1,
) -> tuple[float, float]:
    """
    One pass over the data. If `optimizer` is given, trains with
    gradient accumulation: the loss from each micro-batch is scaled by
    `1/accumulation_steps` before `backward()`, and `optimizer.step()` /
    `zero_grad()` only run every `accumulation_steps` micro-batches -- so
    `accumulation_steps` micro-batches of size `batch_size` behave like
    one step at effective batch size `batch_size * accumulation_steps`,
    without ever materializing that larger batch at once.
    """
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    is_training = optimizer is not None

    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    n_examples = 0
    step = -1

    if is_training:
        optimizer.zero_grad()

    for step, (batch_seqs, batch_labels) in enumerate(
        _iterate_batches(
            sequences, 
            labels, 
            batch_size, 
            shuffle=is_training
        )
    ):
        ids, key_padding_mask = tokenize_batch(batch_seqs)
        ids = ids.to(device)
        key_padding_mask = key_padding_mask.to(device)
        batch_labels = batch_labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(ids, key_padding_mask)
            loss = loss_fn(logits, batch_labels)

        if is_training:
            (loss / (len(batch_seqs) * accumulation_steps)).backward()
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
        n_examples += len(batch_seqs)

    if is_training and (step + 1) % accumulation_steps != 0:
        # the last accumulation window didn't reach a full
        # `accumulation_steps` micro-batches -- flush its gradients too,
        # or they'd silently never get applied
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / n_examples, total_correct / n_examples


def fit(
    model: nn.Module,
    device: torch.device,
    train_sequences: list[str],
    train_labels: torch.Tensor,
    val_sequences: list[str],
    val_labels: torch.Tensor,
    epochs: int = 8,
    batch_size: int = 16,
    accumulation_steps: int = 4,
    lr: float = 3e-4,
    checkpoint_path: Path | None = None,
) -> dict[str, list[float]]:
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.01
    )

    history: dict[str, list[float]] = {
        "train_loss": [], 
        "train_acc": [], 
        "val_loss": [], 
        "val_acc": []
    }
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss, train_acc = _run_epoch(
            model, 
            train_sequences, 
            train_labels, 
            device, 
            batch_size, 
            optimizer, 
            accumulation_steps,
        )
        val_loss, val_acc = _run_epoch(
            model, 
            val_sequences, 
            val_labels, 
            device, 
            batch_size, 
            optimizer=None,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if checkpoint_path is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, checkpoint_path)

    return history


@torch.no_grad()
def predict(
    model: nn.Module, 
    sequences: list[str], 
    device: torch.device, 
    batch_size: int = 32
) -> torch.Tensor:
    model.eval()
    preds = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        ids, key_padding_mask = tokenize_batch(batch)
        ids = ids.to(device)
        key_padding_mask = key_padding_mask.to(device)
        logits = model(ids, key_padding_mask)
        preds.append(logits.argmax(dim=-1).cpu())
    return torch.cat(preds)
