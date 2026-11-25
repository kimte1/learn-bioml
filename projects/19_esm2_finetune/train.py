"""
Regression training loop shared by all 4 fine-tuning strategies -- only
`model.parameters()`'s `requires_grad` flags (set in `models.py`) differ
between them; the training mechanics here don't know or care which
strategy produced the model they're training.

Evaluated with Spearman rank correlation, not MSE/RMSE/accuracy: the
actual ProteinGym benchmark cares about whether the model ranks mutants
in the right order from least to most fit, not whether its predicted
`DMS_score` matches the measured one in absolute units (the measured
score's scale/zero-point is assay-specific and somewhat arbitrary).
Spearman correlation measures exactly "rank order agreement" and ignores
absolute scale, which is why it's the standard metric here instead of a
scale-sensitive one like RMSE.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from transformers import AutoTokenizer

from models import CHECKPOINT

_tokenizer = AutoTokenizer.from_pretrained(
    CHECKPOINT,
)


def get_device() -> torch.device:
    """check if MPS (Apple Silicon GPU) is available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _iterate_batches(
    sequences: list[str],
    scores: list[float],
    batch_size: int,
    shuffle: bool,
):
    n = len(sequences)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size].tolist()
        batch_sequences = [sequences[i] for i in idx]
        batch_scores = [scores[i] for i in idx]
        yield batch_sequences, batch_scores


@torch.no_grad()
def predict(
    model: nn.Module,
    sequences: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Runs the model over every sequence and returns its raw predicted
    `DMS_score`s, as a plain numpy array -- used both for the "before
    training" snapshot and for final test-set evaluation.
    """
    model.eval()
    all_predictions = []
    for batch_sequences, _ in _iterate_batches(
        sequences,
        [0.0] * len(sequences),
        batch_size,
        shuffle=False,
    ):
        encoded = _tokenizer(
            batch_sequences,
            return_tensors="pt",
        )
        predictions = model(
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )
        all_predictions.append(
            predictions.cpu().numpy(),
        )

    return np.concatenate(
        all_predictions,
    )


def evaluate(
    model: nn.Module,
    sequences: list[str],
    scores: list[float],
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mse, spearman_r) on the given set."""

    predictions = predict(
        model,
        sequences,
        device,
    )
    true_scores = np.array(scores)

    mse = float(
        np.mean((predictions - true_scores) ** 2),
    )
    spearman_r = float(
        spearmanr(predictions, true_scores).correlation,
    )
    return mse, spearman_r


def fit(
    model: nn.Module,
    device: torch.device,
    train_sequences: list[str],
    train_scores: list[float],
    val_sequences: list[str],
    val_scores: list[float],
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> dict[str, list[float]]:
    """
    A plain training loop -- MSE loss, AdamW, no gradient accumulation.
    """
    model.to(device)
    trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=lr,
    )
    loss_fn = nn.MSELoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_spearman": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_examples = 0

        for batch_sequences, batch_scores in _iterate_batches(
            train_sequences,
            train_scores,
            batch_size,
            shuffle=True,
        ):
            encoded = _tokenizer(
                batch_sequences,
                return_tensors="pt",
            )
            targets = torch.tensor(
                batch_scores,
                dtype=torch.float32,
            ).to(device)

            predictions = model(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_sequences)
            n_examples += len(batch_sequences)

        val_mse, val_spearman = evaluate(
            model,
            val_sequences,
            val_scores,
            device,
        )
        history["train_loss"].append(
            epoch_loss / n_examples,
        )
        history["val_loss"].append(
            val_mse,
        )
        history["val_spearman"].append(
            val_spearman,
        )
        print(
            f"  epoch {epoch + 1}/{epochs}: train_mse={epoch_loss / n_examples:.4f} "
            f"val_mse={val_mse:.4f} val_spearman={val_spearman:.3f}",
        )

    return history
