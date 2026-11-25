from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_load_prep import AA_TO_INDEX
from data_load_prep import PAD_IDX
from masking import IGNORE_INDEX
from masking import mask_batch


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


def tokenize_batch(
    sequences: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad `sequences` to their batch max length. Same convention as
    `16_transformer_train_encoder/data_load_prep.py`: `key_padding_mask`
    is True where a position is padding, matching
    `nn.TransformerEncoderLayer`'s `src_key_padding_mask` argument.
    """
    lengths = torch.tensor(
        [len(seq) for seq in sequences],
        dtype=torch.long,
    )
    max_len = int(lengths.max())

    ids = torch.full(
        (len(sequences), max_len),
        PAD_IDX,
        dtype=torch.long,
    )
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            ids[i, j] = AA_TO_INDEX[aa]

    positions = torch.arange(max_len).unsqueeze(0)  # (1, max_len)
    key_padding_mask = positions >= lengths.unsqueeze(1)  # (batch, max_len), True = pad

    return ids, key_padding_mask


def _iterate_batches(
    sequences: list[str],
    batch_size: int,
    shuffle: bool,
):
    n = len(sequences)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield [sequences[i] for i in idx.tolist()]


def _run_epoch(
    model: nn.Module,
    sequences: list[str],
    device: torch.device,
    batch_size: int,
    mask_prob: float,
    optimizer: torch.optim.Optimizer | None = None,
    generator: torch.Generator | None = None,
) -> tuple[float, float]:
    
    
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )

    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_scored = 0

    for batch_seqs in _iterate_batches(
        sequences,
        batch_size,
        shuffle=is_training,
    ):
        ids, key_padding_mask = tokenize_batch(
            batch_seqs,
        )
        # KEY SKILL: masking strategies -- corrupt the input here, fresh
        # every batch (every epoch sees a DIFFERENT random 15% hidden),
        # which is part of why self-supervised pretraining can keep
        # learning from the same sequences across many epochs without
        # immediately overfitting to one fixed set of blanks
        corrupted_ids, labels = mask_batch(
            ids,
            key_padding_mask,
            mask_prob=mask_prob,
            generator=generator,
        )
        corrupted_ids = corrupted_ids.to(device)
        labels = labels.to(device)
        key_padding_mask = key_padding_mask.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(corrupted_ids, key_padding_mask)  # (B, L, vocab)
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )

        n_scored = (labels != IGNORE_INDEX).sum().item()
        if is_training:
            (loss / max(n_scored, 1)).backward()
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        predictions = logits.argmax(dim=-1)
        scored_positions = labels != IGNORE_INDEX
        total_correct += (predictions[scored_positions] == labels[scored_positions]).sum().item()
        total_scored += n_scored

    avg_loss = total_loss / max(total_scored, 1)
    accuracy = total_correct / max(total_scored, 1)
    return avg_loss, accuracy


def fit(
    model: nn.Module,
    device: torch.device,
    train_sequences: list[str],
    val_sequences: list[str],
    epochs: int = 15,
    batch_size: int = 32,
    mask_prob: float = 0.15,
    lr: float = 3e-4,
    checkpoint_path: Path | None = None,
) -> dict[str, list[float]]:
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
        "val_masked_acc": [],
    }
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss, _ = _run_epoch(
            model,
            train_sequences,
            device,
            batch_size,
            mask_prob,
            optimizer,
        )
        # validation uses a FIXED masking seed every epoch (via
        # `evaluate`, below), unlike training -- so the model is graded
        # on the literal same hidden positions each time, and any change
        # in val_loss reflects the model improving, not the quiz
        # changing. This is also what makes the checkpoint round-trip
        # check in `main.py` exact: reloading this checkpoint and calling
        # `evaluate(..., seed=0)` again must reproduce this same number.
        val_loss, val_acc = evaluate(
            model,
            val_sequences,
            device,
            batch_size,
            mask_prob,
            seed=0,
        )

        # KEY SKILL: validation loss -- perplexity (exp(cross-entropy
        # loss)) is just loss on a more interpretable scale: it's
        # "roughly how many amino acids the model is torn between" at a
        # masked position. A perplexity near 20 means total confusion
        # (as bad as guessing uniformly among all 20 residues);
        # perplexity near 1 means near-perfect confidence in the right
        # answer.
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)

        history["train_loss"].append(train_loss)
        history["train_ppl"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["val_masked_acc"].append(val_acc)

        print(
            f"epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} "
            f"val_masked_acc={val_acc:.3f}",
        )

        # KEY SKILL: checkpoint management -- save the model only when it
        # genuinely improved on held-out data, so the saved artifact is
        # always the best-seen model, not just whatever the last epoch
        # happened to produce (which could be a worse, overfit epoch)
        if checkpoint_path is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

    return history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    sequences: list[str],
    device: torch.device,
    batch_size: int = 32,
    mask_prob: float = 0.15,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Same as the validation half of `_run_epoch`, but with a FIXED masking
    seed -- used to prove a reloaded checkpoint actually round-trips:
    evaluating the same sequences with the same seed before saving and
    after reloading should give identical (loss, accuracy), since
    identical masking + identical weights must produce identical output.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return _run_epoch(
        model,
        sequences,
        device,
        batch_size,
        mask_prob,
        optimizer=None,
        generator=generator,
    )
