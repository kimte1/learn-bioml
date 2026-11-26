"""
A standard regression training loop -- MSE loss, Adam -- the only thing
specific to graphs here is that each minibatch has to be built with
`collate_molecule_graphs` instead of `torch.stack`.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from batching import BatchedGraphs
from batching import collate_molecule_graphs
from data_load_prep import MoleculeGraph
from mpnn import MPNN


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


def _to_device(
    batch: BatchedGraphs,
    device: torch.device,
) -> BatchedGraphs:
    return BatchedGraphs(
        atom_features=batch.atom_features.to(device),
        bond_index=batch.bond_index.to(device),
        bond_features=batch.bond_features.to(device),
        graph_index=batch.graph_index.to(device),
        targets=batch.targets.to(device),
        num_graphs=batch.num_graphs,
    )


def _forward(
    model: MPNN,
    batch: BatchedGraphs,
) -> torch.Tensor:
    return model(
        batch.atom_features,
        batch.bond_index,
        batch.bond_features,
        batch.graph_index,
        batch.num_graphs,
    )


def _run_epoch(
    model: MPNN,
    graphs: list[MoleculeGraph],
    device: torch.device,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """One pass over `graphs`, minibatched via `collate_molecule_graphs`.
    Pass `optimizer=None` for an eval-only pass (no gradient step).
    """
    is_training = optimizer is not None
    model.train(is_training)

    order = np.random.permutation(len(graphs)) if is_training else np.arange(len(graphs))

    total_loss = 0.0
    total_graphs = 0
    for start in range(0, len(graphs), batch_size):
        chunk_indices = order[start : start + batch_size]
        chunk = [graphs[i] for i in chunk_indices]

        batch = _to_device(
            collate_molecule_graphs(chunk),
            device,
        )

        with torch.set_grad_enabled(is_training):
            predictions = _forward(model, batch)
            loss = nn.functional.mse_loss(predictions, batch.targets)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / total_graphs


def fit(
    model: MPNN,
    train_graphs: list[MoleculeGraph],
    val_graphs: list[MoleculeGraph],
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    history: dict[str, list[float]] = {
        "train_mse": [],
        "val_mse": [],
    }
    for _ in range(epochs):
        train_mse = _run_epoch(
            model,
            train_graphs,
            device,
            batch_size,
            optimizer,
        )
        val_mse = _run_epoch(
            model,
            val_graphs,
            device,
            batch_size,
            optimizer=None,
        )
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)

    return history


@torch.no_grad()
def predict(
    model: MPNN,
    graphs: list[MoleculeGraph],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predictions, targets) over `graphs`, IN THE ORDER GIVEN
    (no shuffling) -- unlike `_run_epoch`, callers need these aligned
    with the original graph list for plotting/metrics.
    """
    model.eval()

    all_predictions = []
    all_targets = []
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start : start + batch_size]
        batch = _to_device(
            collate_molecule_graphs(chunk),
            device,
        )
        predictions = _forward(model, batch)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(batch.targets.cpu().numpy())

    return np.concatenate(all_predictions), np.concatenate(all_targets)
