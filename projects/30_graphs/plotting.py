"""
Visualizations for the MPNN pipeline: what the raw graphs actually look
like, what graph batching actually does to them, and how training went.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from batching import BatchedGraphs
from data_load_prep import MoleculeGraph

CATEGORICAL_CMAP = "tab10"


def plot_molecule_examples(
    graphs: list[MoleculeGraph],
    path: Path,
    num_examples: int = 6,
) -> None:
    """A handful of real molecules spanning a range of sizes -- the
    concrete picture behind "molecules don't share a node count", which
    is the whole reason `batching.py` exists.
    """
    sizes = [graph.atom_features.shape[0] for graph in graphs]
    order = np.argsort(sizes)
    sampled_positions = np.linspace(
        0,
        len(graphs) - 1,
        num_examples,
    ).astype(int)
    sampled_indices = order[sampled_positions]

    fig, axes = plt.subplots(
        1,
        num_examples,
        figsize=(3 * num_examples, 3.2),
    )
    for ax, index in zip(axes, sampled_indices):
        graph = graphs[index]
        mol = Chem.MolFromSmiles(graph.smiles)
        image = Draw.MolToImage(
            mol,
            size=(250, 250),
        )
        compound_id = graph.compound_id
        if len(compound_id) > 20:
            compound_id = compound_id[:17] + "..."

        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{compound_id}\n{sizes[index]} atoms",
            fontsize=9,
        )

    fig.suptitle("Molecular graphs come in very different sizes")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_batch_structure(
    batch: BatchedGraphs,
    path: Path,
) -> None:
    """KEY SKILL: graph batching, made visible -- every row of the
    batch's concatenated atom_features tensor, colored by WHICH graph
    (`graph_index`) it actually belongs to. This is the disjoint-union
    idea directly: underneath, a "batch" is just one bigger graph made
    of several disconnected pieces.
    """
    graph_index = batch.graph_index.numpy()
    boundaries = np.where(np.diff(graph_index) != 0)[0] + 1

    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.imshow(
        graph_index[None, :],
        cmap=CATEGORICAL_CMAP,
        aspect="auto",
    )
    for boundary in boundaries:
        ax.axvline(
            boundary - 0.5,
            color="black",
            linewidth=0.8,
        )
    ax.set_yticks([])
    ax.set_xlabel("Row in the batch's concatenated atom_features tensor")
    ax.set_title(
        f"One batch, {batch.num_graphs} molecules, {graph_index.shape[0]} total atoms "
        "-- color = which molecule that row belongs to",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    history: dict[str, list[float]],
    path: Path,
) -> None:
    epochs = np.arange(1, len(history["train_mse"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_mse"], label="train")
    ax.plot(epochs, history["val_mse"], label="val")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log solubility^2)")
    ax.set_title("MPNN training: mean squared error")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_parity(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Predicted vs. actual -- a perfect model's points all sit on the
    y=x line. The same diagnostic weeks 1-2's linear regression on this
    exact dataset would have used, now for a learned graph representation
    instead of hand-engineered descriptors.
    """
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)

    lo = min(predictions.min(), targets.min())
    hi = max(predictions.max(), targets.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        targets,
        predictions,
        alpha=0.6,
        color="tab:blue",
    )
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="gray",
        linestyle="--",
        label="y = x",
    )
    ax.set_xlabel("Measured log solubility")
    ax.set_ylabel("Predicted log solubility")
    ax.set_title(f"{title} (RMSE={rmse:.3f}, R^2={r2:.3f})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
