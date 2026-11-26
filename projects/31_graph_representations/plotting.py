"""
Visualizations comparing the different graph representations built
this week: what a molecular graph actually looks like, how two
different edge-construction rules connect the SAME set of residues
very differently, and how node degree is distributed across all of it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from graph_types import Graph
from graph_types import degree

CATEGORICAL_CMAP = "tab10"


def plot_molecule(
    smiles: str,
    title: str,
    path: Path,
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    image = Draw.MolToImage(
        mol,
        size=(350, 350),
    )

    fig, ax = plt.subplots(figsize=(4, 4.3))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_adjacency_comparison(
    graphs: list[Graph],
    path: Path,
) -> None:
    """Two (or more) graphs over the SAME kind of nodes, rendered as
    dense adjacency matrices side by side -- the most direct way to see
    that "sequence-adjacency" and "spatial contact" are genuinely
    different edge sets, not just different names for the same thing.
    """
    fig, axes = plt.subplots(
        1,
        len(graphs),
        figsize=(5 * len(graphs), 5),
    )
    if len(graphs) == 1:
        axes = [axes]

    for ax, graph in zip(axes, graphs):
        num_nodes = graph.node_features.shape[0]
        adjacency = np.zeros(
            (num_nodes, num_nodes),
            dtype=bool,
        )
        adjacency[graph.edge_index[0], graph.edge_index[1]] = True

        ax.imshow(
            adjacency,
            cmap="Greys",
        )
        ax.set_title(graph.name, fontsize=10)
        ax.set_xlabel("Node index")
        ax.set_ylabel("Node index")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_degree_distributions(
    graphs: list[Graph],
    path: Path,
) -> None:
    """Density, not raw count -- these graphs range from 13 to 327
    nodes, so a shared COUNT axis would let the biggest graph's bars
    visually swamp every smaller graph's distribution. Density puts
    them all on "what fraction of this graph's nodes have degree k",
    which is comparable regardless of how many nodes each graph has.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap(CATEGORICAL_CMAP)

    for color_index, graph in enumerate(graphs):
        degrees = degree(graph)
        ax.hist(
            degrees,
            bins=np.arange(degrees.min(), degrees.max() + 2) - 0.5,
            density=True,
            alpha=0.5,
            label=f"{graph.name} (n={len(degrees)})",
            color=cmap(color_index),
        )

    ax.set_xlabel("Node degree")
    ax.set_ylabel("Fraction of nodes")
    ax.set_title("Degree distribution across graph representations (density, not count)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison_table(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    display_df = summary.copy()
    display_df["avg_degree"] = display_df["avg_degree"].map(lambda x: f"{x:.1f}")

    fig, ax = plt.subplots(figsize=(13, 0.5 * len(display_df) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    table.auto_set_column_width(col=list(range(len(display_df.columns))))

    ax.set_title("Comparing graph representations", pad=20)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
