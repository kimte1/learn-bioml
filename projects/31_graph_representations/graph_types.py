"""
One domain-agnostic `Graph` container used for every representation
this week builds -- molecular, residue, and atomic -- so `comparison.py`
and `plotting.py` can compare them side by side without caring which
domain a particular graph came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    name: str  # human-readable label, e.g. "1CRN atomic graph (distance-based edges)"
    node_labels: list[str]  # one short string per node (element symbol, or "A26")
    node_features: np.ndarray  # (num_nodes, node_feature_dim)
    edge_index: np.ndarray  # (2, num_directed_edges) -- every undirected edge stored BOTH directions
    edge_features: np.ndarray | None  # (num_directed_edges, edge_feature_dim), or None if edges carry no features
    node_coordinates: np.ndarray | None = None  # (num_nodes, 3) real 3D positions, where they exist


def degree(
    graph: Graph,
) -> np.ndarray:
    """Per-node degree. `edge_index[0]` already has one entry per
    DIRECTED edge (every real edge appears as both i->j and j->i), so a
    plain bincount over source nodes gives each node's true (undirected)
    degree directly -- no separate factor-of-2 correction needed.
    """
    return np.bincount(
        graph.edge_index[0],
        minlength=graph.node_features.shape[0],
    )
