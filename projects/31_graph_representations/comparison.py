"""
Puts every graph this week built on one summary table, regardless of
domain -- the concrete answer to "compare alternative node/edge
features" across molecular graphs, residue graphs (two edge schemes),
and an atomic protein graph.
"""

from __future__ import annotations

import pandas as pd

from graph_types import Graph
from graph_types import degree


def summarize_graphs(
    graphs: list[Graph],
) -> pd.DataFrame:
    rows = []
    for graph in graphs:
        num_nodes = graph.node_features.shape[0]
        num_undirected_edges = graph.edge_index.shape[1] // 2
        degrees = degree(graph)

        rows.append(
            {
                "graph": graph.name,
                "num_nodes": num_nodes,
                "num_edges": num_undirected_edges,
                "avg_degree": degrees.mean(),
                "node_feature_dim": graph.node_features.shape[1],
                "edge_feature_dim": graph.edge_features.shape[1] if graph.edge_features is not None else 0,
                "edges_carry_geometry": graph.edge_features is not None and graph.node_coordinates is not None,
            },
        )

    return pd.DataFrame(rows)
