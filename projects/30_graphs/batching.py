"""
KEY SKILL: graph batching. Molecules don't share a node count (1-55
atoms even within this one dataset), so unlike a batch of fixed-size
vectors, there is no single `(B, N, F)` tensor every graph fits into
without choosing a strategy.

This uses the DISJOINT-UNION approach (the same one PyTorch Geometric
uses internally) rather than padding: every graph in the batch gets
concatenated into ONE big node-feature tensor and ONE big edge-index
tensor, with each graph's own edge indices shifted by a running atom
offset so they still point at the right rows after concatenation. A
`graph_index` vector then records which graph each row of the big node
tensor actually belongs to, which is what lets per-graph pooling
(`mpnn.py`'s readout) work on a batch that is, underneath, just one
enormous graph with several disconnected components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_load_prep import MoleculeGraph


@dataclass
class BatchedGraphs:
    atom_features: torch.Tensor  # (total_atoms, NODE_FEATURE_DIM)
    bond_index: torch.Tensor  # (2, total_directed_edges), indices into atom_features' rows
    bond_features: torch.Tensor  # (total_directed_edges, EDGE_FEATURE_DIM)
    graph_index: torch.Tensor  # (total_atoms,) long, which graph (0..num_graphs-1) each atom belongs to
    targets: torch.Tensor  # (num_graphs,)
    num_graphs: int


def collate_molecule_graphs(
    graphs: list[MoleculeGraph],
) -> BatchedGraphs:
    atom_feature_chunks = []
    bond_index_chunks = []
    bond_feature_chunks = []
    graph_index_chunks = []
    targets = []

    atom_offset = 0
    for graph_position, graph in enumerate(graphs):
        num_atoms = graph.atom_features.shape[0]

        atom_feature_chunks.append(graph.atom_features)
        bond_feature_chunks.append(graph.bond_features)
        # shifting by `atom_offset` is what keeps this graph's own
        # 0-indexed bond_index pointing at ITS rows in the concatenated
        # atom_features tensor, rather than colliding with graph 0's rows
        bond_index_chunks.append(graph.bond_index + atom_offset)
        graph_index_chunks.append(
            np.full(num_atoms, graph_position, dtype=np.int64),
        )
        targets.append(graph.target)

        atom_offset += num_atoms

    return BatchedGraphs(
        atom_features=torch.from_numpy(
            np.concatenate(atom_feature_chunks, axis=0),
        ),
        bond_index=torch.from_numpy(
            np.concatenate(bond_index_chunks, axis=1),
        ),
        bond_features=torch.from_numpy(
            np.concatenate(bond_feature_chunks, axis=0),
        ),
        graph_index=torch.from_numpy(
            np.concatenate(graph_index_chunks, axis=0),
        ),
        targets=torch.tensor(
            targets,
            dtype=torch.float32,
        ),
        num_graphs=len(graphs),
    )
