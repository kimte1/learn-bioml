"""
A message-passing neural network (Gilmer et al. 2017): repeat, for a
fixed number of steps, a MESSAGE step (every edge computes a message
from its two endpoint states plus its own edge features) followed by an
UPDATE step (every node folds its incoming messages into a new hidden
state via a GRU cell) -- then READ OUT a single vector per graph.

The message/update functions are the SAME weights at every step (one
`MessagePassingLayer`, called `NUM_MESSAGE_PASSING_STEPS` times), the
same way a single RNN cell is reused at every timestep: it keeps the
parameter count independent of how many rounds of message passing are
run, and it mirrors how information actually spreads through the graph
one bond-hop at a time, step after step, through the identical rule.
"""

from __future__ import annotations

import torch
from torch import nn

NUM_MESSAGE_PASSING_STEPS = 4


class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_feature_dim: int,
    ):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_cell = nn.GRUCell(
            hidden_dim,
            hidden_dim,
        )

    def forward(
        self,
        node_hidden: torch.Tensor,  # (total_atoms, hidden_dim)
        bond_index: torch.Tensor,  # (2, total_directed_edges)
        bond_features: torch.Tensor,  # (total_directed_edges, edge_feature_dim)
    ) -> torch.Tensor:
        source_nodes, destination_nodes = bond_index[0], bond_index[1]

        # KEY SKILL: message function -- every directed edge (src -> dst)
        # computes its own message from both endpoints' CURRENT hidden
        # states plus that edge's own features
        message_inputs = torch.cat(
            [
                node_hidden[source_nodes],
                node_hidden[destination_nodes],
                bond_features,
            ],
            dim=-1,
        )
        messages = self.message_mlp(message_inputs)  # (total_directed_edges, hidden_dim)

        # KEY SKILL: neighborhood aggregation -- sum every incoming
        # message into its DESTINATION node. `index_add_` is a scatter-
        # sum: this is the vectorized equivalent of, for every node,
        # looping over its neighbors and summing their messages, done in
        # one call instead of a Python loop over nodes
        aggregated = torch.zeros_like(node_hidden)
        aggregated.index_add_(0, destination_nodes, messages)

        # KEY SKILL: update function -- each node's new hidden state
        # depends on its OWN previous state (`node_hidden`, the GRU's
        # hidden input) and what it just heard from its neighbors
        # (`aggregated`, the GRU's data input)
        return self.update_cell(aggregated, node_hidden)


class MPNN(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        num_steps: int = NUM_MESSAGE_PASSING_STEPS,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.atom_embed = nn.Linear(
            node_feature_dim,
            hidden_dim,
        )
        self.message_passing_layer = MessagePassingLayer(
            hidden_dim,
            edge_feature_dim,
        )
        self.predict_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        atom_features: torch.Tensor,  # (total_atoms, node_feature_dim)
        bond_index: torch.Tensor,  # (2, total_directed_edges)
        bond_features: torch.Tensor,  # (total_directed_edges, edge_feature_dim)
        graph_index: torch.Tensor,  # (total_atoms,), which graph each atom belongs to
        num_graphs: int,
    ) -> torch.Tensor:
        node_hidden = self.atom_embed(atom_features)

        for _ in range(self.num_steps):
            node_hidden = self.message_passing_layer(
                node_hidden,
                bond_index,
                bond_features,
            )

        # KEY SKILL: readout -- mean-pool every graph's own atoms into
        # one graph-level vector. Mean (not sum) so a 50-atom molecule's
        # vector isn't just mechanically larger than a 5-atom molecule's
        # purely from having more terms added together.
        hidden_dim = node_hidden.shape[-1]
        pooled_sum = torch.zeros(
            num_graphs,
            hidden_dim,
            device=node_hidden.device,
        )
        pooled_sum.index_add_(0, graph_index, node_hidden)

        atoms_per_graph = torch.zeros(
            num_graphs,
            device=node_hidden.device,
        )
        atoms_per_graph.index_add_(
            0,
            graph_index,
            torch.ones_like(graph_index, dtype=torch.float32),
        )
        pooled_mean = pooled_sum / atoms_per_graph.unsqueeze(-1)

        return self.predict_head(pooled_mean).squeeze(-1)  # (num_graphs,)
