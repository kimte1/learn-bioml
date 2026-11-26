"""

THE SPECIFIC TASK HERE: construct representations (atomic, residue-level, distance) for 
the molecule (aspirin) and a protein structure (1CRN, Crambin)

and compare their node/edge features directly: 
    - how many nodes and edges each has, 
    - what dimensionality their features are, 
    - which of them can carry genuine GEOMETRIC edge features (distance,
direction) at all, and which structurally cannot.
"""

from __future__ import annotations

from pathlib import Path

from comparison import summarize_graphs
from molecular_graphs import build_molecular_graph
from plotting import plot_adjacency_comparison
from plotting import plot_comparison_table
from plotting import plot_degree_distributions
from plotting import plot_molecule
from protein_graphs import build_atomic_graph
from protein_graphs import build_residue_graph
from protein_graphs import parse_structure

FIG_DIR = Path(__file__).resolve().parent / "fig"

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    print("=== molecular graph (aspirin) ===")
    molecule_minimal = build_molecular_graph(
        "aspirin (minimal node features)",
        ASPIRIN_SMILES,
        feature_scheme="minimal",
    )
    molecule_rich = build_molecular_graph(
        "aspirin (rich node features)",
        ASPIRIN_SMILES,
        feature_scheme="rich",
    )
    print(
        f"  {molecule_minimal.node_features.shape[0]} atoms, "
        f"{molecule_rich.edge_index.shape[1] // 2} bonds",
    )
    plot_molecule(
        ASPIRIN_SMILES,
        "Aspirin",
        FIG_DIR / "aspirin.png",
    )

    print("\n=== protein graphs (1CRN, Crambin) ===")
    structure = parse_structure("1CRN")
    residue_sequence_graph = build_residue_graph(
        structure,
        edge_scheme="sequence",
    )
    residue_distance_graph = build_residue_graph(
        structure,
        edge_scheme="distance",
    )
    atomic_graph = build_atomic_graph(structure)

    disulfide_bonds = sum(
        1
        for s, d in zip(*atomic_graph.edge_index)
        if atomic_graph.node_labels[s] == "S"
        and atomic_graph.node_labels[d] == "S"
        and s < d
    )
    print(
        f"  {len(structure.residues)} residues, "
        f"{atomic_graph.node_features.shape[0]} atoms\n"
        f"  distance-based atomic graph found {disulfide_bonds} S-S bonds "
        "(Crambin has 3 real disulfide bridges)",
    )

    plot_adjacency_comparison(
        [residue_sequence_graph, residue_distance_graph],
        FIG_DIR / "residue_adjacency_comparison.png",
    )

    print("\n=== comparing every representation ===")
    summary = summarize_graphs(
        [
            molecule_minimal,
            molecule_rich,
            residue_sequence_graph,
            residue_distance_graph,
            atomic_graph,
        ],
    )
    print(summary.to_string(index=False))
    plot_comparison_table(
        summary,
        FIG_DIR / "comparison_table.png",
    )

    plot_degree_distributions(
        [molecule_rich, residue_sequence_graph, residue_distance_graph, atomic_graph],
        FIG_DIR / "degree_distributions.png",
    )



if __name__ == "__main__":
    main()
