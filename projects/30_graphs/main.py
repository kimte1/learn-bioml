"""
MPNN (Gilmer et al. 2017) learns a representation of a graph: 
    - every atom starts with its own local features, 
    - then repeatedly exchanges "messages" with its bonded neighbors 
    - and updates its own state based on what it hears,

    - so after enough rounds, every atom's state reflects not just itself but
its whole local neighborhood 

THE SPECIFIC TASK HERE: predict the Delaney (2004) ESOL aqueous
solubility target starting from raw SMILES via an MPNN that has
to build its own molecular representation from atoms and bonds alone.
"""

from __future__ import annotations

from pathlib import Path

from batching import collate_molecule_graphs
from data_load_prep import EDGE_FEATURE_DIM
from data_load_prep import NODE_FEATURE_DIM
from data_load_prep import load_molecule_graphs
from data_load_prep import split_molecule_graphs
from mpnn import MPNN
from plotting import plot_batch_structure
from plotting import plot_molecule_examples
from plotting import plot_parity
from plotting import plot_training_curves
from train import fit
from train import get_device
from train import predict
from train import set_seed

FIG_DIR = Path(__file__).resolve().parent / "fig"


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    set_seed(0)
    device = get_device()
    print(f"Using device: {device}")

    graphs = load_molecule_graphs()
    
    train_graphs, val_graphs, test_graphs = split_molecule_graphs(graphs)
    print(
        f"{len(graphs)} molecules "
        f"(train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)})",
    )

    plot_molecule_examples(
        graphs,
        FIG_DIR / "molecule_examples.png",
    )

    example_batch = collate_molecule_graphs(train_graphs[:12])
    plot_batch_structure(
        example_batch,
        FIG_DIR / "batch_structure.png",
    )

    model = MPNN(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
    ).to(device)

    history = fit(
        model,
        train_graphs,
        val_graphs,
        device,
        epochs=100,
    )
    plot_training_curves(
        history,
        FIG_DIR / "training_curves.png",
    )

    test_predictions, test_targets = predict(
        model,
        test_graphs,
        device,
    )
    plot_parity(
        test_predictions,
        test_targets,
        "Test set: MPNN solubility prediction",
        FIG_DIR / "test_parity.png",
    )



if __name__ == "__main__":
    main()
