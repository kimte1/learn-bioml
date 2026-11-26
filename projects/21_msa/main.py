"""
- Get an MSA for TEM-1 beta-lactamase's Pfam family (PF13354, see `data_load_prep.py`)
- Compute per-column conservation and pairwise covariation
- Check conservation against real measured mutation fitness effects (ProteinGym's BLAT_ECOLX_Jacquier_2013
assay) 
- Implement (toy scale, untrained) the row/column attention operations MSA Transformer
(Rao et al. 2021) uses to read these same statistics directly out of an MSA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from conservation import column_entropy
from conservation import gap_fraction
from covariation import apc_correction
from covariation import pairwise_mutual_information
from data_load_prep import NUM_STATES
from data_load_prep import load_dms_scores_by_position
from data_load_prep import load_msa
from msa_attention import MSARowColumnBlock
from plotting import plot_attention_maps
from plotting import plot_conservation_track
from plotting import plot_conservation_vs_dms
from plotting import plot_covariation_heatmap
from plotting import plot_gap_occupancy

FIG_DIR = Path(__file__).resolve().parent / "fig"

ATTENTION_SEQ_SUBSET = 48  # attention maps are only readable/affordable at this scale
D_MODEL = 16
NHEAD = 2


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    torch.manual_seed(0)

    print("Loading MSA...")
    msa = load_msa()
    
    num_seqs, num_cols = msa.encoded.shape
    print(f"{num_seqs} sequences x {num_cols} columns, TEM-1 at row {msa.tem1_index}")

    plot_gap_occupancy(
        gap_fraction(msa.encoded), 
        FIG_DIR / "gap_occupancy.png"
    )

    print("Computing conservation...")
    entropy = column_entropy(msa.encoded)
    plot_conservation_track(
        msa.column_to_uniprot_position, 
        entropy, 
        FIG_DIR / "conservation_track.png"
    )

    print("Computing covariation...")
    mi = pairwise_mutual_information(msa.encoded)
    mi_apc = apc_correction(mi)
    plot_covariation_heatmap(
        mi_apc, 
        FIG_DIR / "covariation_heatmap.png"
    )

    print("Comparing conservation against measured DMS fitness effects...")
    dms_scores_by_position = load_dms_scores_by_position()
    plot_conservation_vs_dms(
        msa.column_to_uniprot_position,
        entropy,
        dms_scores_by_position,
        FIG_DIR / "conservation_vs_dms.png",
    )

    print("Running untrained MSA-Transformer-style row/column attention...")
    ids = torch.from_numpy(msa.encoded[:ATTENTION_SEQ_SUBSET].astype(np.int64))
    embedding = torch.nn.Embedding(NUM_STATES, D_MODEL)
    x = embedding(ids)  # (ATTENTION_SEQ_SUBSET, num_cols, D_MODEL)

    block = MSARowColumnBlock(d_model=D_MODEL, nhead=NHEAD)
    with torch.no_grad():
        _, row_weights, col_weights = block(x)

    conserved_col = int(np.nanargmin(entropy))
    variable_col = int(np.nanargmax(entropy))
    plot_attention_maps(
        row_weights.numpy(),
        col_weights.numpy(),
        conserved_col,
        variable_col,
        FIG_DIR / "attention_maps.png",
    )

    # print(f"Done. Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
