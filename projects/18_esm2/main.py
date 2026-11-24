"""
- Load ESM-2 
- extract residue-level and sequence-level embeddings
- analyze representations for the human globin protein family in `data/uniprot_human_sprot.fasta`.
"""


from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import torch

from data_load_prep import load_globin_family
from embeddings import noncontextual_embeddings
from embeddings import residue_embeddings
from embeddings import sequence_embedding
from embeddings import token_strings
from plotting import plot_contextual_vs_noncontextual
from plotting import plot_family_representations
from plotting import plot_tokenization

FIG_DIR = Path(__file__).resolve().parent / "fig"



def demo_tokenization_and_context(
    gene_name: str,
    sequence: str,
) -> None:
    """
    Tokenize + contextual-vs-noncontextual on one representative sequence.
    """
    
    tokens = token_strings(sequence)
    
    plot_tokenization(
        tokens,
        FIG_DIR / "tokenization.png",
    )
    
    # pick the most frequent residue in this sequence so the
    # contextual-vs-noncontextual contrast has several occurrences to compare
    residue_counts = collections.Counter(sequence)
    
    target_residue, count = residue_counts.most_common(1)[0]
    
    occurrence_positions = [
        i 
        for i, token in enumerate(tokens) 
        if token == target_residue
    ]
    occurrence_labels = [f"pos {i}" for i in occurrence_positions]

    # embeddings after seq passes through encoder
    contextual = residue_embeddings(sequence) # shape (149, 320)
    # embeddings before seq passes through encoder
    noncontextual = noncontextual_embeddings(sequence) # shape (149, 320)
    
    contextual_occurrences = contextual[occurrence_positions].numpy()
    noncontextual_occurrences = noncontextual[occurrence_positions].numpy()

    plot_contextual_vs_noncontextual(
        contextual_occurrences,
        noncontextual_occurrences,
        occurrence_labels,
        target_residue,
        FIG_DIR / "contextual_vs_noncontextual.png",
    )

    # verify the contrast is real, not just visually suggestive
    noncontextual_identical = torch.allclose(
        noncontextual[occurrence_positions[0]],
        noncontextual[occurrence_positions[1]],
    )
    contextual_identical = torch.allclose(
        contextual[occurrence_positions[0]],
        contextual[occurrence_positions[1]],
    )
    assert noncontextual_identical, "non-contextual lookup should be identical across occurrences"
    assert not contextual_identical, "contextual embeddings should differ across occurrences"
    print(
        f"[contextual embeddings] residue '{target_residue}' occurs {count} times in {gene_name}; "
        "non-contextual vectors are identical every time, contextual vectors are not.",
    )


def demo_representation_probing(
    family: list[tuple[str, str, str]],
) -> None:
    """
    Computes one pooled sequence embedding for the protein family member
    
    Probes whether the representation space groups them sensibly.
    """
    gene_names = []
    sequence_vecs = []
    
    for gene_name, accession, sequence in family:
        vec = sequence_embedding(sequence)
        
        gene_names.append(gene_name)
        sequence_vecs.append(vec.numpy())

    sequence_vecs = np.stack(sequence_vecs)
    plot_family_representations(
        gene_names,
        sequence_vecs,
        FIG_DIR / "family_representations.png",
    )


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    print("Load ESM-2 (biohub/esm2_t6_8M_UR50D)")

    family = load_globin_family()

    # pick one protein
    representative_gene, _, representative_sequence = family[2] 
    
    # demo_tokenization_and_context(
    #     representative_gene,
    #     representative_sequence,
    # )
    demo_representation_probing(family)


if __name__ == "__main__":
    main()
