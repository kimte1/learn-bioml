"""
Load a human globin protein family from `data/uniprot_human_sprot.fasta`
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UNIPROT_FASTA = REPO_ROOT / "data" / "uniprot_human_sprot.fasta"

# accession -> gene name, real globin-fold proteins only
GLOBIN_FAMILY = {
    "P69905": "HBA",
    "P02008": "HBAZ",
    "P68871": "HBB",
    "P02042": "HBD",
    "P02100": "HBE",
    "P69891": "HBG1",
    "P69892": "HBG2",
    "P02144": "MYG",
    "Q8WWM9": "CYGB",
    "Q9NPG2": "NGB",
}


def _parse_fasta_by_accession(
    path: Path,
) -> dict[str, str]:
    """
    UniProt headers look like `>sp|ACCESSION|NAME_SPECIES description`.
    Returns {accession: sequence} for every entry in the file.
    """
    
    sequences: dict[str, str] = {}
    accession: str | None = None
    seq_chars: list[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if accession is not None:
                    sequences[accession] = "".join(seq_chars)
                accession = line.split("|")[1]
                seq_chars = []
            else:
                seq_chars.append(line)
    if accession is not None:
        sequences[accession] = "".join(seq_chars)

    return sequences


def load_globin_family(
    path: Path = UNIPROT_FASTA,
) -> list[tuple[str, str, str]]:
    """
    Returns a list of (gene_name, accession, sequence) tuples
    """
    all_sequences = _parse_fasta_by_accession(path)

    family = []
    for accession, gene_name in GLOBIN_FAMILY.items():
        sequence = all_sequences[accession]
        family.append(
            (
                gene_name, 
                accession, 
                sequence
            ),
        )

    return family
