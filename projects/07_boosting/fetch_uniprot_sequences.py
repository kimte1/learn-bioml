"""
- Fetch the reviewed (Swiss-Prot) human proteome from UniProt, 
- join each Meltome Atlas (Jarzab et al., 2020) measurement in data/meltome.csv.zip to its gene' sequence, 
- save the result as data/meltome_with_sequences.csv.zip.

- Run this once (or whenever you want to rebuild the dataset)
"""

from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_FASTA_CACHE = REPO_ROOT / "data" / "uniprot_human_sprot.fasta"
MELTOME_CSV_ZIP = REPO_ROOT / "data" / "meltome.csv.zip"
OUTPUT_CSV_ZIP = REPO_ROOT / "data" / "meltome_with_sequences.csv.zip"
OUTPUT_CSV_NAME = "meltome_with_sequences.csv"

UNIPROT_STREAM_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=organism_id%3A9606+AND+reviewed%3Atrue"
    "&format=fasta"
)
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def fetch_human_swissprot_fasta(
    cache_path: Path = RAW_FASTA_CACHE,
    force: bool = False,
    timeout: float = 180.0,
) -> Path:
    """Download the full reviewed human proteome in one bulk request and
    cache it at cache_path. A no-op once cached, so re-running this
    script to pick up filter changes doesn't re-hit the network.
    """
    if cache_path.exists() and not force:
        return cache_path

    request = urllib.request.Request(UNIPROT_STREAM_URL, headers={"User-Agent": "learn-bioml/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return cache_path


def parse_gene_to_sequence(fasta_path: Path) -> dict[str, str]:
    """Parse a UniProt FASTA (header `GN=<gene symbol>`) into
    {gene symbol: sequence}. A gene symbol can appear on more than one
    entry (isoforms, readthrough transcripts); we keep the longest
    sequence for it, i.e. the canonical/full-length form.
    """
    gene_to_seq: dict[str, str] = {}
    gene: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if gene is None:
            return
        seq = "".join(chunks)
        if gene not in gene_to_seq or len(seq) > len(gene_to_seq[gene]):
            gene_to_seq[gene] = seq

    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                flush()
                match = re.search(r"GN=(\S+)", line)
                gene = match.group(1).upper() if match else None
                chunks = []
            else:
                chunks.append(line.strip())
    flush()
    return gene_to_seq


def build_meltome_with_sequences(
    min_length: int = 30,
    max_length: int = 2000,
    force_refetch: bool = False,
) -> pd.DataFrame:
    """Join every meltome.csv row to its gene's UniProt sequence, dropping
    rows with no sequence match (gene-symbol aliases/mismatches between
    the two sources) or a non-standard-amino-acid / out-of-range sequence.
    """
    fasta_path = fetch_human_swissprot_fasta(force=force_refetch)
    gene_to_seq = parse_gene_to_sequence(fasta_path)

    # the uploaded zip also carries a macOS "__MACOSX/._meltome.csv"
    # resource-fork entry; read the real CSV member by name.
    with zipfile.ZipFile(MELTOME_CSV_ZIP) as zf:
        with zf.open("meltome.csv") as f:
            df = pd.read_csv(f)
    df["gene_name"] = df["gene_name"].str.upper()
    df["sequence"] = df["gene_name"].map(gene_to_seq)

    has_sequence = df["sequence"].notna()
    valid_length = df["sequence"].str.len().between(min_length, max_length)
    valid_alphabet = df["sequence"].apply(lambda s: isinstance(s, str) and set(s) <= STANDARD_AA)

    return df[has_sequence & valid_length & valid_alphabet].reset_index(drop=True)


def save_meltome_with_sequences(df: pd.DataFrame, path: Path = OUTPUT_CSV_ZIP) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression={"method": "zip", "archive_name": OUTPUT_CSV_NAME})


if __name__ == "__main__":
    merged = build_meltome_with_sequences()
    save_meltome_with_sequences(merged)
    print(
        f"Matched sequences for {merged['gene_name'].nunique()} genes "
        f"({len(merged)} measurement rows); saved to {OUTPUT_CSV_ZIP}"
    )
