"""
Self-supervised pretraining means learning a reusable representation
from large amounts of UNLABELED data, by constructing a free supervisory
signal out of the data itself -- here, hiding a random slice of a
sequence's own residues and training the model to predict them back from
context. 

No human ever labeled anything, which is exactly why this kind
of pretraining can use far more data than any task in this curriculum so
far that needed real labels

THE SPECIFIC TASK HERE: masked amino-acid prediction (BERT-style, Devlin
et al. 2019) over a toy-scale subset of real human SwissProt sequences
(`data/uniprot_human_sprot.fasta`) -- a small, from-scratch, laptop-scale
rehearsal of how a real protein language model like ESM-2 (`18_esm2`/
`19_esm2_finetune`) was pretrained at a much larger scale on much more
data and compute.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from data_load_prep import PAD_IDX
from data_load_prep import VOCAB_SIZE
from data_load_prep import load_sequences
from masking import mask_batch
from model import BertStyleEncoder
from plotting import plot_fill_in_the_blank
from plotting import plot_masking_sanity_check
from plotting import plot_pretraining_curves
from train import evaluate
from train import fit
from train import get_device
from train import set_seed
from train import tokenize_batch

FIG_DIR = Path(__file__).resolve().parent / "fig"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "bert_pretrain.pt"


def build_model() -> BertStyleEncoder:
    # bumped up from the original toy size (d_model=128, 3 layers,
    # ~403K params) after the first run plateaued at ~19% masked-token
    # accuracy with the model mostly predicting the single most common
    # amino acid everywhere -- a sign the bottleneck was capacity/data,
    # not training time (loss/accuracy had already flattened by ~epoch 5)
    return BertStyleEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        pad_idx=PAD_IDX,
    )


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)

    set_seed(0)
    device = get_device()
    print(f"device: {device}")

    # also loosened from the first run's (max_len=200, n_sequences=4000)
    # -- that config only had 3,654 sequences available after filtering,
    # and self-supervised pretraining scales with DATA more than almost
    # anything else, since there's no labeling cost to using more of it.
    # max_len=300 alone raises the qualifying pool from 3,654 to 6,470
    # (checked directly against the fasta); n_sequences=20000 is
    # intentionally higher than that pool so every qualifying sequence
    # gets used, same as `min(n_sequences, len(filtered))` already
    # guards for in `load_sequences`. (max_len=400 would reach 9,819
    # sequences, but at a noticeably higher per-step cost on MPS since
    # attention cost grows with the square of sequence length -- 300 is
    # the chosen balance for a still-toy-scale run.)
    dataset = load_sequences(
        max_len=300,
        n_sequences=20000,
    )

    print(
        f"train/val sizes: {len(dataset.train_sequences)}/{len(dataset.val_sequences)}",
    )

    model = build_model()

    # numel = number of elements
    n_params = sum(p.numel() for p in model.parameters())

    print(f"model parameters: {n_params:,}")

    # batch_size dropped from 32 -> 16 since sequences are now up to
    # 400 residues (vs. 200 before) -- attention cost grows with the
    # square of sequence length, so a smaller batch keeps per-step
    # memory/time reasonable on MPS
    history = fit(
        model,
        device,
        dataset.train_sequences,
        dataset.val_sequences,
        epochs=15,
        batch_size=16,
        mask_prob=0.15,
        lr=3e-4,
        checkpoint_path=CHECKPOINT_PATH,
    )

    plot_pretraining_curves(
        history,
        FIG_DIR / "pretraining_curves.png",
    )

    # KEY SKILL: masking strategies (sanity check) -- confirm the 80/10/10
    # split actually realized by `mask_batch` on a real batch of validation
    # sequences matches what it's supposed to be
    sample_ids, sample_key_padding_mask = tokenize_batch(dataset.val_sequences[:64])
    sample_generator = torch.Generator()
    sample_generator.manual_seed(0)
    sample_corrupted_ids, sample_labels = mask_batch(
        sample_ids,
        sample_key_padding_mask,
        mask_prob=0.15,
        generator=sample_generator,
    )
    plot_masking_sanity_check(
        sample_ids,
        sample_corrupted_ids,
        sample_labels,
        FIG_DIR / "masking_sanity_check.png",
    )

    # KEY SKILL: checkpoint management -- reload the saved BEST checkpoint
    # into a brand-new model instance (not the one that just finished
    # training) and confirm it reproduces the exact val_loss recorded at
    # save time. This proves the checkpoint actually round-trips --
    # weights, architecture, everything -- not merely that a file got
    # written to disk.
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    reloaded_model = build_model()
    reloaded_model.load_state_dict(checkpoint["model_state"])
    reloaded_model.to(device)

    # batch_size here MUST match the batch_size `fit()` used for its own
    # validation calls (also via `evaluate`, seed=0) -- `mask_batch`'s
    # random draws are consumed sequentially from the generator as
    # `_iterate_batches` groups sequences into batches, so a different
    # batch_size changes which positions get selected/corrupted even
    # with the same seed, which would break the round-trip comparison
    # below for a reason that has nothing to do with the checkpoint
    reloaded_val_loss, reloaded_val_acc = evaluate(
        reloaded_model,
        dataset.val_sequences,
        device,
        16,
        0.15,
        seed=0,
    )
    print(
        f"checkpoint round-trip check: saved val_loss={checkpoint['val_loss']:.6f} "
        f"(epoch {checkpoint['epoch'] + 1}), reloaded val_loss={reloaded_val_loss:.6f}, "
        f"reloaded val_masked_acc={reloaded_val_acc:.3f}",
    )
    assert math.isclose(checkpoint["val_loss"], reloaded_val_loss, rel_tol=1e-4), (
        "reloaded checkpoint did not reproduce the saved val_loss -- the "
        "save/load round-trip is broken"
    )

    # the qualitative demo: does the reloaded, pretrained model actually
    # fill in hidden residues sensibly?
    plot_fill_in_the_blank(
        dataset.val_sequences,
        reloaded_model,
        device,
        FIG_DIR / "fill_in_the_blank.png",
    )

    print(f"\nFigures saved to {FIG_DIR}")
    print(f"Best checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
