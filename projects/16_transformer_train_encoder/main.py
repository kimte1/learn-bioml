"""
Train a Transformer encoder classifier (PyTorch's built-in
`nn.TransformerEncoder) DeepLoc subcellular-localization labels, with gradient accumulation; evaluate
with scikit-learn; then separately profile forward+backward time/memory
across sequence length and batch size.

Usage:
    python main.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from data_load_prep import PAD_IDX, VOCAB_SIZE, load_dataset, tokenize_batch
from model import TransformerClassifier
from plotting import plot_confusion_matrix, plot_convergence, plot_profiling
from profiling import sweep
from train import fit, get_device, predict, set_seed

FIG_DIR = Path(__file__).resolve().parent / "fig"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def verify_padding_mask_blocks_gradient(
    model: torch.nn.Module, 
    device: torch.device
) -> None:
    """A batch with real padding: gradient on the PAD embedding row must
    stay exactly zero (nn.Embedding's padding_idx guarantees this), and
    no NaNs/infs should appear from attending to padded positions.
    """
    model.to(device)
    model.zero_grad(set_to_none=True)
    sequences = ["ACDEFGHIKL", "ACDEFG"]  # deliberately different lengths -> real padding
    ids, key_padding_mask = tokenize_batch(sequences)
    ids, key_padding_mask = ids.to(device), key_padding_mask.to(device)

    logits = model(ids, key_padding_mask)
    assert torch.isfinite(logits).all()
    logits.sum().backward()

    pad_grad = model.embedding.weight.grad[PAD_IDX]
    assert torch.all(pad_grad == 0.0), pad_grad.abs().max().item()
    model.zero_grad(set_to_none=True)
    print("[padding mask] gradient on the PAD embedding row is exactly zero; no NaNs from padded attention.")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    set_seed(0)
    device = get_device()
    print(f"device: {device}")

    ds = load_dataset(seed=0)
    
    num_classes = len(ds.label_encoder.classes_)
    
    print(f"train/val/test sizes: {len(ds.train_sequences)}/{len(ds.val_sequences)}/{len(ds.test_sequences)}")
    print(f"classes: {list(ds.label_encoder.classes_)}")

    model = TransformerClassifier(
        VOCAB_SIZE, 
        num_classes=num_classes, 
        pad_idx=PAD_IDX
    )
    verify_padding_mask_blocks_gradient(model, device)

    history = fit(
        model,
        device,
        ds.train_sequences,
        ds.train_labels,
        ds.val_sequences,
        ds.val_labels,
        epochs=8,
        batch_size=16,
        accumulation_steps=4,  # effective batch size 64, 4 micro-batches at a time
        lr=3e-4,
        checkpoint_path=CHECKPOINT_DIR / "best.pt",
    )
    plot_convergence(history, FIG_DIR / "convergence.png")

    test_preds = predict(
        model, 
        ds.test_sequences, 
        device
    )
    print(classification_report(
        ds.test_labels.numpy(), 
        test_preds.numpy(), 
        target_names=ds.label_encoder.classes_, 
        zero_division=0
    ))

    cm = confusion_matrix(ds.test_labels.numpy(), test_preds.numpy())
    plot_confusion_matrix(
        cm, 
        list(ds.label_encoder.classes_),
        FIG_DIR / "confusion_matrix.png"
    )

    print("\n[profiling] sequence-length x batch-size sweep (dense, unpadded dummy batches):")
    results = sweep(
        model, 
        device, 
        seq_lens=[32, 64, 128, 256, 512], 
        batch_sizes=[8, 16, 32, 64]
    )
    plot_profiling(results, FIG_DIR / "profiling.png")



if __name__ == "__main__":
    main()
