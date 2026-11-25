"""
Fine-tuning means taking that already-pretrained (generic) model and
continuing to train it (fully, or only a small part of it) on a much
smaller, task-specific labeled dataset, so it adapts what it already
knows to the one thing you actually care about. 

Then, I don't need to train a new model from random initialization for that task, 
which would need far more labeled data than most real tasks have available.

THE SPECIFIC TASK HERE: 
    - predict a continuous fitness score (`DMS_score`)
for a single point-mutant of TEM-1 beta-lactamase; 
i.e. regression from a mutant protein sequence to a
number. 

The 989 (sequence, score) pairs come from one real ProteinGym
Deep Mutational Scanning assay (`BLAT_ECOLX_Jacquier_2013`) -- see
`data_load_prep.py`.

This project trains the SAME task 4 different ways ('models.py'):
- linear probing,
- frozen-backbone training
- LoRA, 

to compare how much of the backbone actually needs to change to do well.
"""

from __future__ import annotations

from pathlib import Path

from data_load_prep import load_fitness_dataset
from models import build_frozen_backbone
from models import build_full_finetune
from models import build_linear_probe
from models import build_lora
from plotting import plot_before_after_finetuning
from plotting import plot_convergence
from plotting import plot_trainable_params
from train import evaluate
from train import fit
from train import get_device
from train import predict
from train import set_seed

FIG_DIR = Path(
    __file__,
).resolve().parent / "fig"


# how much of the pretrained ESM-2 backbone do i let training touch
STRATEGY_BUILDERS = {
    "linear_probe": build_linear_probe, # freeze entire backbone, and train only a single linear layer on top of the pooled embedding
    "frozen_backbone": build_frozen_backbone, # freeze entire backbone, but gets a small head: linear -> ReLU -> linear
    "lora": build_lora, # freeze entire backbone, but a small trainable adapter inside the backbone (into the attention layer; a low-rank correction)
    "full_finetune": build_full_finetune, # fine-tune the backbone and head
}

LEARNING_RATES = {
    "linear_probe": 1e-3,
    "frozen_backbone": 1e-3,
    "lora": 1e-3,
    "full_finetune": 2e-5,
}


def run_strategy(
    strategy_name: str,
    dataset,
    device,
) -> dict:
    """
    Builds one strategy's model, captures its test-set predictions
    before any training happens, trains it, then captures predictions and
    the final Spearman correlation after training.
    """
    print(
        f"\n=== {strategy_name} ===",
    )
    build_fn = STRATEGY_BUILDERS[strategy_name]
    
    model, n_trainable = build_fn()
    print(f"trainable parameters: {n_trainable:,}")

    model.to(device)

    preds_before = predict(
        model,
        dataset.test_sequences,
        device,
    )
    _, spearman_before = evaluate(
        model,
        dataset.test_sequences,
        dataset.test_scores,
        device,
    )
    print(
        f"before training: test Spearman r = {spearman_before:.3f} (should be close to 0)",
    )

    history = fit(
        model,
        device,
        dataset.train_sequences,
        dataset.train_scores,
        dataset.val_sequences,
        dataset.val_scores,
        epochs=10,
        batch_size=16,
        lr=LEARNING_RATES[strategy_name],
    )

    preds_after = predict(
        model,
        dataset.test_sequences,
        device,
    )
    test_mse, spearman_after = evaluate(
        model,
        dataset.test_sequences,
        dataset.test_scores,
        device,
    )
    print(
        f"after training:  test Spearman r = {spearman_after:.3f}, test MSE = {test_mse:.4f}",
    )

    return {
        "n_trainable": n_trainable,
        "history": history,
        "preds_before": preds_before,
        "preds_after": preds_after,
        "spearman_before": spearman_before,
        "spearman_after": spearman_after,
    }


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    set_seed(0)
    device = get_device()

    dataset = load_fitness_dataset()

    print(
        f"train/val/test sizes: {len(dataset.train_sequences)}/"
        f"{len(dataset.val_sequences)}/{len(dataset.test_sequences)}",
    )

    results = {}
    for strategy_name in STRATEGY_BUILDERS:
        results[strategy_name] = run_strategy(
            strategy_name,
            dataset,
            device,
        )

    plot_convergence(
        {name: r["history"] for name, r in results.items()},
        FIG_DIR / "convergence.png",
    )
    plot_trainable_params(
        {name: r["n_trainable"] for name, r in results.items()},
        FIG_DIR / "trainable_params.png",
    )
    plot_before_after_finetuning(
        dataset.test_scores,
        {name: r["preds_before"] for name, r in results.items()},
        {name: r["preds_after"] for name, r in results.items()},
        {name: r["spearman_before"] for name, r in results.items()},
        {name: r["spearman_after"] for name, r in results.items()},
        FIG_DIR / "before_after_finetuning.png",
    )

    print(
        "\n=== summary (test Spearman r, after training) ===",
    )
    for name, r in results.items():
        print(
            f"  {name:16s} trainable={r['n_trainable']:>9,}  spearman={r['spearman_after']:.3f}",
        )


if __name__ == "__main__":
    main()
