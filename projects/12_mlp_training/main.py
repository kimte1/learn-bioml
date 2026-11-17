"""
A more rigorous training pipeline and controlled ablations for MLP using the ESOL dataset trained to predict aqueous solubility.

Each ablation below holds everything fixed except the one knob under
test, and carries its winner forward into the next ablation, so the
final model reflects every choice made along the way:

    optimizer (SGD / SGD+momentum / Adam / AdamW)
        -> weight decay sweep
            -> dropout x batch-norm ablation
                -> learning-rate schedule (none / step / cosine / plateau)

Gradient clipping is demonstrated separately, at a deliberately high
learning rate, since its point is to show a failure mode rather than to
be folded into the "pick the best" chain. The final run trains the
winning configuration with early stopping and checkpointing under a
fixed seed, then reloads the checkpoint to confirm it reproduces the
saved validation loss before evaluating on the held-out test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data_load_prep import load_dataset, make_loaders
from mlp import MLP
from plotting import (
    plot_final_predictions,
    plot_grad_clip_comparison,
    plot_lr_schedule_comparison,
    plot_optimizer_comparison,
    plot_regularization_ablation,
    plot_weight_decay_sweep,
)
from train import evaluate, fit, get_device, load_checkpoint, predict, rmse_r2, set_seed

FIG_DIR = Path(__file__).resolve().parent / "fig"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

OPTIMIZER_NAMES = ["sgd", "sgd_momentum", "adam", "adamw"]
WEIGHT_DECAYS = [0.0, 1e-4, 1e-3, 1e-2]
SCHEDULE_NAMES = [None, "step", "cosine", "plateau"]
DROPOUT_P = 0.2


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hidden-sizes", 
        type=int, 
        nargs="+", 
        default=[64, 32], 
        help="Hidden layer widths"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Minibatch size"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100, 
        help="Epochs for each ablation run"
    )
    parser.add_argument(
        "--final-epochs", 
        type=int, 
        default=300, 
        help="Max epochs for the final run (early stopping may cut it short)"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=20, 
        help="Early-stopping patience (epochs) for the final run"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-2, 
        help="Learning rate for Adam/AdamW"
    )
    parser.add_argument(
        "--sgd-lr",
        type=float,
        default=1e-3,
        help="Learning rate for plain SGD / SGD+momentum -- deliberately smaller than --lr, "
        "since SGD has no per-parameter adaptive step size and diverges at Adam-scale learning "
        "rates on this architecture; using the same lr for both would make the comparison an "
        "artifact of a mismatched lr rather than the optimizers themselves",
    )
    parser.add_argument(
        "--clip-demo-lr",
        type=float,
        default=5e-3,
        help="Deliberately high lr (for SGD+momentum) used to demonstrate gradient clipping -- "
        "high enough that an early large gradient sends unclipped loss to ~1e15 within the first "
        "epoch, from which the unclipped run never recovers a useful loss",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="Seed for reproducible data splits and training runs"
    )
    return parser.parse_args()


def main() -> None:
    args = build_parser()
    FIG_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f"Training device: {device}")

    data = load_dataset(seed=args.seed)
    train_loader, val_loader, test_loader = make_loaders(data, batch_size=args.batch_size)
    print(
        f"Loaded {len(data.X_train)} train",
        f"{len(data.X_val)} val",
        f"{len(data.X_test)} test compounds",
        sep='\n'
    )

    in_features = data.X_train.shape[1]
    hidden_sizes = tuple(args.hidden_sizes)

    def new_model(
        use_batchnorm: bool = False, 
        dropout_p: float = 0.0
    ) -> MLP:
        # Re-seed immediately before construction so every model across
        # every ablation branch starts from identical initial weights --
    
        set_seed(args.seed)
        return MLP(
            in_features, 
            hidden_sizes, 
            use_batchnorm=use_batchnorm, 
            dropout_p=dropout_p
        )

    ### Sequential hyperparm tuning
    ### Establish a baseline
    ### then optimize one component
    ### carry the best config to the next experiment
    
    # 1. Optimizer ablation: SGD / SGD+momentum / Adam / AdamW
    print("\n=== Optimizer ablation ===")
    optimizer_lrs = {
        "sgd": args.sgd_lr, 
        "sgd_momentum": args.sgd_lr, 
        "adam": args.lr, 
        "adamw": args.lr
    }

    optimizer_histories = {}
    for name in OPTIMIZER_NAMES:
        history = fit(
            new_model(), 
            train_loader, 
            val_loader, 
            device,
            epochs=args.epochs, 
            optimizer_name=name, 
            lr=optimizer_lrs[name], 
            seed=args.seed,
        )
        optimizer_histories[name] = history
        print(
            f"optimizer={name:<12} lr={optimizer_lrs[name]:<8}",
            f"best val MSE = {history['best_val_loss']:.4f}",
            f"epoch {history['best_epoch']}",
            sep="\n"
        )

    best_optimizer = min(
        optimizer_histories, 
        key=lambda n: optimizer_histories[n]["best_val_loss"]
    )

    best_lr = optimizer_lrs[best_optimizer]
    print(f"Best optimizer: {best_optimizer} (lr={best_lr})")
    plot_optimizer_comparison(
        optimizer_histories, 
        FIG_DIR / "01_optimizer_comparison.png"
    )

    ### 2. Weight-decay sweep, using the winning optimizer
    print("\n=== Weight-decay sweep ===")
    wd_histories = {}
    for wd in WEIGHT_DECAYS:
        history = fit(
            new_model(), 
            train_loader, 
            val_loader, 
            device,
            epochs=args.epochs, 
            optimizer_name=best_optimizer, 
            lr=best_lr, 
            weight_decay=wd, 
            seed=args.seed,
        )
        wd_histories[str(wd)] = history
        print(f"weight_decay={wd:<8} best val MSE = {history['best_val_loss']:.4f}")

    best_wd = float(min(wd_histories, key=lambda k: wd_histories[k]["best_val_loss"]))
    print(f"Best weight decay: {best_wd}")
    plot_weight_decay_sweep(
        wd_histories, 
        FIG_DIR / "02_weight_decay_sweep.png"
    )

    ### 4. Learning-rate schedule ablation, using the winning config so far
    print("\n=== Learning-rate schedule ablation ===")
    sched_histories = {}
    for sched in SCHEDULE_NAMES:
        label = sched or "constant"
        history = fit(
            new_model(
                use_batchnorm=best_use_bn, 
                dropout_p=DROPOUT_P if best_use_do else 0.0
            ),
            train_loader, val_loader, device,
            epochs=args.epochs, 
            optimizer_name=best_optimizer, 
            lr=best_lr, 
            weight_decay=best_wd,
            scheduler_name=sched, 
            seed=args.seed,
        )
        sched_histories[label] = history
        print(f"schedule={label:<10} best val MSE = {history['best_val_loss']:.4f}")

    best_sched_label = min(sched_histories, key=lambda n: sched_histories[n]["best_val_loss"])
    best_scheduler = None if best_sched_label == "constant" else best_sched_label
    print(f"Best LR schedule: {best_sched_label}")
    plot_lr_schedule_comparison(
        sched_histories, 
        FIG_DIR / "04_lr_schedule_comparison.png"
    )

    ### 5. Gradient-clipping ablation: independent demo at a deliberately high lr
    print("\n=== Gradient-clipping ablation (high-lr demo) ===")
    clip_histories = {}
    for label, clip_norm in [("unclipped", None), ("clipped", 1.0)]:
        history = fit(
            new_model(), 
            train_loader, 
            val_loader, 
            device,
            epochs=args.epochs, 
            optimizer_name="sgd_momentum", 
            lr=args.clip_demo_lr,
            grad_clip_norm=clip_norm, 
            seed=args.seed,
        )
        clip_histories[label] = history
        print(
            f"{label:<10} max train MSE = {max(history['train_loss']):.3g}, "
            f"final train MSE = {history['train_loss'][-1]:.4g}"
        )
    plot_grad_clip_comparison(
        clip_histories, 
        FIG_DIR / "05_grad_clip_comparison.png"
    )

    ### 6. Final run: winning config, early stopping + checkpointing + reproducible seeding
    print("\n=== Final tuned model (early stopping + checkpointing) ===")
    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
    final_model = new_model(
        use_batchnorm=best_use_bn, 
        dropout_p=DROPOUT_P if best_use_do else 0.0
    )

    final_history = fit(
        final_model, 
        train_loader, 
        val_loader, 
        device,
        epochs=args.final_epochs, 
        optimizer_name=best_optimizer, 
        lr=best_lr, 
        weight_decay=best_wd,
        scheduler_name=best_scheduler, 
        grad_clip_norm=1.0, 
        patience=args.patience, # if val loss doesn't improve by n_patience epochs, stop
        checkpoint_path=checkpoint_path, 
        seed=args.seed,
    )
    print(
        f"Stopped at epoch {final_history['stopped_epoch']}, "
        f"best val MSE = {final_history['best_val_loss']:.4f} (epoch {final_history['best_epoch']})"
    )

    # Checkpoint correctness check: reload into a fresh model and confirm
    # it reproduces the val loss it was checkpointed with.
    reload_model = new_model(
        use_batchnorm=best_use_bn, 
        dropout_p=DROPOUT_P if best_use_do else 0.0
    ).to(device)

    epoch, saved_val_loss = load_checkpoint(checkpoint_path, reload_model)
    recomputed_val_loss = evaluate(reload_model, val_loader, device)

    print(f"Checkpoint reload: epoch {epoch}, saved val MSE = {saved_val_loss:.4f}, recomputed val MSE = {recomputed_val_loss:.4f}")

    assert abs(saved_val_loss - recomputed_val_loss) < 1e-4, "checkpoint reload did not reproduce the saved val loss"

    print("Checkpoint reload OK -- matches saved val loss.")

    preds_std, targets_std = predict(reload_model, test_loader, device)
    mlp_preds = data.inverse_y(preds_std)
    measured = data.inverse_y(targets_std)
    baseline_preds = data.baseline_test

    mlp_rmse, mlp_r2 = rmse_r2(mlp_preds, measured)
    baseline_rmse, baseline_r2 = rmse_r2(baseline_preds, measured)
    print(f"MLP (tuned, from checkpoint): test RMSE = {mlp_rmse:.3f}, R^2 = {mlp_r2:.3f}")
    print(f"ESOL linear baseline:         test RMSE = {baseline_rmse:.3f}, R^2 = {baseline_r2:.3f}")

    plot_final_predictions(
        mlp_preds, 
        measured, 
        baseline_preds, 
        mlp_rmse, 
        baseline_rmse, 
        FIG_DIR / "06_final_predictions.png"
    )


if __name__ == "__main__":
    main()
