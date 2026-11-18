"""
MLP training for Delaney ESOL solubility, using Weights & Biases
 
Usage:
    python main.py --optimizer adam --lr 0.01
    wandb sweep sweep.yaml          # prints a sweep id
    wandb agent <entity>/<project>/<sweep_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
from data_load_prep import load_dataset, make_loaders
from mlp import MLP
from train import evaluate, fit, get_device, load_checkpoint, predict, rmse_r2, set_seed

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"



def str2bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes")


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "sgd_momentum", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_batchnorm", type=str2bool, default=False)
    parser.add_argument("--dropout_p", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "cosine", "plateau"])
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    # W&B-specific flags
    parser.add_argument("--project", type=str, default="learn-bioml")
    parser.add_argument("--entity", type=str, default="tekim", help="Team/user namespace")
    parser.add_argument("--group", type=str, default=None, help="Optional group label, e.g. 'manual-optimizer-check'")
    parser.add_argument("--tags", type=str, nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = build_parser()
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # convert args into a dict 
    config = vars(args).copy()

    # remove the following from `config`
    for key in ("project", "entity", "group", "tags"):
        config.pop(key)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        config=config,
        group=args.group, # opt str labels to compare runs
        tags=args.tags, # opt str labels to compare runs
        job_type="train",
        save_code=True, # so the run can be reproduced later (along with args.seed)
    )

    set_seed(args.seed)
    device = get_device()
    print(f"Training device: {device}")
    wandb.config.update({"device": str(device)})

    data = load_dataset(seed=args.seed)
    train_loader, val_loader, test_loader = make_loaders(
        data, 
        batch_size=args.batch_size
    )
    print(f"Loaded {len(data.X_train)} train / {len(data.X_val)} val / {len(data.X_test)} test compounds")

    in_features = data.X_train.shape[1]
    model = MLP(
        in_features,
        tuple(args.hidden_sizes),
        use_batchnorm=args.use_batchnorm,
        dropout_p=args.dropout_p,
    )

    checkpoint_path = CHECKPOINT_DIR / f"{run.id}.pt"
    history = fit(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        optimizer_name=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        scheduler_name=None if args.scheduler == "none" else args.scheduler,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
    )
    print(
        f"Stopped at epoch {history['stopped_epoch']}, "
        f"best val MSE = {history['best_val_loss']:.4f} (epoch {history['best_epoch']})"
    )

    # Explicitly record the BEST val loss seen during training 
    wandb.summary["best_val_loss"] = history["best_val_loss"]
    wandb.summary["best_epoch"] = history["best_epoch"]

    # Reload the checkpointed (best-val-loss) weights for evaluation
    reload_model = MLP(
        in_features, 
        tuple(args.hidden_sizes),
        use_batchnorm=args.use_batchnorm, dropout_p=args.dropout_p,
    ).to(device)

    epoch, saved_val_loss = load_checkpoint(
        checkpoint_path, 
        reload_model
    )
    assert abs(saved_val_loss - evaluate(reload_model, val_loader, device)) < 1e-4

    artifact = wandb.Artifact(
        name=f"mlp-checkpoint-{run.id}",
        type="model",
        metadata={
            **config, 
            "best_val_loss": history["best_val_loss"], "best_epoch": history["best_epoch"]
        },
    )
    artifact.add_file(str(checkpoint_path))
    run.log_artifact(artifact)

    preds_std, targets_std = predict(
        reload_model, 
        test_loader, 
        device
    )
    mlp_preds = data.inverse_y(preds_std)
    measured = data.inverse_y(targets_std)
    baseline_preds = data.baseline_test

    mlp_rmse, mlp_r2 = rmse_r2(mlp_preds, measured)
    baseline_rmse, baseline_r2 = rmse_r2(baseline_preds, measured)

    print(f"MLP (tuned, from checkpoint): test RMSE = {mlp_rmse:.3f}, R^2 = {mlp_r2:.3f}")

    print(f"ESOL linear baseline:         test RMSE = {baseline_rmse:.3f}, R^2 = {baseline_r2:.3f}")

    wandb.log({"test_rmse": mlp_rmse, "test_r2": mlp_r2, "baseline_rmse": baseline_rmse, "baseline_r2": baseline_r2})

    
    # REUSABLE pattern: log raw per-example data as a Table whenever you
    # want it to be sliceable/filterable later, not just plotted once.
    table = wandb.Table(columns=["measured", "mlp_pred", "baseline_pred"])
    for m, p, b in zip(measured.tolist(), mlp_preds.tolist(), baseline_preds.tolist()):
        table.add_data(m, p, b)

    wandb.log(
        {
            "test/predictions": table,
            "test/mlp_vs_measured": wandb.plot.scatter(
                table, "measured", "mlp_pred", title="MLP: predicted vs. measured log solubility"
            ),
            "test/baseline_vs_measured": wandb.plot.scatter(
                table, "measured", "baseline_pred", title="ESOL baseline: predicted vs. measured log solubility"
            ),
        }
    )

    run.finish()
    print(f"\nView this run at: {run.url}")


if __name__ == "__main__":
    main()
