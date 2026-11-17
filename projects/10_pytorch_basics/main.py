"""
Linear regression reimplemented in PyTorch to predict aqueous solubility from the Delaney ESOL dataset

- scratch_linear_regression.py -- raw tensors, manual forward pass, manual MSE, loss.backward(), and a hand-written parameter update.

- torch_linear_regression.py   -- nn.Linear, nn.MSELoss, and
       torch.optim.SGD doing the same job.

- Both use the exact same data, learning rate, loss reduction, and epochs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from numpy.random.mtrand import f
import torch

from data_load_prep import load_dataset, make_loaders
from plotting import plot_loss_curves, plot_predictions, plot_weight_comparison
from scratch_linear_regression import ScratchLinearRegression, fit_scratch
from torch_linear_regression import TorchLinearRegression, fit_torch
from train import get_device, predict, rmse_r2

FIG_DIR = Path(__file__).resolve().parent / "fig"


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Minibatch size for both models"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=150, 
        help="Training epochs for both models"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-2, 
        help="SGD learning rate, shared by both models"
    )
    return parser.parse_args()



def main() -> None:
    args = build_parser()
    FIG_DIR.mkdir(exist_ok=True)

    device = get_device()
    print(f"Training device: {device}")

    rng_seed = 0
    data = load_dataset(seed=rng_seed)
    train_loader, val_loader, test_loader = make_loaders(data, batch_size=args.batch_size)

    print(
        f"Loaded {len(data.X_train)} train",
        f"{len(data.X_val)} val",
        f"{len(data.X_test)} test compounds",
        sep="\n"
    )

    in_features = data.X_train.shape[1]
    
    

    ### Train linear reg from scratch
    torch.manual_seed(rng_seed)
    scratch_model = ScratchLinearRegression(in_features)
    scratch_history = fit_scratch(
        scratch_model, 
        train_loader, 
        val_loader, 
        device, 
        epochs=args.epochs, 
        lr=args.lr,
    )
  
    print(f"Scratch model: final train MSE = {scratch_history['train_loss'][-1]:.4f}")

    ### Train linear reg using nn.Linear + nn.MSELoss + optim.SGD,
    ### but same as scratch lin reg (same lr/epochs, zero-init, and batch
     
    # reset the rng so that this model starts with the same starting state as the lin reg scratch
    # bc I want to compare the 2 modles
    torch.manual_seed(rng_seed)
    torch_model = TorchLinearRegression(in_features)
    torch_history = fit_torch(
        torch_model, 
        train_loader, 
        val_loader, 
        device, 
        epochs=args.epochs, 
        lr=args.lr,
    )
    print(f"nn.Linear model: final train MSE = {torch_history['train_loss'][-1]:.4f}")

    plot_loss_curves(
        scratch_history, 
        torch_history, 
        FIG_DIR / "01_loss_curves.png"
    )

    ### Compare the two models' learned weights directly 
    scratch_weights = scratch_model.w.detach().cpu().numpy()
    torch_weights = torch_model.linear.weight.detach().cpu().numpy().ravel()
    plot_weight_comparison(
        scratch_weights, 
        torch_weights, 
        data.feature_names, 
        FIG_DIR / "02_weight_comparison.png"
    )

    ### Evaluate both models on the test set
    # compare against ESOL's own published linear baseline

    scratch_preds_std, targets_std = predict(
        scratch_model, 
        test_loader, 
        device
    )
    torch_preds_std, _ = predict(
        torch_model, 
        test_loader, 
        device
    )

    # convert predictions back to their original scale (they were standardized for training)
    scratch_preds = data.inverse_y(scratch_preds_std)
    torch_preds = data.inverse_y(torch_preds_std)
    measured = data.inverse_y(targets_std)
    baseline_preds = data.baseline_test  # the published ESOL linear model

    scratch_rmse, scratch_r2 = rmse_r2(scratch_preds, measured)
    torch_rmse, torch_r2 = rmse_r2(torch_preds, measured)
    baseline_rmse, baseline_r2 = rmse_r2(baseline_preds, measured)

    print(f"Scratch: test RMSE = {scratch_rmse:.3f}, R^2 = {scratch_r2:.3f}")
    print(f"nn.Linear: test RMSE = {torch_rmse:.3f}, R^2 = {torch_r2:.3f}")
    print(f"ESOL baseline: test RMSE = {baseline_rmse:.3f}, R^2 = {baseline_r2:.3f}")

    plot_predictions(
        scratch_preds,
        torch_preds,
        baseline_preds,
        measured,
        scratch_rmse,
        torch_rmse,
        baseline_rmse,
        FIG_DIR / "03_predictions.png",
    )


if __name__ == "__main__":
    main()
