"""
multilayer perceptron trained to predict aqueous
solubility from the Delaney ESOL dataset's molecular descriptors
(Delaney, 2004).

Pipeline:
    train the same MLP architecture (nn.Linear + activation) under four
    weight-init schemes -- zeros / random / xavier / he -- with Adam and
    backprop, holding everything else fixed -> compare their training-
    loss curves and test-set predictions against the dataset's own
    published linear baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data_load_prep import load_dataset, make_loaders

from mlp import MLP
from plotting import plot_init_comparison, plot_predictions
from train import fit, get_device, predict, rmse_r2

FIG_DIR = Path(__file__).resolve().parent / "fig"

INIT_SCHEMES = ["zeros", "random", "xavier", "he"]


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer widths",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Minibatch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs per init scheme",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Adam learning rate",
    )
    return parser.parse_args()


def main() -> None:
    args = build_parser()
    FIG_DIR.mkdir(exist_ok=True)

    device = get_device()
    print(f"Training device: {device}")

    data = load_dataset(seed=0)
    train_loader, val_loader, test_loader = make_loaders(
        data,
        batch_size=args.batch_size
    )

    print(f"Loaded {len(data.X_train)} train / {len(data.X_val)} val / {len(data.X_test)} test compounds")

    in_features = data.X_train.shape[1]

    # Train the same architecture from scratch under each init scheme,
    # keeping everything else (data, batch size, lr, epochs) fixed, so
    # any difference in the results below is attributable to init alone.
    histories = {}
    preds_by_scheme = {}
    rmse_by_scheme = {}
    measured = None

    for scheme in INIT_SCHEMES:
        model = MLP(
            in_features,
            hidden_sizes=tuple(args.hidden_sizes),
            activation="relu",
            init_scheme=scheme,
        )

        history = fit(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr
        )
        histories[scheme] = history
        print(
            f"  init={scheme:<8} final train MSE = {history['train_loss'][-1]:.4f}, "
            f"val MSE = {history['val_loss'][-1]:.4f}"
        )

        preds_std, targets_std = predict(model, test_loader, device)
        # convert predictions back to their original scale (they were standardized for training)
        preds_by_scheme[scheme] = data.inverse_y(preds_std)
        measured = data.inverse_y(targets_std)

        rmse, r2 = rmse_r2(preds_by_scheme[scheme], measured)
        rmse_by_scheme[scheme] = rmse
        print(f"  init={scheme:<8} test RMSE = {rmse:.3f}, R^2 = {r2:.3f}")

    # the published ESOL linear model -- check the MLP against this
    baseline_preds = data.baseline_test
    baseline_rmse, baseline_r2 = rmse_r2(baseline_preds, measured)
    print(f"ESOL linear baseline: test RMSE = {baseline_rmse:.3f}, R^2 = {baseline_r2:.3f}")

    plot_init_comparison(histories, FIG_DIR / "01_init_comparison.png")
    plot_predictions(
        preds_by_scheme,
        measured,
        baseline_preds,
        rmse_by_scheme,
        baseline_rmse,
        FIG_DIR / "02_predictions.png"
    )


if __name__ == "__main__":
    main()
