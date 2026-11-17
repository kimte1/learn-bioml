ssssss"""The rigorous training pipeline this project is about: reproducible
seeding, a choice of optimizer (SGD / SGD+momentum / Adam / AdamW) and
learning-rate schedule, gradient clipping, early stopping, and
checkpointing -- on top of the same forward/backward loop from
`projects/11_mlp/train.py`.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, _LRScheduler
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """check if MPS (Apple Silicon GPU) is available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Reproducible seeding: every source of randomness a training run
    touches (Python's own `random`, NumPy's RNG via `data_load_prep`'s
    train/val/test split, and PyTorch's weight init / dropout masks /
    data-loader shuffling) is seeded from one place, so re-running with
    the same seed reproduces the same run bit-for-bit (modulo
    nondeterministic GPU kernels, which don't apply on CPU/MPS here).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_optimizer(
    model: nn.Module,
    name: str,
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    - "sgd": plain stochastic gradient descent -- each step moves
      directly against the current gradient, nothing more.
    - "sgd_momentum": SGD with a velocity term that accumulates an
      exponential moving average of past gradients, which damps
      oscillations across narrow loss-surface valleys and can speed up
      convergence along consistent directions.
    - "adam" (Kingma & Ba, 2015): adapts a per-parameter learning rate
      from running estimates of the gradient's first and second moments.
    - "adamw" (Loshchilov & Hutter, 2019): same adaptive updates as Adam,
      but decouples weight decay from the gradient-based update instead
      of folding it into the gradient like Adam/SGD do -- the
      "correct" way to apply L2-style regularization under an adaptive
      optimizer.
    """
    params = model.parameters()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer: {name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str | None,
    epochs: int,
) -> _LRScheduler | ReduceLROnPlateau | None:
    """
    - None: constant learning rate throughout.
    - "step": multiply lr by 0.5 every epochs/4 epochs -- coarse,
      scheduled decay.
    - "cosine": anneal lr from its initial value down to ~0 following a
      cosine curve over the full run -- smooth decay with no
      hyperparameters besides the horizon.
    - "plateau": only drop lr when val loss stops improving for several
      epochs -- the schedule reacts to training dynamics instead of
      following a fixed curve.
    """
    if name is None or name == "none":
        return None
    elif name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 4), gamma=0.5)
    elif name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, epochs // 10))
    else:
        raise ValueError(f"unknown scheduler: {name!r}")


class EarlyStopper:
    """Stops training once val loss hasn't improved by at least
    `min_delta` for `patience` consecutive epochs -- a guard against
    training well past the point of diminishing (or negative) returns on
    generalization, without having to guess the right epoch count ahead
    of time.
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.is_best = False
        self.should_stop = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.is_best = True
        else:
            self.counter += 1
            self.is_best = False
            if self.counter >= self.patience:
                self.should_stop = True


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
) -> None:
    """Save everything needed to either resume training (model + optimizer
    state) or just reload the trained model for inference."""
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str | None = "cpu",
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"], checkpoint["val_loss"]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float | None,
) -> float:
    loss_fn = nn.MSELoss(reduction="sum")
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, n_examples = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(is_training):
            preds = model(X)  # forward pass
            loss = loss_fn(preds, y)

        if is_training:
            optimizer.zero_grad()
            loss.backward()  # backpropagation
            if grad_clip_norm is not None:
                # Rescale the gradient vector (across all parameters) to
                # have norm at most grad_clip_norm, *before* the
                # optimizer uses it -- caps how far a single bad batch
                # (or too-high a learning rate) can move the weights in
                # one step, without changing the gradient's direction.
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        n_examples += X.shape[0]

    return total_loss / n_examples


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = 200,
    optimizer_name: str = "adam",
    lr: float = 1e-2,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    scheduler_name: str | None = None,
    grad_clip_norm: float | None = None,
    patience: int | None = None,
    checkpoint_path: Path | None = None,
    seed: int | None = None,
) -> dict:
    """Train `model`, returning its loss/lr history plus early-stopping
    and checkpointing bookkeeping. Every other ablation in this project
    is this same loop with different arguments -- the goal is that
    changing one argument changes exactly one thing about training.
    """
    if seed is not None:
        set_seed(seed)

    model.to(device)
    optimizer = build_optimizer(model, optimizer_name, lr, momentum, weight_decay)
    scheduler = build_scheduler(optimizer, scheduler_name, epochs)
    stopper = EarlyStopper(patience) if patience is not None else None

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = 0
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, device, optimizer, grad_clip_norm)
        val_loss = _run_epoch(model, val_loader, device, None, None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)  # needs the metric it's reacting to
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model, optimizer, epoch, val_loss)

        if stopper is not None:
            stopper.step(val_loss)
            if stopper.should_stop:
                stopped_epoch = epoch
                break

    return {
        **history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute MSE loss over `loader` without updating any parameters --
    used standalone outside of `fit()`'s loop, e.g. to verify a reloaded
    checkpoint reproduces the val loss it was saved with.
    """
    return _run_epoch(model, loader, device, None, None)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).cpu())
        targets.append(y)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def rmse_r2(preds: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return rmse, r2
