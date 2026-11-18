"""The Week 8 training pipeline (`projects/12_mlp_training/train.py`),
instrumented with Weights & Biases.

REUSABLE: `get_device`, `set_seed`, `build_optimizer`, `build_scheduler`,
`EarlyStopper`, `save_checkpoint`/`load_checkpoint`, `evaluate`,
`predict`, and `rmse_r2` are all copied over completely unchanged -- none
of them know anything about W&B, and none of them will need to change
for the Transformer/GPU/diffusion projects later in the curriculum
either. The only function that changed is `fit()`, and only in the two
places marked below.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, _LRScheduler
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """check if MPS (Apple Silicon GPU) is available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Reproducible seeding -- unrelated to W&B itself, but it's what
    makes the *config logging* below meaningful: if `seed` is in
    `wandb.config`, anyone looking at a run later knows the exact random
    state the run started from, not just its hyperparameters.
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
    """Writes the checkpoint to a *local* file only. This is deliberately
    unaware of W&B -- `main.py` decides when a local checkpoint is worth
    uploading as a versioned W&B Artifact (see ARTIFACT/CHECKPOINT
    LOGGING below). Keeping those two concerns apart means this function
    still works verbatim in a project that isn't using W&B at all.
    """
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
            preds = model(X)
            loss = loss_fn(preds, y)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None:
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
    watch_log_freq: int | None = None,
) -> dict:
    """Same training loop as Week 8, with two additions, each marked
    below: GRADIENT AND PARAMETER MONITORING (once, before the loop) and
    TRAIN/VALIDATION METRICS + LEARNING-RATE LOGGING (once per epoch,
    inside the loop). Everything else -- the optimizer, the scheduler,
    gradient clipping, early stopping, checkpointing -- is identical to
    `12_mlp_training/train.py`'s `fit()`.
    """
    if seed is not None:
        set_seed(seed)

    model.to(device)
    optimizer = build_optimizer(model, optimizer_name, lr, momentum, weight_decay)
    scheduler = build_scheduler(optimizer, scheduler_name, epochs)
    stopper = EarlyStopper(patience) if patience is not None else None

    # --- GRADIENT AND PARAMETER MONITORING (Week 9 key skill) ---
    # wandb.watch hooks into the model's backward pass and periodically
    # logs a histogram of every parameter's VALUES and its GRADIENTS to
    # the dashboard (under the "Gradients"/"Parameters" sections of the
    # run page) -- this is how you'd actually *see* a vanishing or
    # exploding gradient problem develop over training, rather than
    # inferring it indirectly from the loss curve. `log_freq` is in
    # units of backward() calls (batches), not epochs; we default to
    # roughly once per epoch so this doesn't dominate the upload volume.
    # REUSABLE as-is: this one line is the entire gradient-monitoring
    # story for any nn.Module, MLP or Transformer alike.
    log_freq = watch_log_freq or max(1, len(train_loader))
    wandb.watch(model, log="all", log_freq=log_freq)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = 0
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, device, optimizer, grad_clip_norm)
        val_loss = _run_epoch(model, val_loader, device, None, None)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # --- TRAIN/VALIDATION METRICS + LEARNING-RATE LOGGING (Week 9 key skills) ---
        # One wandb.log() call per epoch, sent to whichever run is
        # currently active (the one `main.py` opened with wandb.init).
        # Logging `lr` here (not just loss) is what makes a learning-rate
        # SCHEDULE visible as its own chart in the dashboard -- you can
        # watch the step/cosine/plateau decay curve directly, the same
        # way `plot_lr_schedule_comparison` did with matplotlib in Week
        # 8, except now it's live and doesn't need any plotting code.
        # REUSABLE pattern: "one wandb.log(dict) call per training step,
        # with every metric you'd want on a chart as one key" is exactly
        # how every future project's training loop will report progress.
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
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
