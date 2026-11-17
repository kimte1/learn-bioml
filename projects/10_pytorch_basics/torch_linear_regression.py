"""
The same model and the same update rule as `scratch_linear_regression.py`,
built using:
   - `nn.Linear` instead of raw `w`/`b` tensors,
   - `nn.MSELoss` instead of a hand-written sum of squares,
   - `torch.optim.SGD` instead of a manual `w -= lr * w.grad`. 

   Trained with the same learning rate and the same per-batch loss reduction as the scratch model, so any difference between the two in `main.py` is coming from randomness in batch order, not from the two actually computing
something different. 
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader



class TorchLinearRegression(nn.Module):
    """key skill: nn.Module -- subclassing nn.Module and registering
    layers as attributes (here, a single nn.Linear) is what makes
    `.parameters()`, `.to(device)`, and `.train()`/`.eval()` all work
    without writing any of that bookkeeping by hand, unlike
    ScratchLinearRegression next door.
    """

    def __init__(self, in_features: int):
        super().__init__()

        # model has 1 layer with:
            # n_features as input
            # 1 output (prediction)
        self.linear = nn.Linear(in_features, 1)

        # nn.Linear's default init is a random Kaiming-uniform draw
        # overwrite it to zero so it matches the scratch model
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Linear(in_features, 1) outputs shape (batch, 1), not
        # (batch,) 
        # so squeeze(-1) drops that trailing size-1 dimension 
        # so predictions line up shape-for-shape with the (batch,) targets,
        # same as the scratch model's X @ w + b.
        return self.linear(x).squeeze(-1)



def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    # mean reduction (nn.MSELoss's default) to match scratch_linear_regression.py's
    # loss.mean() -- see the comment there on why sum reduction would
    # implicitly scale up the learning rate by the batch size.
    loss_fn = nn.MSELoss(reduction="mean")
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
            optimizer.step()

        total_loss += loss.item() * X.shape[0]
        n_examples += X.shape[0]

    return total_loss / n_examples



def fit_torch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-2,
) -> dict[str, list[float]]:
    # key skill: CPU/MPS/GPU devices -- model.to(device) recurses through
    # every nn.Parameter registered on the module (here, nn.Linear's
    # weight and bias) and moves each one, in contrast to
    # ScratchLinearRegression.to(), which has to move w and b one at a
    # time because nothing is "registered" anywhere.
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        history["train_loss"].append(_run_epoch(
            model, 
            train_loader, 
            device, 
            optimizer
        ))
        history["val_loss"].append(_run_epoch(
            model, 
            val_loader, 
            device, 
            None
        ))
    return history
