"""
Linear regression built from scratch using raw tensors
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader



class ScratchLinearRegression:

    def __init__(self, in_features: int, device: torch.device | None = None):
        self.w = torch.zeros(
            in_features, 
            device=device, 
            requires_grad=True
        )
        self.b = torch.zeros(
            1, 
            device=device, 
            requires_grad=True
        )

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        # automatically called when the class is invoked
        return X @ self.w + self.b

    def parameters(self) -> list[torch.Tensor]:
        return [self.w, self.b]

    def to(self, device: torch.device) -> "ScratchLinearRegression":
        # since I'm building from scratch (i.e. not using nn.Module)
        # .detach() to remove tensor from its earlier operations, bc a tensor already in a graphy can't be reassigned
        # .to(device) move it to MPS
        # .requires_grad() enable automatic gradient computation
        
        self.w = self.w.detach().to(device).requires_grad_()
        self.b = self.b.detach().to(device).requires_grad_()
        return self



def _run_epoch(
    model: ScratchLinearRegression,
    loader: DataLoader,
    device: torch.device,
    lr: float | None,
) -> float:
    is_training = lr is not None
    total_loss, n_examples = 0.0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(is_training):
            preds = model(X)
            loss = ((preds - y) ** 2).mean()

        if is_training:
            loss.backward()

            with torch.no_grad():
                model.w -= lr * model.w.grad
                model.b -= lr * model.b.grad

            # Gradients accumulate (add onto .grad) by default every time
            # .backward() is called, so they must be cleared after every
            # step -- otherwise this step's update would also include
            # every previous step's gradient.
            model.w.grad = None
            model.b.grad = None

        total_loss += loss.item() * X.shape[0]
        n_examples += X.shape[0]

    return total_loss / n_examples



def fit_scratch(
    model: ScratchLinearRegression,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-2,
) -> dict[str, list[float]]:
    model.to(device)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        history["train_loss"].append(_run_epoch(
            model, 
            train_loader, 
            device, 
            lr,
        ))

        # the trained model is passed to validation
        history["val_loss"].append(_run_epoch(
            model, 
            val_loader, 
            device, 
            None,
        ))
    return history
