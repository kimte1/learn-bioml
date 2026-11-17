from __future__ import annotations

import torch
from torch import nn


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


def init_weights(module: nn.Module, scheme: str) -> None:
    """Parameter initialization, applied once before training starts.

    The forward pass is a chain of matrix multiplies, so the scale of
    the initial weights controls whether pre-activation variance stays
    roughly constant layer to layer, shrinks toward zero ("vanishing"),
    or grows without bound ("exploding") -- and the same is true in
    reverse for gradients during backprop. Identical initial weights
    would also leave every hidden unit computing the same function and
    receiving the same gradient (no symmetry breaking), which is why
    these schemes draw from a random distribution rather than a constant.

    - "zeros": every weight starts at exactly 0. Every hidden unit in a
      layer then computes the same output and receives the same
      gradient, so they stay identical forever -- a textbook symmetry-
      breaking failure, kept here to make that failure visible rather
      than just asserted.
    - "random": unscaled N(0, 1) -- breaks symmetry (unlike "zeros") but
      ignores fan-in entirely, so pre-activation variance grows with
      every layer's width instead of staying constant; with enough
      layers/width this tends to saturate (sigmoid/tanh) or blow up
      (ReLU) rather than train cleanly.
    - "xavier": Glorot & Bengio, 2010 -- scales by fan-in and fan-out so
      variance is preserved forward and backward; derived for
      linear/tanh-like (roughly symmetric) activations.
    - "he": He et al., 2015 -- scales by fan-in only, with a factor of 2
      to compensate for ReLU zeroing out (on average) half of the
      pre-activations.
    """
    if not isinstance(module, nn.Linear):
        return
    if scheme == "zeros":
        nn.init.zeros_(module.weight)
    elif scheme == "random":
        nn.init.normal_(module.weight, mean=0.0, std=1.0)
    elif scheme == "xavier":
        nn.init.xavier_uniform_(module.weight)
    elif scheme == "he":
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    else:
        raise ValueError(f"unknown init scheme: {scheme!r}")
    nn.init.zeros_(module.bias)


class MLP(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        activation: str = "relu",
        init_scheme: str = "he",
    ):
        super().__init__()
        activation_cls = ACTIVATIONS[activation]

        dims = [in_features, *hidden_sizes]

        layers: list[nn.Module] = []

        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))  # affine: z = xW^T + b
            layers.append(activation_cls())  # nonlinearity: a = f(z)

        layers.append(nn.Linear(dims[-1], 1))  # final layer stays linear -- this is regressio
        self.net = nn.Sequential(*layers)

        self.apply(lambda m: init_weights(m, init_scheme))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass: push the input through every affine layer and
        # activation in sequence. Each nn.Linear produces a new set of
        # features (a "representation") from the previous layer's
        # output, and each activation makes that representation
        # nonlinear before the next layer builds on top of it. PyTorch
        # silently records every tensor op here into a computation graph
        # -- that graph is what `loss.backward()` later walks to compute
        # gradients; nothing is "trained" by calling this method.
        return self.net(x).squeeze(-1)
