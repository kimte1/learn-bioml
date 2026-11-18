"""
MLP with regularization/normalization knobs
Weight init is fixed to "he" (which was deecided in 11_mlp)

REUSABLE: unchanged from `projects/12_mlp_training/mlp.py` -- the model
architecture doesn't know or care that it's about to be watched by W&B.
"""

from __future__ import annotations

import torch
from torch import nn


def init_weights(module: nn.Module) -> None:
    """He init (He et al., 2015): scales by fan-in, with a factor of 2 to
    compensate for ReLU zeroing out (on average) half of the
    pre-activations. See `projects/11_mlp/mlp.py` for the comparison
    against zeros/random/xavier init that justifies this choice.
    """
    if not isinstance(module, nn.Linear):
        return
    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    nn.init.zeros_(module.bias)


class MLP(nn.Module):
    """Per hidden layer: `Linear -> [BatchNorm1d] -> ReLU -> [Dropout]`,
    then a final plain `Linear` to a single regression output.

    - BatchNorm1d (Ioffe & Szegedy, 2015) re-centers each feature to zero
      mean / unit variance across the batch before the activation, then
      applies a learnable scale/shift -- it keeps the distribution of
      inputs to each layer from drifting as the parameters below it
      change during training, which generally makes optimization faster
      and less sensitive to the learning rate. It behaves differently in
      train() (uses the current batch's statistics) vs. eval() (uses a
      running average), which is why `model.eval()` matters at test time.
   
    - Dropout (Srivastava et al., 2014) randomly zeroes units during
      training only, forcing the network not to rely on any single unit
      too heavily -- a stochastic regularizer against overfitting. It is
      a no-op in eval() (nothing is dropped at test time).
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        use_batchnorm: bool = False,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        dims = [in_features, *hidden_sizes]
        layers: list[nn.Module] = []
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))  # affine: z = xW^T + b
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())  # nonlinearity: a = f(z)
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))

        layers.append(nn.Linear(dims[-1], 1))  # final layer stays linear -- this is regression
        self.net = nn.Sequential(*layers)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
