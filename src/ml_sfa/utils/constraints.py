"""Monotonicity constraint modules for neural network frontiers.

Provides ``MonotonicLinear`` and ``MonotonicMLP`` that guarantee
non-decreasing output with respect to each input dimension by
constraining weights to be non-negative via softplus transformation.

Requires ``torch`` (install via ``pip install ml-sfa[nn]``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MonotonicLinear", "MonotonicMLP"]


class MonotonicLinear(nn.Module):
    """Linear layer with non-negative weights via softplus transform.

    The effective weight matrix is ``softplus(raw_weight)``, ensuring all
    weights are non-negative.  The bias is unrestricted.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # (in, out) layout for x @ W matmul.  kaiming_uniform_ default
        # mode="fan_in" uses dim[0]=in_features, which is correct.
        self.raw_weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.raw_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transform with non-negative weights.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, out_features)``.
        """
        w = F.softplus(self.raw_weight)
        return x @ w + self.bias


class MonotonicMLP(nn.Module):
    """Multi-layer perceptron with monotonicity guarantee.

    All hidden layers use ``MonotonicLinear`` (non-negative weights) with
    ``Softplus`` activation.  The output layer is also ``MonotonicLinear``
    but without activation, producing an unbounded scalar output.

    The combination of non-negative weights and monotone-increasing activation
    guarantees that the network output is non-decreasing in each input
    dimension.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    hidden_dims : list[int]
        Width of each hidden layer.
    """

    def __init__(self, in_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(MonotonicLinear(prev_dim, h_dim))
            layers.append(nn.Softplus())
            prev_dim = h_dim

        # Output layer: monotonic linear, no activation
        layers.append(MonotonicLinear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the monotonic MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, in_dim)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, 1)``.
        """
        result: torch.Tensor = self.net(x)
        return result
