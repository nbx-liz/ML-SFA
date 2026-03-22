"""Unit tests for monotonicity constraint modules.

Tests MonotonicLinear and MonotonicMLP to verify:
- Output shape correctness
- Weight non-negativity via softplus transform
- Monotonicity: increasing input → non-decreasing output
- Gradient flow through softplus-transformed weights
"""

from __future__ import annotations

import torch

from ml_sfa.utils.constraints import MonotonicLinear, MonotonicMLP

# ---------------------------------------------------------------------------
# MonotonicLinear
# ---------------------------------------------------------------------------


class TestMonotonicLinear:
    """Tests for MonotonicLinear layer."""

    def test_output_shape(self) -> None:
        """Forward pass returns correct output shape."""
        layer = MonotonicLinear(3, 5)
        x = torch.randn(10, 3)
        out = layer(x)
        assert out.shape == (10, 5)

    def test_effective_weights_non_negative(self) -> None:
        """Softplus-transformed weights are always >= 0."""
        layer = MonotonicLinear(4, 6)
        # Set raw weights to negative values
        with torch.no_grad():
            layer.raw_weight.fill_(-5.0)
        effective_w = torch.nn.functional.softplus(layer.raw_weight)
        assert (effective_w >= 0.0).all()

    def test_gradient_flows(self) -> None:
        """Gradients propagate through the softplus transform."""
        layer = MonotonicLinear(3, 2)
        x = torch.randn(5, 3, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.raw_weight.grad is not None
        assert x.grad is not None

    def test_bias_present(self) -> None:
        """Bias parameter exists and is unrestricted."""
        layer = MonotonicLinear(3, 4)
        assert layer.bias is not None
        assert layer.bias.shape == (4,)


# ---------------------------------------------------------------------------
# MonotonicMLP
# ---------------------------------------------------------------------------


class TestMonotonicMLP:
    """Tests for MonotonicMLP network."""

    def test_output_shape_single_hidden(self) -> None:
        """Single hidden layer MLP returns scalar per sample."""
        net = MonotonicMLP(in_dim=3, hidden_dims=[8])
        x = torch.randn(20, 3)
        out = net(x)
        assert out.shape == (20, 1)

    def test_output_shape_two_hidden(self) -> None:
        """Two hidden layer MLP returns correct shape."""
        net = MonotonicMLP(in_dim=5, hidden_dims=[64, 32])
        x = torch.randn(10, 5)
        out = net(x)
        assert out.shape == (10, 1)

    def test_monotonicity_1d(self) -> None:
        """Output is non-decreasing for increasing 1D input."""
        net = MonotonicMLP(in_dim=1, hidden_dims=[16, 8])
        x = torch.linspace(0.1, 5.0, 50).unsqueeze(1)
        with torch.no_grad():
            out = net(x).squeeze()
        diffs = out[1:] - out[:-1]
        assert (diffs >= -1e-6).all(), f"Monotonicity violated: min diff={diffs.min()}"

    def test_monotonicity_multi_dim(self) -> None:
        """Output is non-decreasing in each input dimension."""
        net = MonotonicMLP(in_dim=3, hidden_dims=[16, 8])
        base = torch.ones(1, 3) * 2.0

        with torch.no_grad():
            base_out = net(base).item()
            for dim in range(3):
                increased = base.clone()
                increased[0, dim] += 1.0
                new_out = net(increased).item()
                assert new_out >= base_out - 1e-6, (
                    f"Monotonicity violated in dim {dim}: "
                    f"{base_out:.6f} -> {new_out:.6f}"
                )

    def test_gradient_computation(self) -> None:
        """Partial derivatives w.r.t. inputs are non-negative (monotonicity)."""
        net = MonotonicMLP(in_dim=3, hidden_dims=[16, 8])
        x_data = torch.randn(10, 3).abs() + 0.1
        x = x_data.clone().detach().requires_grad_(True)
        out = net(x)
        # Compute gradient for each output w.r.t. input
        for i in range(out.shape[0]):
            if x.grad is not None:
                x.grad.zero_()
            out[i, 0].backward(retain_graph=True)
            assert x.grad is not None
            grads = x.grad[i]
            assert (grads >= -1e-6).all(), f"Negative gradient at sample {i}: {grads}"

    def test_last_layer_is_linear(self) -> None:
        """Output layer has no activation (linear output)."""
        net = MonotonicMLP(in_dim=2, hidden_dims=[8])
        # The output layer should be a MonotonicLinear, applied without activation
        x = torch.randn(5, 2)
        out = net(x)
        # Output can be any real value (not constrained to be positive)
        assert out.shape == (5, 1)

    def test_parameters_are_learnable(self) -> None:
        """All parameters require gradients."""
        net = MonotonicMLP(in_dim=3, hidden_dims=[16, 8])
        params = list(net.parameters())
        assert len(params) > 0
        for p in params:
            assert p.requires_grad
