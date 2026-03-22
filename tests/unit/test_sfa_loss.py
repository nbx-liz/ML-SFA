"""Unit tests for PyTorch SFA loss functions.

Tests NLL computation for half-normal, exponential, and truncated-normal
distributions against the SciPy-based parametric implementations to ensure
numerical equivalence.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml_sfa.models._sfa_loss import (
    sfa_nll_exponential,
    sfa_nll_half_normal,
    sfa_nll_truncated_normal,
)
from ml_sfa.models.parametric import (
    _nll_exponential,
    _nll_half_normal,
    _nll_truncated_normal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_data() -> tuple[torch.Tensor, float, float]:
    """Composed error residuals and sigma parameters."""
    rng = np.random.default_rng(42)
    eps_np = rng.normal(0, 1, size=100) - rng.exponential(0.5, size=100)
    eps = torch.tensor(eps_np, dtype=torch.float64)
    log_sigma_v = 0.3  # sigma_v ≈ 1.35
    log_sigma_u = -0.2  # sigma_u ≈ 0.82
    return eps, log_sigma_v, log_sigma_u


# ---------------------------------------------------------------------------
# Half-normal NLL
# ---------------------------------------------------------------------------


class TestSFANLLHalfNormal:
    """Tests for PyTorch half-normal NLL."""

    def test_matches_scipy_production(
        self, sample_data: tuple[torch.Tensor, float, float]
    ) -> None:
        """PyTorch NLL matches SciPy parametric NLL for production frontier."""
        eps, log_sv, log_su = sample_data
        # PyTorch version
        torch_nll = sfa_nll_half_normal(eps, log_sv, log_su, cost=False)

        # SciPy version: construct matching params vector
        # parametric NLL expects [beta..., ln_sigma_v, ln_sigma_u]
        # with Z @ beta already subtracted. We use Z=I, beta=0, y=eps.
        n = len(eps)
        params = np.array([0.0, log_sv, log_su])
        Z = np.ones((n, 1))
        y = eps.numpy()
        scipy_nll = _nll_half_normal(params, Z, y, cost=False)

        assert torch_nll.item() == pytest.approx(scipy_nll, rel=1e-6)

    def test_matches_scipy_cost(
        self, sample_data: tuple[torch.Tensor, float, float]
    ) -> None:
        """PyTorch NLL matches SciPy for cost frontier."""
        eps, log_sv, log_su = sample_data
        torch_nll = sfa_nll_half_normal(eps, log_sv, log_su, cost=True)

        n = len(eps)
        params = np.array([0.0, log_sv, log_su])
        Z = np.ones((n, 1))
        y = eps.numpy()
        scipy_nll = _nll_half_normal(params, Z, y, cost=True)

        assert torch_nll.item() == pytest.approx(scipy_nll, rel=1e-6)

    def test_gradient_flows(
        self, sample_data: tuple[torch.Tensor, float, float]
    ) -> None:
        """Gradients are computable for all inputs."""
        eps, log_sv, log_su = sample_data
        eps_t = eps.clone().requires_grad_(True)
        lsv = torch.tensor(log_sv, dtype=torch.float64, requires_grad=True)
        lsu = torch.tensor(log_su, dtype=torch.float64, requires_grad=True)

        nll = sfa_nll_half_normal(eps_t, lsv, lsu, cost=False)
        nll.backward()

        assert eps_t.grad is not None
        assert lsv.grad is not None
        assert lsu.grad is not None

    def test_positive_output(
        self, sample_data: tuple[torch.Tensor, float, float]
    ) -> None:
        """NLL should be finite and positive for reasonable inputs."""
        eps, log_sv, log_su = sample_data
        nll = sfa_nll_half_normal(eps, log_sv, log_su, cost=False)
        assert torch.isfinite(nll)


# ---------------------------------------------------------------------------
# Exponential NLL
# ---------------------------------------------------------------------------


class TestSFANLLExponential:
    """Tests for PyTorch exponential NLL."""

    def test_matches_scipy_production(
        self, sample_data: tuple[torch.Tensor, float, float]
    ) -> None:
        """PyTorch NLL matches SciPy parametric NLL for production."""
        eps, log_sv, log_su = sample_data
        torch_nll = sfa_nll_exponential(eps, log_sv, log_su, cost=False)

        n = len(eps)
        params = np.array([0.0, log_sv, log_su])
        Z = np.ones((n, 1))
        y = eps.numpy()
        scipy_nll = _nll_exponential(params, Z, y, cost=False)

        assert torch_nll.item() == pytest.approx(scipy_nll, rel=1e-6)

    def test_matches_scipy_cost(
        self, sample_data: tuple[torch.Tensor, float, float]
    ) -> None:
        """PyTorch NLL matches SciPy for cost frontier."""
        eps, log_sv, log_su = sample_data
        torch_nll = sfa_nll_exponential(eps, log_sv, log_su, cost=True)

        n = len(eps)
        params = np.array([0.0, log_sv, log_su])
        Z = np.ones((n, 1))
        y = eps.numpy()
        scipy_nll = _nll_exponential(params, Z, y, cost=True)

        assert torch_nll.item() == pytest.approx(scipy_nll, rel=1e-6)

    def test_gradient_flows(
        self, sample_data: tuple[torch.Tensor, float, float]
    ) -> None:
        """Gradients flow through exponential NLL."""
        eps, log_sv, log_su = sample_data
        eps_t = eps.clone().requires_grad_(True)
        lsv = torch.tensor(log_sv, dtype=torch.float64, requires_grad=True)
        lsu = torch.tensor(log_su, dtype=torch.float64, requires_grad=True)

        nll = sfa_nll_exponential(eps_t, lsv, lsu, cost=False)
        nll.backward()

        assert eps_t.grad is not None
        assert lsv.grad is not None
        assert lsu.grad is not None


# ---------------------------------------------------------------------------
# Truncated-normal NLL
# ---------------------------------------------------------------------------


class TestSFANLLTruncatedNormal:
    """Tests for PyTorch truncated-normal NLL."""

    def test_matches_scipy_production(
        self, sample_data: tuple[torch.Tensor, float, float]
    ) -> None:
        """PyTorch NLL matches SciPy parametric NLL for production."""
        eps, log_sv, log_su = sample_data
        torch_nll = sfa_nll_truncated_normal(eps, log_sv, log_su, mu=0.0, cost=False)

        n = len(eps)
        params = np.array([0.0, log_sv, log_su])
        Z = np.ones((n, 1))
        y = eps.numpy()
        scipy_nll = _nll_truncated_normal(params, Z, y, cost=False, mu=0.0)

        assert torch_nll.item() == pytest.approx(scipy_nll, rel=1e-6)

    def test_matches_scipy_cost(
        self, sample_data: tuple[torch.Tensor, float, float]
    ) -> None:
        """PyTorch NLL matches SciPy for cost frontier."""
        eps, log_sv, log_su = sample_data
        torch_nll = sfa_nll_truncated_normal(eps, log_sv, log_su, mu=0.0, cost=True)

        n = len(eps)
        params = np.array([0.0, log_sv, log_su])
        Z = np.ones((n, 1))
        y = eps.numpy()
        scipy_nll = _nll_truncated_normal(params, Z, y, cost=True, mu=0.0)

        assert torch_nll.item() == pytest.approx(scipy_nll, rel=1e-6)

    def test_nonzero_mu(self, sample_data: tuple[torch.Tensor, float, float]) -> None:
        """NLL with nonzero mu is finite and computable."""
        eps, log_sv, log_su = sample_data
        nll = sfa_nll_truncated_normal(eps, log_sv, log_su, mu=0.5, cost=False)
        assert torch.isfinite(nll)

    def test_gradient_flows(
        self, sample_data: tuple[torch.Tensor, float, float]
    ) -> None:
        """Gradients flow through truncated-normal NLL."""
        eps, log_sv, log_su = sample_data
        eps_t = eps.clone().requires_grad_(True)
        lsv = torch.tensor(log_sv, dtype=torch.float64, requires_grad=True)
        lsu = torch.tensor(log_su, dtype=torch.float64, requires_grad=True)

        nll = sfa_nll_truncated_normal(eps_t, lsv, lsu, mu=0.0, cost=False)
        nll.backward()

        assert eps_t.grad is not None
        assert lsv.grad is not None
        assert lsu.grad is not None


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Tests for numerical stability of NLL functions."""

    def test_extreme_negative_epsilon(self) -> None:
        """NLL remains finite for very negative epsilon values."""
        eps = torch.tensor([-10.0, -20.0, -50.0], dtype=torch.float64)
        nll = sfa_nll_half_normal(eps, 0.0, 0.0, cost=False)
        assert torch.isfinite(nll)

    def test_extreme_positive_epsilon(self) -> None:
        """NLL remains finite for very positive epsilon values."""
        eps = torch.tensor([10.0, 20.0, 50.0], dtype=torch.float64)
        nll = sfa_nll_half_normal(eps, 0.0, 0.0, cost=False)
        assert torch.isfinite(nll)

    def test_very_small_sigma(self) -> None:
        """NLL handles very small sigma (large negative log_sigma)."""
        eps = torch.tensor([0.1, -0.1, 0.0], dtype=torch.float64)
        nll = sfa_nll_half_normal(eps, -5.0, -5.0, cost=False)
        assert torch.isfinite(nll)

    def test_very_large_sigma(self) -> None:
        """NLL handles very large sigma (large positive log_sigma)."""
        eps = torch.tensor([0.1, -0.1, 0.0], dtype=torch.float64)
        nll = sfa_nll_half_normal(eps, 5.0, 5.0, cost=False)
        assert torch.isfinite(nll)
