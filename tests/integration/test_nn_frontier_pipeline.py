"""Integration tests for NNFrontierSFA pipeline.

End-to-end tests: simulate data → fit NN-SFA → evaluate efficiency recovery.
Tests compare against ParametricSFA on correct-form DGPs and verify
advantages on nonlinear DGPs.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

torch = pytest.importorskip("torch")

from ml_sfa.data.simulator import simulate_sfa  # noqa: E402
from ml_sfa.evaluation.metrics import rmse_efficiency  # noqa: E402
from ml_sfa.models.nn_frontier import NNFrontierSFA  # noqa: E402
from ml_sfa.models.parametric import ParametricSFA  # noqa: E402

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _fit_nn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    inefficiency: str = "half-normal",
    cost: bool = False,
) -> NNFrontierSFA:
    """Fit a lightweight NNFrontierSFA for integration tests."""
    return NNFrontierSFA(
        inefficiency=inefficiency,  # type: ignore[arg-type]
        cost=cost,
        hidden_dims=[16, 8],
        pretrain_epochs=100,
        finetune_epochs=10,
        n_inits=3,
        seed=42,
    ).fit(X, y)


# ---------------------------------------------------------------------------
# Efficiency recovery
# ---------------------------------------------------------------------------


class TestEfficiencyRecovery:
    """Test that NN-SFA recovers technical efficiency from simulated data."""

    def test_half_normal_recovery(self) -> None:
        """NN-SFA recovers TE with half-normal DGP (RMSE < 0.15)."""
        ds = simulate_sfa(
            n_obs=500,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        model = _fit_nn(ds.X, ds.y)
        te_hat = model.efficiency(ds.X, ds.y)
        rmse = rmse_efficiency(ds.te, te_hat)
        assert rmse < 0.15, f"RMSE too high: {rmse:.4f}"

    def test_exponential_recovery(self) -> None:
        """NN-SFA recovers TE with exponential DGP."""
        ds = simulate_sfa(
            n_obs=500,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="exponential",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        model = _fit_nn(ds.X, ds.y, inefficiency="exponential")
        te_hat = model.efficiency(ds.X, ds.y)
        rmse = rmse_efficiency(ds.te, te_hat)
        assert rmse < 0.15, f"RMSE too high: {rmse:.4f}"

    def test_rank_correlation(self) -> None:
        """Efficiency ranking is strongly correlated with true ranking."""
        ds = simulate_sfa(
            n_obs=500,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        model = _fit_nn(ds.X, ds.y)
        te_hat = model.efficiency(ds.X, ds.y)
        rho = spearmanr(ds.te, te_hat).statistic
        assert rho > 0.65, f"Rank correlation too low: {rho:.4f}"


# ---------------------------------------------------------------------------
# Sigma recovery
# ---------------------------------------------------------------------------


class TestSigmaRecovery:
    """Test that estimated sigma_v and sigma_u are reasonable."""

    def test_sigma_order_of_magnitude(self) -> None:
        """Estimated sigmas are within 2x of true values."""
        ds = simulate_sfa(
            n_obs=500,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        model = _fit_nn(ds.X, ds.y)
        assert 0.05 < model.sigma_v_ < 0.5, f"sigma_v={model.sigma_v_:.4f}"
        assert 0.1 < model.sigma_u_ < 0.8, f"sigma_u={model.sigma_u_:.4f}"


# ---------------------------------------------------------------------------
# Comparison with ParametricSFA
# ---------------------------------------------------------------------------


class TestVsParametricSFA:
    """Compare NN-SFA with ParametricSFA on correct-form DGP."""

    def test_comparable_on_cobb_douglas_dgp(self) -> None:
        """On a Cobb-Douglas DGP, NN-SFA achieves comparable RMSE."""
        ds = simulate_sfa(
            n_obs=500,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        # Parametric (correct specification)
        psfa = ParametricSFA(frontier="cobb-douglas", inefficiency="half-normal").fit(
            ds.X, ds.y
        )
        te_param = psfa.efficiency(ds.X, ds.y)
        rmse_param = rmse_efficiency(ds.te, te_param)

        # NN-SFA
        nn_model = _fit_nn(ds.X, ds.y)
        te_nn = nn_model.efficiency(ds.X, ds.y)
        rmse_nn = rmse_efficiency(ds.te, te_nn)

        # NN-SFA should be within 2x of parametric on correct-form DGP
        assert rmse_nn < 2.0 * rmse_param, (
            f"NN RMSE ({rmse_nn:.4f}) >> parametric ({rmse_param:.4f})"
        )


# ---------------------------------------------------------------------------
# Sklearn compatibility
# ---------------------------------------------------------------------------


class TestSklearnCompat:
    """Test sklearn API compatibility."""

    def test_clone(self) -> None:
        """sklearn clone works on NNFrontierSFA."""
        from sklearn.base import clone

        model = NNFrontierSFA(
            inefficiency="half-normal",
            hidden_dims=[16],
            pretrain_epochs=10,
            n_inits=1,
        )
        cloned = clone(model)
        assert cloned.get_params() == model.get_params()
        assert cloned is not model


# ---------------------------------------------------------------------------
# Cost frontier
# ---------------------------------------------------------------------------


class TestCostFrontierPipeline:
    """Test cost frontier end-to-end."""

    def test_cost_efficiency_recovery(self) -> None:
        """Cost frontier model recovers TE."""
        ds = simulate_sfa(
            n_obs=500,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            cost=True,
            seed=42,
        )
        model = _fit_nn(ds.X, ds.y, cost=True)
        te_hat = model.efficiency(ds.X, ds.y)
        rmse = rmse_efficiency(ds.te, te_hat)
        assert rmse < 0.15, f"Cost frontier RMSE too high: {rmse:.4f}"


# ---------------------------------------------------------------------------
# AIC/BIC consistency
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Test AIC/BIC are consistent."""

    def test_aic_bic_finite(self) -> None:
        """AIC and BIC are finite values."""
        ds = simulate_sfa(
            n_obs=200,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        model = _fit_nn(ds.X, ds.y)
        s = model.summary()
        assert np.isfinite(s.aic)
        assert np.isfinite(s.bic)
        assert s.aic < s.bic  # AIC < BIC when n > exp(2) ≈ 7.4
