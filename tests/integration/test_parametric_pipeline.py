"""Integration tests for the full simulate -> fit -> evaluate pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from ml_sfa.data.simulator import SFADataset, simulate_sfa
from ml_sfa.evaluation.metrics import aic, bic, rank_correlation, rmse_efficiency
from ml_sfa.models.parametric import ParametricSFA

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cobb_douglas_data() -> SFADataset:
    """Cobb-Douglas DGP with moderate inefficiency."""
    return simulate_sfa(
        n_obs=2000,
        n_inputs=3,
        frontier_type="cobb-douglas",
        inefficiency_dist="half-normal",
        sigma_v=0.1,
        sigma_u=0.2,
        seed=42,
    )


@pytest.fixture
def translog_data() -> SFADataset:
    """Translog DGP with moderate inefficiency."""
    return simulate_sfa(
        n_obs=2000,
        n_inputs=2,
        frontier_type="translog",
        inefficiency_dist="half-normal",
        sigma_v=0.1,
        sigma_u=0.2,
        seed=123,
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------


class TestCobbDouglasPipeline:
    """Full pipeline: simulate -> fit -> predict -> evaluate (Cobb-Douglas)."""

    def test_simulate_fit_evaluate_cobb_douglas(
        self, cobb_douglas_data: SFADataset
    ) -> None:
        """Fit ParametricSFA and check efficiency estimation quality."""
        data = cobb_douglas_data
        model = ParametricSFA(frontier="cobb-douglas", inefficiency="half-normal")
        model.fit(data.X, data.y)

        te_hat = model.efficiency(data.X, data.y)

        rmse = rmse_efficiency(data.te, te_hat)
        rho = rank_correlation(data.te, te_hat)

        assert rmse < 0.15, f"RMSE(TE) too high: {rmse:.4f}"
        assert rho > 0.7, f"Rank correlation too low: {rho:.4f}"

    def test_predict_close_to_true_frontier(
        self, cobb_douglas_data: SFADataset
    ) -> None:
        """Frontier prediction should approximate the true frontier."""
        data = cobb_douglas_data
        model = ParametricSFA(frontier="cobb-douglas", inefficiency="half-normal")
        model.fit(data.X, data.y)

        y_hat = model.predict(data.X)
        # Reconstruct true frontier from data: y = frontier + v - u
        true_frontier = data.y - data.v + data.u

        mse = float(np.mean((y_hat - true_frontier) ** 2))
        assert mse < 0.05, f"Frontier MSE too high: {mse:.4f}"


class TestTranslogPipeline:
    """Full pipeline for Translog frontier."""

    def test_simulate_fit_evaluate_translog(self, translog_data: SFADataset) -> None:
        """Translog frontier achieves reasonable efficiency estimation."""
        data = translog_data
        model = ParametricSFA(frontier="translog", inefficiency="half-normal")
        model.fit(data.X, data.y)

        te_hat = model.efficiency(data.X, data.y)

        rmse = rmse_efficiency(data.te, te_hat)
        rho = rank_correlation(data.te, te_hat)

        assert rmse < 0.15, f"RMSE(TE) too high: {rmse:.4f}"
        assert rho > 0.75, f"Rank correlation too low: {rho:.4f}"


class TestAllDistributions:
    """Fit should converge for all inefficiency distributions."""

    @pytest.mark.parametrize("dist", ["half-normal", "truncated-normal", "exponential"])
    def test_fit_converges(self, dist: str, cobb_douglas_data: SFADataset) -> None:
        """Each distribution should converge and produce valid TE."""
        data = cobb_douglas_data
        model = ParametricSFA(frontier="cobb-douglas", inefficiency=dist)
        model.fit(data.X, data.y)

        te_hat = model.efficiency(data.X, data.y)
        assert np.all((te_hat > 0) & (te_hat <= 1))
        assert np.isfinite(model.log_likelihood())


class TestSklearnCompat:
    """Verify scikit-learn API compatibility."""

    def test_sklearn_clone_and_refit(self, cobb_douglas_data: SFADataset) -> None:
        """Cloned model should produce similar results after refit."""
        data = cobb_douglas_data
        model = ParametricSFA(frontier="cobb-douglas", inefficiency="half-normal")
        model.fit(data.X, data.y)

        cloned = clone(model)
        cloned.fit(data.X, data.y)

        te_original = model.efficiency(data.X, data.y)
        te_cloned = cloned.efficiency(data.X, data.y)

        rho = rank_correlation(te_original, te_cloned)
        assert rho > 0.99, f"Clone diverged: rank_corr = {rho:.4f}"

    def test_get_set_params(self) -> None:
        """get_params and set_params work correctly."""
        model = ParametricSFA(
            frontier="translog", inefficiency="exponential", cost=True
        )
        params = model.get_params()

        assert params["frontier"] == "translog"
        assert params["inefficiency"] == "exponential"
        assert params["cost"] is True

        model.set_params(frontier="cobb-douglas")
        assert model.frontier == "cobb-douglas"


class TestCostFrontier:
    """Cost frontier flips the sign convention."""

    def test_cost_frontier_flips_sign(self) -> None:
        """Cost frontier should estimate efficiency correctly."""
        data = simulate_sfa(
            n_obs=1000,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.1,
            sigma_u=0.2,
            seed=99,
            cost=True,
        )
        model = ParametricSFA(
            frontier="cobb-douglas", inefficiency="half-normal", cost=True
        )
        model.fit(data.X, data.y)

        te_hat = model.efficiency(data.X, data.y)
        assert np.all((te_hat > 0) & (te_hat <= 1))

        rho = rank_correlation(data.te, te_hat)
        assert rho > 0.7, f"Cost frontier rank corr too low: {rho:.4f}"


class TestMetricsConsistency:
    """AIC/BIC from summary match metric module."""

    def test_aic_bic_consistency(self, cobb_douglas_data: SFADataset) -> None:
        """Summary AIC/BIC should match standalone metric functions."""
        data = cobb_douglas_data
        model = ParametricSFA(frontier="cobb-douglas", inefficiency="half-normal")
        model.fit(data.X, data.y)

        s = model.summary()

        expected_aic = aic(s.log_likelihood, s.n_params)
        expected_bic = bic(s.log_likelihood, s.n_params, s.n_obs)

        assert abs(s.aic - expected_aic) < 1e-10
        assert abs(s.bic - expected_bic) < 1e-10
