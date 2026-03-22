"""Tests for ParametricSFA estimator.

Comprehensive tests covering design matrix construction, MLE fitting,
efficiency estimation, summary statistics, and all supported distributions.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from ml_sfa.data.simulator import SFADataset, simulate_sfa
from ml_sfa.evaluation.metrics import rank_correlation
from ml_sfa.models.parametric import ParametricSFA, build_design_matrix

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cd_data() -> SFADataset:
    """Simulated Cobb-Douglas production frontier data (n=2000)."""
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
def tl_data() -> SFADataset:
    """Simulated Translog production frontier data (n=2000)."""
    return simulate_sfa(
        n_obs=2000,
        n_inputs=3,
        frontier_type="translog",
        inefficiency_dist="half-normal",
        sigma_v=0.1,
        sigma_u=0.2,
        seed=42,
    )


@pytest.fixture
def cost_data() -> SFADataset:
    """Simulated cost frontier data."""
    return simulate_sfa(
        n_obs=1000,
        n_inputs=2,
        frontier_type="cobb-douglas",
        inefficiency_dist="half-normal",
        sigma_v=0.1,
        sigma_u=0.2,
        seed=99,
        cost=True,
    )


@pytest.fixture
def fitted_cd_model(cd_data: SFADataset) -> ParametricSFA:
    """A ParametricSFA model already fitted on Cobb-Douglas data."""
    model = ParametricSFA(frontier="cobb-douglas", inefficiency="half-normal")
    model.fit(cd_data.X, cd_data.y)
    return model


# ---------------------------------------------------------------------------
# Design Matrix tests
# ---------------------------------------------------------------------------


class TestDesignMatrix:
    """Tests for the build_design_matrix module-level function."""

    def test_cobb_douglas_design_matrix(self, rng: np.random.Generator) -> None:
        """Cobb-Douglas: log-transforms inputs and adds intercept column."""
        n, p = 50, 3
        X = rng.uniform(1.0, 10.0, size=(n, p))
        Z = build_design_matrix(X, "cobb-douglas")

        assert Z.shape == (n, p + 1)
        np.testing.assert_allclose(Z[:, 0], 1.0)
        np.testing.assert_allclose(Z[:, 1:], np.log(X))

    def test_translog_design_matrix(self, rng: np.random.Generator) -> None:
        """Translog: intercept + log terms + 0.5*squared + cross-products."""
        n, p = 50, 3
        X = rng.uniform(1.0, 10.0, size=(n, p))
        Z = build_design_matrix(X, "translog")

        expected_cols = 1 + p + p * (p + 1) // 2
        assert Z.shape == (n, expected_cols)
        np.testing.assert_allclose(Z[:, 0], 1.0)
        ln_x = np.log(X)
        np.testing.assert_allclose(Z[:, 1 : p + 1], ln_x)
        offset = 1 + p
        for j in range(p):
            np.testing.assert_allclose(Z[:, offset + j], 0.5 * ln_x[:, j] ** 2)
        offset += p
        idx = 0
        for j in range(p):
            for k in range(j + 1, p):
                np.testing.assert_allclose(Z[:, offset + idx], ln_x[:, j] * ln_x[:, k])
                idx += 1

    @pytest.mark.parametrize(
        ("frontier_type", "n_inputs", "expected_cols"),
        [
            ("cobb-douglas", 2, 3),
            ("cobb-douglas", 5, 6),
            ("translog", 2, 6),
            ("translog", 3, 10),
            ("translog", 4, 15),
        ],
    )
    def test_design_matrix_shapes(
        self,
        rng: np.random.Generator,
        frontier_type: str,
        n_inputs: int,
        expected_cols: int,
    ) -> None:
        """Design matrix has correct number of columns."""
        n = 20
        X = rng.uniform(1.0, 10.0, size=(n, n_inputs))
        Z = build_design_matrix(X, frontier_type)  # type: ignore[arg-type]
        assert Z.shape == (n, expected_cols)

    def test_non_positive_input_raises(self) -> None:
        """Non-positive values in X raise ValueError."""
        X = np.array([[1.0, -1.0], [2.0, 3.0]])
        with pytest.raises(ValueError, match="strictly positive"):
            build_design_matrix(X, "cobb-douglas")


# ---------------------------------------------------------------------------
# MLE tests
# ---------------------------------------------------------------------------


class TestMLE:
    """Tests for maximum likelihood estimation via fit()."""

    def test_fit_sets_attributes(self, cd_data: SFADataset) -> None:
        """After fit, model has all required fitted attributes."""
        model = ParametricSFA(frontier="cobb-douglas", inefficiency="half-normal")
        model.fit(cd_data.X, cd_data.y)

        assert hasattr(model, "coef_")
        assert hasattr(model, "sigma_v_")
        assert hasattr(model, "sigma_u_")
        assert hasattr(model, "log_likelihood_")
        assert hasattr(model, "n_features_in_")
        assert model.is_fitted_ is True

    def test_mle_recovers_true_params_cobb_douglas(self, cd_data: SFADataset) -> None:
        """MLE recovers true Cobb-Douglas params within tolerance."""
        model = ParametricSFA(frontier="cobb-douglas", inefficiency="half-normal")
        model.fit(cd_data.X, cd_data.y)

        true_beta = cd_data.beta
        for i, (est, true) in enumerate(zip(model.coef_, true_beta, strict=True)):
            if abs(true) > 0.01:
                rel_error = abs(est - true) / abs(true)
                assert rel_error < 0.20, (
                    f"beta[{i}]: estimated={est:.4f}, true={true:.4f}, "
                    f"rel_error={rel_error:.2%}"
                )

        assert abs(model.sigma_v_ - cd_data.sigma_v) / cd_data.sigma_v < 0.30
        assert abs(model.sigma_u_ - cd_data.sigma_u) / cd_data.sigma_u < 0.30

    def test_mle_recovers_true_params_translog(self, tl_data: SFADataset) -> None:
        """MLE recovers true translog sigma params within tolerance."""
        model = ParametricSFA(frontier="translog", inefficiency="half-normal")
        model.fit(tl_data.X, tl_data.y)

        assert abs(model.sigma_v_ - tl_data.sigma_v) / tl_data.sigma_v < 0.30
        assert abs(model.sigma_u_ - tl_data.sigma_u) / tl_data.sigma_u < 0.30

    def test_predict_returns_frontier(
        self, cd_data: SFADataset, fitted_cd_model: ParametricSFA
    ) -> None:
        """Prediction equals design_matrix @ coef_ (no error terms)."""
        y_hat = fitted_cd_model.predict(cd_data.X)
        assert y_hat.shape == (cd_data.n_obs,)

        Z = build_design_matrix(cd_data.X, "cobb-douglas")
        expected = Z @ fitted_cd_model.coef_
        np.testing.assert_allclose(y_hat, expected, atol=1e-10)

    def test_fit_returns_self(self, cd_data: SFADataset) -> None:
        """fit() returns self for method chaining."""
        model = ParametricSFA()
        result = model.fit(cd_data.X, cd_data.y)
        assert result is model

    def test_predict_before_fit_raises(self) -> None:
        """Calling predict before fit raises NotFittedError."""
        model = ParametricSFA()
        X = np.random.default_rng(0).uniform(1.0, 5.0, size=(10, 3))
        with pytest.raises(NotFittedError):
            model.predict(X)


# ---------------------------------------------------------------------------
# Efficiency tests
# ---------------------------------------------------------------------------


class TestEfficiency:
    """Tests for technical efficiency and inefficiency estimation."""

    def test_efficiency_between_0_and_1(
        self, cd_data: SFADataset, fitted_cd_model: ParametricSFA
    ) -> None:
        """All TE values are in (0, 1]."""
        te = fitted_cd_model.efficiency(cd_data.X, cd_data.y)
        assert te.shape == (cd_data.n_obs,)
        assert np.all(te > 0.0)
        assert np.all(te <= 1.0)

    def test_efficiency_rank_correlation(self) -> None:
        """On simulated data, rank correlation with true TE exceeds 0.7."""
        data = simulate_sfa(
            n_obs=2000,
            n_inputs=3,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.1,
            sigma_u=0.3,
            seed=123,
        )
        model = ParametricSFA(frontier="cobb-douglas", inefficiency="half-normal")
        model.fit(data.X, data.y)
        te_hat = model.efficiency(data.X, data.y)
        corr = rank_correlation(data.te, te_hat)
        assert corr > 0.7, f"Rank correlation too low: {corr:.3f}"

    def test_get_inefficiency_non_negative(
        self, cd_data: SFADataset, fitted_cd_model: ParametricSFA
    ) -> None:
        """JLMS inefficiency estimates u_hat are non-negative."""
        u_hat = fitted_cd_model.get_inefficiency(cd_data.X, cd_data.y)
        assert u_hat.shape == (cd_data.n_obs,)
        assert np.all(u_hat >= 0.0)

    def test_noise_decomposition(
        self, cd_data: SFADataset, fitted_cd_model: ParametricSFA
    ) -> None:
        """For production frontier: residual = v_hat - u_hat."""
        y_hat = fitted_cd_model.predict(cd_data.X)
        residual = cd_data.y - y_hat
        u_hat = fitted_cd_model.get_inefficiency(cd_data.X, cd_data.y)
        v_hat = fitted_cd_model.get_noise(cd_data.X, cd_data.y)

        np.testing.assert_allclose(residual, v_hat - u_hat, atol=1e-8)


# ---------------------------------------------------------------------------
# Summary and model info tests
# ---------------------------------------------------------------------------


class TestSummary:
    """Tests for summary() and log_likelihood()."""

    def test_summary_returns_valid_summary(
        self, cd_data: SFADataset, fitted_cd_model: ParametricSFA
    ) -> None:
        """Summary has all fields populated and AIC/BIC follow formulas."""
        s = fitted_cd_model.summary()

        assert s.n_obs == cd_data.n_obs
        assert s.n_params > 0
        assert s.sigma_v > 0.0
        assert s.sigma_u > 0.0
        assert 0.0 < s.mean_efficiency < 1.0
        assert s.frontier_type == "cobb-douglas"
        assert s.inefficiency_dist == "half-normal"

        expected_aic = -2.0 * s.log_likelihood + 2.0 * s.n_params
        np.testing.assert_allclose(s.aic, expected_aic, rtol=1e-10)

        expected_bic = -2.0 * s.log_likelihood + s.n_params * np.log(s.n_obs)
        np.testing.assert_allclose(s.bic, expected_bic, rtol=1e-10)

    def test_log_likelihood_finite(self, fitted_cd_model: ParametricSFA) -> None:
        """Log-likelihood should be a finite number."""
        ll = fitted_cd_model.log_likelihood()
        assert np.isfinite(ll)

    def test_cost_frontier(self, cost_data: SFADataset) -> None:
        """cost=True flips the sign convention."""
        model = ParametricSFA(
            frontier="cobb-douglas",
            inefficiency="half-normal",
            cost=True,
        )
        model.fit(cost_data.X, cost_data.y)

        te = model.efficiency(cost_data.X, cost_data.y)
        assert np.all(te > 0.0)
        assert np.all(te <= 1.0)

        assert model.sigma_v_ > 0.0
        assert model.sigma_u_ > 0.0


# ---------------------------------------------------------------------------
# All distributions
# ---------------------------------------------------------------------------


class TestDistributions:
    """Tests that all supported inefficiency distributions work."""

    def test_fit_with_truncated_normal(self) -> None:
        """Truncated-normal distribution fits without error."""
        data = simulate_sfa(
            n_obs=1000,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="truncated-normal",
            sigma_v=0.1,
            sigma_u=0.2,
            seed=77,
        )
        model = ParametricSFA(frontier="cobb-douglas", inefficiency="truncated-normal")
        model.fit(data.X, data.y)

        assert model.is_fitted_
        assert model.sigma_v_ > 0.0
        assert model.sigma_u_ > 0.0

        te = model.efficiency(data.X, data.y)
        assert np.all(te > 0.0)
        assert np.all(te <= 1.0)

    def test_fit_with_exponential(self) -> None:
        """Exponential distribution fits and recovers sigma within 30%."""
        data = simulate_sfa(
            n_obs=2000,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="exponential",
            sigma_v=0.1,
            sigma_u=0.2,
            seed=88,
        )
        model = ParametricSFA(frontier="cobb-douglas", inefficiency="exponential")
        model.fit(data.X, data.y)

        assert model.is_fitted_
        assert model.sigma_v_ > 0.0
        assert model.sigma_u_ > 0.0

        te = model.efficiency(data.X, data.y)
        assert np.all(te > 0.0)
        assert np.all(te <= 1.0)

        # Sigma recovery check (detects NLL formula bugs)
        assert abs(model.sigma_u_ - data.sigma_u) / data.sigma_u < 0.30, (
            f"sigma_u: {model.sigma_u_:.4f} vs true {data.sigma_u}"
        )
