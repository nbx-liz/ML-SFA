"""Unit tests for BARTFrontierSFA estimator.

Tests the BART-based SFA model including:
- Constructor and parameter handling
- Sklearn compatibility (get_params, set_params)
- fit() with PyMC MCMC sampling
- predict(), efficiency(), get_inefficiency(), get_noise()
- summary() and log_likelihood()
- credible_interval() for Bayesian uncertainty
- NotFittedError before fit

Note: Tests use minimal MCMC settings (few draws, 1 chain) for speed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

pymc = pytest.importorskip("pymc")

from ml_sfa.data.simulator import simulate_sfa  # noqa: E402
from ml_sfa.models.bart_frontier import BARTFrontierSFA  # noqa: E402

# Minimal MCMC settings for fast tests
_FAST_KWARGS = dict(
    n_trees=10,
    n_draws=50,
    n_tune=50,
    n_chains=1,
    seed=42,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Small simulated dataset for quick tests."""
    ds = simulate_sfa(
        n_obs=100,
        n_inputs=2,
        frontier_type="cobb-douglas",
        inefficiency_dist="half-normal",
        sigma_v=0.2,
        sigma_u=0.3,
        seed=42,
    )
    return ds.X, ds.y


@pytest.fixture()
def fitted_model(
    simple_data: tuple[np.ndarray, np.ndarray],
) -> BARTFrontierSFA:
    """A fitted BARTFrontierSFA model with minimal MCMC."""
    X, y = simple_data
    model = BARTFrontierSFA(**_FAST_KWARGS)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Constructor / sklearn compatibility
# ---------------------------------------------------------------------------


class TestBARTFrontierSFAConstructor:
    """Tests for constructor and sklearn API."""

    def test_default_params(self) -> None:
        """Default parameter values are set correctly."""
        model = BARTFrontierSFA()
        params = model.get_params()
        assert params["inefficiency"] == "half-normal"
        assert params["cost"] is False
        assert params["n_trees"] == 50
        assert params["n_draws"] == 2000
        assert params["n_tune"] == 1000
        assert params["n_chains"] == 4

    def test_custom_params(self) -> None:
        """Custom parameters are preserved."""
        model = BARTFrontierSFA(
            inefficiency="exponential",
            cost=True,
            n_trees=100,
            n_draws=500,
            n_tune=200,
            n_chains=2,
        )
        assert model.inefficiency == "exponential"
        assert model.cost is True
        assert model.n_trees == 100
        assert model.n_draws == 500

    def test_set_params(self) -> None:
        """set_params works for sklearn compatibility."""
        model = BARTFrontierSFA()
        model.set_params(inefficiency="exponential", n_trees=100)
        assert model.inefficiency == "exponential"
        assert model.n_trees == 100

    def test_get_params_deep(self) -> None:
        """get_params(deep=True) returns all parameters."""
        model = BARTFrontierSFA(n_trees=30, n_chains=2)
        params = model.get_params(deep=True)
        assert "n_trees" in params
        assert "n_chains" in params


# ---------------------------------------------------------------------------
# NotFittedError
# ---------------------------------------------------------------------------


class TestNotFitted:
    """Methods raise NotFittedError before fit."""

    def test_predict_not_fitted(self) -> None:
        """predict raises NotFittedError."""
        model = BARTFrontierSFA()
        X = np.random.default_rng(0).normal(size=(10, 2))
        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_efficiency_not_fitted(self) -> None:
        """efficiency raises NotFittedError."""
        model = BARTFrontierSFA()
        X = np.random.default_rng(0).normal(size=(10, 2))
        y = np.random.default_rng(0).normal(size=10)
        with pytest.raises(NotFittedError):
            model.efficiency(X, y)

    def test_log_likelihood_not_fitted(self) -> None:
        """log_likelihood raises NotFittedError."""
        model = BARTFrontierSFA()
        with pytest.raises(NotFittedError):
            model.log_likelihood()

    def test_summary_not_fitted(self) -> None:
        """summary raises NotFittedError."""
        model = BARTFrontierSFA()
        with pytest.raises(NotFittedError):
            model.summary()

    def test_credible_interval_not_fitted(self) -> None:
        """credible_interval raises NotFittedError."""
        model = BARTFrontierSFA()
        X = np.random.default_rng(0).normal(size=(10, 2))
        y = np.random.default_rng(0).normal(size=10)
        with pytest.raises(NotFittedError):
            model.credible_interval(X, y)


# ---------------------------------------------------------------------------
# fit and output shapes
# ---------------------------------------------------------------------------


class TestFit:
    """Tests for fit method and fitted attributes."""

    def test_fit_returns_self(self, simple_data: tuple[np.ndarray, np.ndarray]) -> None:
        """fit() returns self for method chaining."""
        X, y = simple_data
        model = BARTFrontierSFA(**_FAST_KWARGS)
        result = model.fit(X, y)
        assert result is model

    def test_fitted_attributes(self, fitted_model: BARTFrontierSFA) -> None:
        """Fitted model has required attributes."""
        assert fitted_model.is_fitted_
        assert fitted_model.sigma_v_ > 0
        assert fitted_model.sigma_u_ > 0
        assert isinstance(fitted_model.log_likelihood_, float)
        assert fitted_model.n_features_in_ == 2

    def test_predict_shape(
        self,
        fitted_model: BARTFrontierSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """predict returns correct shape."""
        X, _ = simple_data
        pred = fitted_model.predict(X)
        assert pred.shape == (X.shape[0],)

    def test_efficiency_shape_and_range(
        self,
        fitted_model: BARTFrontierSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """efficiency returns values in (0, 1]."""
        X, y = simple_data
        te = fitted_model.efficiency(X, y)
        assert te.shape == (X.shape[0],)
        assert np.all(te > 0)
        assert np.all(te <= 1.0 + 1e-6)

    def test_get_inefficiency_non_negative(
        self,
        fitted_model: BARTFrontierSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """get_inefficiency returns non-negative values."""
        X, y = simple_data
        u_hat = fitted_model.get_inefficiency(X, y)
        assert u_hat.shape == (X.shape[0],)
        assert np.all(u_hat >= 0)

    def test_noise_decomposition(
        self,
        fitted_model: BARTFrontierSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """v_hat = epsilon + u_hat for production frontier."""
        X, y = simple_data
        pred = fitted_model.predict(X)
        epsilon = y - pred
        u_hat = fitted_model.get_inefficiency(X, y)
        v_hat = fitted_model.get_noise(X, y)
        np.testing.assert_allclose(v_hat, epsilon + u_hat, atol=1e-6)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    """Tests for summary and log_likelihood."""

    def test_summary_fields(self, fitted_model: BARTFrontierSFA) -> None:
        """Summary contains all required fields."""
        s = fitted_model.summary()
        assert s.n_obs == 100
        assert s.n_params > 0
        assert isinstance(s.log_likelihood, float)
        assert np.isfinite(s.aic)
        assert np.isfinite(s.bic)
        assert s.sigma_v > 0
        assert s.sigma_u > 0
        assert s.mean_efficiency > 0
        assert s.mean_efficiency <= 1.0
        assert s.frontier_type == "bart"
        assert s.inefficiency_dist == "half-normal"

    def test_log_likelihood_matches_summary(
        self, fitted_model: BARTFrontierSFA
    ) -> None:
        """log_likelihood() matches summary().log_likelihood."""
        ll = fitted_model.log_likelihood()
        s = fitted_model.summary()
        assert ll == pytest.approx(s.log_likelihood, rel=1e-10)


# ---------------------------------------------------------------------------
# Credible intervals (Bayesian uncertainty)
# ---------------------------------------------------------------------------


class TestCredibleInterval:
    """Tests for Bayesian TE credible intervals."""

    def test_credible_interval_shape(
        self,
        fitted_model: BARTFrontierSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """credible_interval returns (lower, upper) with correct shape."""
        X, y = simple_data
        lower, upper = fitted_model.credible_interval(X, y)
        assert lower.shape == (X.shape[0],)
        assert upper.shape == (X.shape[0],)

    def test_credible_interval_bounds(
        self,
        fitted_model: BARTFrontierSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Lower < mean TE < upper for credible intervals."""
        X, y = simple_data
        te = fitted_model.efficiency(X, y)
        lower, upper = fitted_model.credible_interval(X, y)
        assert np.all(lower <= te + 1e-6)
        assert np.all(upper >= te - 1e-6)
        assert np.all(lower >= 0)
        assert np.all(upper <= 1.0 + 1e-6)

    def test_credible_interval_custom_alpha(
        self,
        fitted_model: BARTFrontierSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Wider interval for higher confidence level."""
        X, y = simple_data
        lower_90, upper_90 = fitted_model.credible_interval(X, y, alpha=0.10)
        lower_50, upper_50 = fitted_model.credible_interval(X, y, alpha=0.50)
        # 90% CI should be wider than 50% CI on average
        width_90 = np.mean(upper_90 - lower_90)
        width_50 = np.mean(upper_50 - lower_50)
        assert width_90 >= width_50 - 1e-6

    def test_credible_interval_invalid_alpha(
        self,
        fitted_model: BARTFrontierSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Invalid alpha raises ValueError."""
        X, y = simple_data
        with pytest.raises(ValueError, match="alpha must be in"):
            fitted_model.credible_interval(X, y, alpha=1.5)
        with pytest.raises(ValueError, match="alpha must be in"):
            fitted_model.credible_interval(X, y, alpha=0.0)


# ---------------------------------------------------------------------------
# Out-of-sample guard
# ---------------------------------------------------------------------------


class TestOutOfSample:
    """Test OOS prediction raises NotImplementedError."""

    def test_predict_oos_raises(self, fitted_model: BARTFrontierSFA) -> None:
        """predict with different data raises NotImplementedError."""
        X_new = np.random.default_rng(99).normal(size=(5, 2))
        with pytest.raises(NotImplementedError, match="Out-of-sample"):
            fitted_model.predict(X_new)


# ---------------------------------------------------------------------------
# Sklearn compatibility
# ---------------------------------------------------------------------------


class TestSklearnCompat:
    """Test sklearn API compatibility."""

    def test_clone(self) -> None:
        """sklearn clone works on BARTFrontierSFA."""
        from sklearn.base import clone

        model = BARTFrontierSFA(n_trees=30, n_draws=100)
        cloned = clone(model)
        assert cloned.get_params() == model.get_params()
        assert cloned is not model
