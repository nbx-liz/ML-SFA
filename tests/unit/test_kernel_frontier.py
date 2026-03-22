"""Unit tests for KernelSFA estimator.

Tests the local polynomial kernel-weighted SFA model including:
- Constructor and parameter handling
- Sklearn compatibility (get_params, set_params)
- NotFittedError before fit
- fit() shapes and attribute ranges
- predict(), efficiency(), get_inefficiency(), get_noise()
- Out-of-sample guard (NotImplementedError)
- summary() and log_likelihood()
- Cost frontier support
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from ml_sfa.data.simulator import simulate_sfa
from ml_sfa.models.kernel_frontier import KernelSFA

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
) -> KernelSFA:
    """A fitted KernelSFA model with small N for speed."""
    X, y = simple_data
    model = KernelSFA(
        inefficiency="half-normal",
        bandwidth="scott",
        kernel="gaussian",
        seed=42,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Constructor / sklearn compatibility
# ---------------------------------------------------------------------------


class TestKernelSFAConstructor:
    """Tests for constructor and sklearn API."""

    def test_default_params(self) -> None:
        """Default parameter values are set correctly."""
        model = KernelSFA()
        params = model.get_params()
        assert params["inefficiency"] == "half-normal"
        assert params["cost"] is False
        assert params["bandwidth"] == "scott"
        assert params["kernel"] == "gaussian"
        assert params["seed"] is None

    def test_custom_params(self) -> None:
        """Custom parameters are preserved."""
        model = KernelSFA(
            inefficiency="half-normal",
            cost=True,
            bandwidth=0.5,
            kernel="gaussian",
            seed=123,
        )
        assert model.cost is True
        assert model.bandwidth == 0.5
        assert model.seed == 123

    def test_set_params(self) -> None:
        """set_params works for sklearn compatibility."""
        model = KernelSFA()
        model.set_params(cost=True, bandwidth=1.0)
        assert model.cost is True
        assert model.bandwidth == 1.0

    def test_get_params_deep(self) -> None:
        """get_params(deep=True) returns all parameters."""
        model = KernelSFA(bandwidth=0.5, seed=42)
        params = model.get_params(deep=True)
        assert "bandwidth" in params
        assert "seed" in params
        assert "kernel" in params


# ---------------------------------------------------------------------------
# NotFittedError
# ---------------------------------------------------------------------------


class TestNotFitted:
    """Methods raise NotFittedError before fit."""

    def test_predict_not_fitted(self) -> None:
        """predict raises NotFittedError."""
        model = KernelSFA()
        X = np.random.default_rng(0).normal(size=(10, 2))
        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_efficiency_not_fitted(self) -> None:
        """efficiency raises NotFittedError."""
        model = KernelSFA()
        X = np.random.default_rng(0).normal(size=(10, 2))
        y = np.random.default_rng(0).normal(size=10)
        with pytest.raises(NotFittedError):
            model.efficiency(X, y)

    def test_get_inefficiency_not_fitted(self) -> None:
        """get_inefficiency raises NotFittedError."""
        model = KernelSFA()
        X = np.random.default_rng(0).normal(size=(10, 2))
        y = np.random.default_rng(0).normal(size=10)
        with pytest.raises(NotFittedError):
            model.get_inefficiency(X, y)

    def test_get_noise_not_fitted(self) -> None:
        """get_noise raises NotFittedError."""
        model = KernelSFA()
        X = np.random.default_rng(0).normal(size=(10, 2))
        y = np.random.default_rng(0).normal(size=10)
        with pytest.raises(NotFittedError):
            model.get_noise(X, y)

    def test_log_likelihood_not_fitted(self) -> None:
        """log_likelihood raises NotFittedError."""
        model = KernelSFA()
        with pytest.raises(NotFittedError):
            model.log_likelihood()

    def test_summary_not_fitted(self) -> None:
        """summary raises NotFittedError."""
        model = KernelSFA()
        with pytest.raises(NotFittedError):
            model.summary()


# ---------------------------------------------------------------------------
# Fit and output shapes
# ---------------------------------------------------------------------------


class TestFit:
    """Tests for fit method and fitted attributes."""

    def test_fit_returns_self(self, simple_data: tuple[np.ndarray, np.ndarray]) -> None:
        """fit() returns self for method chaining."""
        X, y = simple_data
        model = KernelSFA(seed=0)
        result = model.fit(X, y)
        assert result is model

    def test_fitted_attributes(self, fitted_model: KernelSFA) -> None:
        """Fitted model has required attributes."""
        assert fitted_model.is_fitted_
        assert fitted_model.sigma_v_ > 0
        assert fitted_model.sigma_u_ > 0
        assert isinstance(fitted_model.log_likelihood_, float)
        assert fitted_model.n_features_in_ == 2

    def test_predict_shape(
        self,
        fitted_model: KernelSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """predict returns correct shape."""
        X, _ = simple_data
        pred = fitted_model.predict(X)
        assert pred.shape == (X.shape[0],)

    def test_efficiency_shape_and_range(
        self,
        fitted_model: KernelSFA,
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
        fitted_model: KernelSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """get_inefficiency returns non-negative values."""
        X, y = simple_data
        u_hat = fitted_model.get_inefficiency(X, y)
        assert u_hat.shape == (X.shape[0],)
        assert np.all(u_hat >= 0)

    def test_noise_decomposition(
        self,
        fitted_model: KernelSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """v_hat = epsilon + u_hat for production frontier."""
        X, y = simple_data
        pred = fitted_model.predict(X)
        epsilon = y - pred
        u_hat = fitted_model.get_inefficiency(X, y)
        v_hat = fitted_model.get_noise(X, y)
        # Production: epsilon = v - u => v = epsilon + u
        np.testing.assert_allclose(v_hat, epsilon + u_hat, atol=1e-6)

    def test_predict_oos_raises(
        self,
        fitted_model: KernelSFA,
    ) -> None:
        """Out-of-sample prediction raises NotImplementedError."""
        X_new = np.random.default_rng(99).normal(size=(5, 2))
        with pytest.raises(NotImplementedError, match="out-of-sample"):
            fitted_model.predict(X_new)

    def test_numeric_bandwidth(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Model fits with a numeric bandwidth."""
        X, y = simple_data
        model = KernelSFA(bandwidth=1.0, seed=0)
        model.fit(X, y)
        assert model.is_fitted_
        assert model.sigma_v_ > 0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    """Tests for summary and log_likelihood."""

    def test_summary_fields(self, fitted_model: KernelSFA) -> None:
        """Summary contains all required fields."""
        s = fitted_model.summary()
        assert s.n_obs == 100
        assert s.n_params > 0
        assert isinstance(s.log_likelihood, float)
        assert s.sigma_v > 0
        assert s.sigma_u > 0
        assert 0 < s.mean_efficiency <= 1.0
        assert s.frontier_type == "kernel"
        assert s.inefficiency_dist == "half-normal"

    def test_log_likelihood_matches_summary(self, fitted_model: KernelSFA) -> None:
        """log_likelihood() matches summary().log_likelihood."""
        ll = fitted_model.log_likelihood()
        s = fitted_model.summary()
        assert ll == pytest.approx(s.log_likelihood, rel=1e-10)

    def test_aic_bic_consistency(self, fitted_model: KernelSFA) -> None:
        """AIC = -2*LL + 2*k, BIC = -2*LL + k*ln(n)."""
        s = fitted_model.summary()
        expected_aic = -2.0 * s.log_likelihood + 2.0 * s.n_params
        expected_bic = -2.0 * s.log_likelihood + s.n_params * np.log(s.n_obs)
        assert s.aic == pytest.approx(expected_aic, rel=1e-10)
        assert s.bic == pytest.approx(expected_bic, rel=1e-10)


# ---------------------------------------------------------------------------
# Sklearn compatibility
# ---------------------------------------------------------------------------


class TestSklearnCompat:
    """Tests for sklearn estimator protocol compliance."""

    def test_clone(self) -> None:
        """sklearn clone produces equivalent unfitted estimator."""
        from sklearn.base import clone

        model = KernelSFA(bandwidth=0.5, seed=42)
        cloned = clone(model)
        assert cloned.get_params() == model.get_params()

    def test_repr(self) -> None:
        """repr is a valid string."""
        model = KernelSFA()
        r = repr(model)
        assert "KernelSFA" in r


# ---------------------------------------------------------------------------
# Cost frontier
# ---------------------------------------------------------------------------


class TestCostFrontier:
    """Test cost frontier mode."""

    def test_cost_frontier_fit(self) -> None:
        """Model fits in cost frontier mode."""
        ds = simulate_sfa(
            n_obs=80,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.2,
            sigma_u=0.3,
            cost=True,
            seed=42,
        )
        model = KernelSFA(
            inefficiency="half-normal",
            cost=True,
            seed=42,
        )
        model.fit(ds.X, ds.y)
        assert model.is_fitted_

        te = model.efficiency(ds.X, ds.y)
        assert np.all(te > 0)
        assert np.all(te <= 1.0 + 1e-6)

    def test_cost_noise_decomposition(self) -> None:
        """v_hat = epsilon - u_hat for cost frontier."""
        ds = simulate_sfa(
            n_obs=80,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.2,
            sigma_u=0.3,
            cost=True,
            seed=42,
        )
        model = KernelSFA(cost=True, seed=42)
        model.fit(ds.X, ds.y)

        pred = model.predict(ds.X)
        epsilon = ds.y - pred
        u_hat = model.get_inefficiency(ds.X, ds.y)
        v_hat = model.get_noise(ds.X, ds.y)
        # Cost: epsilon = v + u => v = epsilon - u
        np.testing.assert_allclose(v_hat, epsilon - u_hat, atol=1e-6)
