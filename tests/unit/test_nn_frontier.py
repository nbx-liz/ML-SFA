"""Unit tests for NNFrontierSFA estimator.

Tests the joint estimation neural network SFA model including:
- Constructor and parameter handling
- Sklearn compatibility (get_params, set_params)
- fit() with MSE pretraining + SFA fine-tuning
- predict(), efficiency(), get_inefficiency(), get_noise()
- summary() and log_likelihood()
- Multiple initializations for non-convex optimization
- All three inefficiency distributions
- Cost frontier support
- NotFittedError before fit
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

torch = pytest.importorskip("torch")

from ml_sfa.data.simulator import simulate_sfa  # noqa: E402
from ml_sfa.models.nn_frontier import NNFrontierSFA  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Small simulated dataset for quick tests."""
    ds = simulate_sfa(
        n_obs=200,
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
) -> NNFrontierSFA:
    """A fitted NNFrontierSFA model."""
    X, y = simple_data
    model = NNFrontierSFA(
        inefficiency="half-normal",
        hidden_dims=[16, 8],
        pretrain_epochs=50,
        finetune_epochs=5,
        n_inits=1,
        seed=42,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Constructor / sklearn compatibility
# ---------------------------------------------------------------------------


class TestNNFrontierSFAConstructor:
    """Tests for constructor and sklearn API."""

    def test_default_params(self) -> None:
        """Default parameter values are set correctly."""
        model = NNFrontierSFA()
        params = model.get_params()
        assert params["inefficiency"] == "half-normal"
        assert params["cost"] is False
        assert params["hidden_dims"] == [64, 32]
        assert params["monotonic"] is True

    def test_custom_params(self) -> None:
        """Custom parameters are preserved."""
        model = NNFrontierSFA(
            inefficiency="exponential",
            cost=True,
            hidden_dims=[32, 16],
            monotonic=False,
            pretrain_epochs=100,
            finetune_epochs=10,
            n_inits=3,
        )
        assert model.inefficiency == "exponential"
        assert model.cost is True
        assert model.hidden_dims == [32, 16]
        assert model.monotonic is False

    def test_set_params(self) -> None:
        """set_params works for sklearn compatibility."""
        model = NNFrontierSFA()
        model.set_params(inefficiency="exponential", cost=True)
        assert model.inefficiency == "exponential"
        assert model.cost is True

    def test_get_params_deep(self) -> None:
        """get_params(deep=True) returns all parameters."""
        model = NNFrontierSFA(hidden_dims=[32], n_inits=5)
        params = model.get_params(deep=True)
        assert "hidden_dims" in params
        assert "n_inits" in params


# ---------------------------------------------------------------------------
# NotFittedError
# ---------------------------------------------------------------------------


class TestNotFitted:
    """Methods raise NotFittedError before fit."""

    def test_predict_not_fitted(self) -> None:
        """predict raises NotFittedError."""
        model = NNFrontierSFA()
        X = np.random.default_rng(0).normal(size=(10, 2))
        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_efficiency_not_fitted(self) -> None:
        """efficiency raises NotFittedError."""
        model = NNFrontierSFA()
        X = np.random.default_rng(0).normal(size=(10, 2))
        y = np.random.default_rng(0).normal(size=10)
        with pytest.raises(NotFittedError):
            model.efficiency(X, y)

    def test_log_likelihood_not_fitted(self) -> None:
        """log_likelihood raises NotFittedError."""
        model = NNFrontierSFA()
        with pytest.raises(NotFittedError):
            model.log_likelihood()

    def test_summary_not_fitted(self) -> None:
        """summary raises NotFittedError."""
        model = NNFrontierSFA()
        with pytest.raises(NotFittedError):
            model.summary()


# ---------------------------------------------------------------------------
# fit and output shapes
# ---------------------------------------------------------------------------


class TestFit:
    """Tests for fit method and fitted attributes."""

    def test_fit_returns_self(self, simple_data: tuple[np.ndarray, np.ndarray]) -> None:
        """fit() returns self for method chaining."""
        X, y = simple_data
        model = NNFrontierSFA(
            hidden_dims=[8],
            pretrain_epochs=10,
            finetune_epochs=2,
            n_inits=1,
            seed=0,
        )
        result = model.fit(X, y)
        assert result is model

    def test_fitted_attributes(self, fitted_model: NNFrontierSFA) -> None:
        """Fitted model has required attributes."""
        assert fitted_model.is_fitted_
        assert fitted_model.sigma_v_ > 0
        assert fitted_model.sigma_u_ > 0
        assert isinstance(fitted_model.log_likelihood_, float)
        assert fitted_model.n_features_in_ == 2

    def test_predict_shape(
        self,
        fitted_model: NNFrontierSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """predict returns correct shape."""
        X, _ = simple_data
        pred = fitted_model.predict(X)
        assert pred.shape == (X.shape[0],)

    def test_efficiency_shape_and_range(
        self,
        fitted_model: NNFrontierSFA,
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
        fitted_model: NNFrontierSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """get_inefficiency returns non-negative values."""
        X, y = simple_data
        u_hat = fitted_model.get_inefficiency(X, y)
        assert u_hat.shape == (X.shape[0],)
        assert np.all(u_hat >= 0)

    def test_noise_decomposition(
        self,
        fitted_model: NNFrontierSFA,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """v_hat + u_hat (or v_hat - u_hat) reconstructs epsilon."""
        X, y = simple_data
        pred = fitted_model.predict(X)
        epsilon = y - pred
        u_hat = fitted_model.get_inefficiency(X, y)
        v_hat = fitted_model.get_noise(X, y)
        # Production: epsilon = v - u → v = epsilon + u
        np.testing.assert_allclose(v_hat, epsilon + u_hat, atol=1e-6)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    """Tests for summary and log_likelihood."""

    def test_summary_fields(self, fitted_model: NNFrontierSFA) -> None:
        """Summary contains all required fields."""
        s = fitted_model.summary()
        assert s.n_obs == 200
        assert s.n_params > 0
        assert isinstance(s.log_likelihood, float)
        assert s.aic > 0
        assert s.bic > 0
        assert s.sigma_v > 0
        assert s.sigma_u > 0
        assert s.mean_efficiency > 0
        assert s.mean_efficiency <= 1.0
        assert s.frontier_type == "nn"
        assert s.inefficiency_dist == "half-normal"

    def test_log_likelihood_matches_summary(self, fitted_model: NNFrontierSFA) -> None:
        """log_likelihood() matches summary().log_likelihood."""
        ll = fitted_model.log_likelihood()
        s = fitted_model.summary()
        assert ll == pytest.approx(s.log_likelihood, rel=1e-10)

    def test_aic_bic_consistency(self, fitted_model: NNFrontierSFA) -> None:
        """AIC = -2*LL + 2*k, BIC = -2*LL + k*ln(n)."""
        s = fitted_model.summary()
        expected_aic = -2.0 * s.log_likelihood + 2.0 * s.n_params
        expected_bic = -2.0 * s.log_likelihood + s.n_params * np.log(s.n_obs)
        assert s.aic == pytest.approx(expected_aic, rel=1e-10)
        assert s.bic == pytest.approx(expected_bic, rel=1e-10)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


class TestDistributions:
    """Test with different inefficiency distributions."""

    @pytest.mark.parametrize("dist", ["half-normal", "truncated-normal", "exponential"])
    def test_fit_all_distributions(
        self,
        dist: str,
        simple_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Model fits successfully with each distribution."""
        X, y = simple_data
        model = NNFrontierSFA(
            inefficiency=dist,  # type: ignore[arg-type]
            hidden_dims=[8],
            pretrain_epochs=20,
            finetune_epochs=3,
            n_inits=1,
            seed=42,
        )
        model.fit(X, y)
        assert model.is_fitted_
        assert model.sigma_v_ > 0
        assert model.sigma_u_ > 0

        te = model.efficiency(X, y)
        assert np.all(te > 0)
        assert np.all(te <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Cost frontier
# ---------------------------------------------------------------------------


class TestCostFrontier:
    """Test cost frontier mode."""

    def test_cost_frontier_fit(self) -> None:
        """Model fits in cost frontier mode."""
        ds = simulate_sfa(
            n_obs=200,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.2,
            sigma_u=0.3,
            cost=True,
            seed=42,
        )
        model = NNFrontierSFA(
            inefficiency="half-normal",
            cost=True,
            hidden_dims=[8],
            pretrain_epochs=20,
            finetune_epochs=3,
            n_inits=1,
            seed=42,
        )
        model.fit(ds.X, ds.y)
        assert model.is_fitted_

        te = model.efficiency(ds.X, ds.y)
        assert np.all(te > 0)
        assert np.all(te <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Seed-based reproducibility."""

    def test_same_seed_same_result(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Same seed produces identical results."""
        X, y = simple_data
        kwargs = dict(
            hidden_dims=[8],
            pretrain_epochs=20,
            finetune_epochs=3,
            n_inits=1,
            seed=123,
        )
        m1 = NNFrontierSFA(**kwargs).fit(X, y)  # type: ignore[arg-type]
        m2 = NNFrontierSFA(**kwargs).fit(X, y)  # type: ignore[arg-type]
        np.testing.assert_allclose(m1.predict(X), m2.predict(X), atol=1e-6)
