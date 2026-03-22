"""Tests for BaseSFAEstimator and SFASummary."""

from __future__ import annotations

from typing import Self

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from ml_sfa._types import FloatArray
from ml_sfa.models.base import BaseSFAEstimator, SFASummary


class _ConcreteStub(BaseSFAEstimator):
    """Minimal concrete subclass for testing the abstract base."""

    def fit(self, X: FloatArray, y: FloatArray) -> Self:
        """Fit the stub model by storing coefficients."""
        X_val, y_val = self._validate_data(X, y)
        self.coef_: NDArray[np.floating] = np.linalg.lstsq(X_val, y_val, rcond=None)[0]
        self.n_features_in_ = X_val.shape[1]
        self.sigma_v_: float = 0.1
        self.sigma_u_: float = 0.2
        self.is_fitted_ = True
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        """Predict frontier values."""
        self._check_fitted()
        return np.asarray(X @ self.coef_)

    def efficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Return ones as trivial efficiency estimates."""
        self._check_fitted()
        return np.ones(len(y))

    def get_inefficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Return zeros as trivial inefficiency estimates."""
        self._check_fitted()
        return np.zeros(len(y))

    def get_noise(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Return residuals as noise estimates."""
        self._check_fitted()
        return np.asarray(y - self.predict(X))

    def log_likelihood(self) -> float:
        """Return a dummy log-likelihood value."""
        self._check_fitted()
        return -100.0

    def summary(self) -> SFASummary:
        """Return a summary with dummy values."""
        self._check_fitted()
        return SFASummary(
            n_obs=100,
            n_params=3,
            log_likelihood=self.log_likelihood(),
            aic=206.0,
            bic=214.0,
            sigma_v=self.sigma_v_,
            sigma_u=self.sigma_u_,
            mean_efficiency=1.0,
            frontier_type=self.frontier,
            inefficiency_dist=self.inefficiency,
        )


class TestBaseSFAEstimatorABC:
    """Tests for abstract base class behaviour."""

    def test_cannot_instantiate_base_directly(self) -> None:
        """BaseSFAEstimator is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            BaseSFAEstimator()  # type: ignore[abstract]

    def test_concrete_subclass_sklearn_compatible(self) -> None:
        """A concrete subclass supports get_params / set_params from sklearn."""
        stub = _ConcreteStub(frontier="translog", inefficiency="exponential", cost=True)

        params = stub.get_params()
        assert params["frontier"] == "translog"
        assert params["inefficiency"] == "exponential"
        assert params["cost"] is True

        stub.set_params(frontier="cobb-douglas")
        assert stub.get_params()["frontier"] == "cobb-douglas"


class TestFitPredict:
    """Tests for fit / predict lifecycle."""

    @pytest.fixture()
    def sample_data(self, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
        """Small random dataset for fit/predict tests."""
        X = rng.standard_normal((50, 3))
        y = X @ np.array([1.0, 2.0, 3.0]) + rng.standard_normal(50) * 0.1
        return np.asarray(X), np.asarray(y)

    def test_fit_returns_self(self, sample_data: tuple[FloatArray, FloatArray]) -> None:
        """fit() must return self to support method chaining."""
        X, y = sample_data
        stub = _ConcreteStub()
        result = stub.fit(X, y)
        assert result is stub

    def test_predict_before_fit_raises(
        self, sample_data: tuple[FloatArray, FloatArray]
    ) -> None:
        """Calling predict before fit raises NotFittedError."""
        X, _y = sample_data
        stub = _ConcreteStub()
        with pytest.raises(NotFittedError):
            stub.predict(X)


class TestEfficiency:
    """Tests for efficiency estimation."""

    @pytest.fixture()
    def fitted_stub(
        self, rng: np.random.Generator
    ) -> tuple[_ConcreteStub, FloatArray, FloatArray]:
        """Return a fitted stub together with X, y."""
        X = rng.standard_normal((50, 3))
        y = X @ np.array([1.0, 2.0, 3.0]) + rng.standard_normal(50) * 0.1
        stub = _ConcreteStub()
        stub.fit(np.asarray(X), np.asarray(y))
        return stub, np.asarray(X), np.asarray(y)

    def test_efficiency_returns_0_to_1(
        self, fitted_stub: tuple[_ConcreteStub, FloatArray, FloatArray]
    ) -> None:
        """Technical efficiency values must lie in [0, 1]."""
        stub, X, y = fitted_stub
        te = stub.efficiency(X, y)
        assert np.all(te >= 0.0)
        assert np.all(te <= 1.0)


class TestSummary:
    """Tests for the summary() method and SFASummary dataclass."""

    def test_summary_returns_dataclass(self, rng: np.random.Generator) -> None:
        """summary() returns an SFASummary with all expected fields."""
        X = rng.standard_normal((50, 3))
        y = X @ np.array([1.0, 2.0, 3.0]) + rng.standard_normal(50) * 0.1
        stub = _ConcreteStub()
        stub.fit(np.asarray(X), np.asarray(y))

        result = stub.summary()
        assert isinstance(result, SFASummary)
        assert isinstance(result.n_obs, int)
        assert isinstance(result.n_params, int)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.aic, float)
        assert isinstance(result.bic, float)
        assert isinstance(result.sigma_v, float)
        assert isinstance(result.sigma_u, float)
        assert isinstance(result.mean_efficiency, float)
        assert isinstance(result.frontier_type, str)
        assert isinstance(result.inefficiency_dist, str)


class TestCheckIsFitted:
    """Tests for the is_fitted_ attribute pattern."""

    def test_check_is_fitted_flag(self, rng: np.random.Generator) -> None:
        """is_fitted_ must be False before fit and True after."""
        stub = _ConcreteStub()
        assert not hasattr(stub, "is_fitted_")

        X = rng.standard_normal((20, 2))
        y = X @ np.array([1.0, 2.0]) + 0.1 * rng.standard_normal(20)
        stub.fit(np.asarray(X), np.asarray(y))
        assert stub.is_fitted_ is True
