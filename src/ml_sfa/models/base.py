"""Base classes for Stochastic Frontier Analysis estimators.

Provides the abstract base estimator ``BaseSFAEstimator`` that all SFA model
implementations must subclass, and the ``SFASummary`` frozen dataclass used to
report fitted-model diagnostics.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Self

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data

from ml_sfa.models._types import FloatArray, FrontierType, InefficiencyType

__all__ = ["BaseSFAEstimator", "SFASummary"]


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SFASummary:
    """Immutable summary of a fitted SFA model.

    Attributes
    ----------
    n_obs : int
        Number of observations used during fitting.
    n_params : int
        Number of estimated parameters.
    log_likelihood : float
        Maximised log-likelihood value.
    aic : float
        Akaike information criterion.
    bic : float
        Bayesian information criterion.
    sigma_v : float
        Estimated standard deviation of the noise component *v*.
    sigma_u : float
        Estimated standard deviation of the inefficiency component *u*.
    mean_efficiency : float
        Mean technical efficiency across observations.
    frontier_type : str
        Frontier function specification (e.g. ``"cobb-douglas"``).
    inefficiency_dist : str
        Assumed distribution of inefficiency (e.g. ``"half-normal"``).
    """

    n_obs: int
    n_params: int
    log_likelihood: float
    aic: float
    bic: float
    sigma_v: float
    sigma_u: float
    mean_efficiency: float
    frontier_type: str
    inefficiency_dist: str


# ---------------------------------------------------------------------------
# Abstract base estimator
# ---------------------------------------------------------------------------


class BaseSFAEstimator(BaseEstimator, abc.ABC):  # type: ignore[misc]
    """Abstract base class for all SFA estimators.

    Inherits from :class:`sklearn.base.BaseEstimator` so that concrete
    subclasses automatically gain ``get_params`` / ``set_params`` and are
    compatible with scikit-learn pipelines, grid search, etc.

    Parameters
    ----------
    frontier : FrontierType
        Functional form of the deterministic frontier.
        One of ``"cobb-douglas"`` or ``"translog"``.
    inefficiency : InefficiencyType
        Distributional assumption for the one-sided inefficiency term *u*.
        One of ``"half-normal"``, ``"truncated-normal"``, or ``"exponential"``.
    cost : bool
        If ``True`` the model is a *cost* frontier (inefficiency increases
        the dependent variable).  Default is ``False`` (production frontier).
    """

    def __init__(
        self,
        frontier: FrontierType = "cobb-douglas",
        inefficiency: InefficiencyType = "half-normal",
        *,
        cost: bool = False,
    ) -> None:
        self.frontier = frontier
        self.inefficiency = inefficiency
        self.cost = cost

    # -- abstract interface -------------------------------------------------

    @abc.abstractmethod
    def fit(self, X: FloatArray, y: FloatArray) -> Self:
        """Fit the model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input matrix.
        y : array-like of shape (n_samples,)
            Observed output (production or cost).

        Returns
        -------
        Self
            The fitted estimator instance.
        """

    @abc.abstractmethod
    def predict(self, X: FloatArray) -> FloatArray:
        """Predict frontier values for *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input matrix.

        Returns
        -------
        FloatArray
            Predicted frontier values of shape (n_samples,).
        """

    @abc.abstractmethod
    def efficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate technical efficiency for each observation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input matrix.
        y : array-like of shape (n_samples,)
            Observed output.

        Returns
        -------
        FloatArray
            Technical efficiency values in [0, 1] of shape (n_samples,).
        """

    @abc.abstractmethod
    def get_inefficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate the one-sided inefficiency component *u* for each observation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input matrix.
        y : array-like of shape (n_samples,)
            Observed output.

        Returns
        -------
        FloatArray
            Estimated inefficiency values of shape (n_samples,).
        """

    @abc.abstractmethod
    def get_noise(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate the symmetric noise component *v* for each observation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input matrix.
        y : array-like of shape (n_samples,)
            Observed output.

        Returns
        -------
        FloatArray
            Estimated noise values of shape (n_samples,).
        """

    @abc.abstractmethod
    def log_likelihood(self) -> float:
        """Return the maximised log-likelihood of the fitted model.

        Returns
        -------
        float
            Log-likelihood value.
        """

    @abc.abstractmethod
    def summary(self) -> SFASummary:
        """Return a summary of the fitted model.

        Returns
        -------
        SFASummary
            Frozen dataclass with model diagnostics.
        """

    # -- concrete helpers ---------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise ``NotFittedError`` if the model has not been fitted yet.

        Uses :func:`sklearn.utils.validation.check_is_fitted` which inspects
        the ``is_fitted_`` attribute (or any trailing-underscore attribute).
        """
        check_is_fitted(self)

    def _validate_data(
        self,
        X: FloatArray,
        y: FloatArray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and convert input arrays using sklearn utilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input matrix.
        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Validated (X, y) as numpy arrays with dtype ``float64``.
        """
        X_out, y_out = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            dtype="float64",
            ensure_2d=True,
            multi_output=False,
        )
        return X_out, y_out
