"""BART-based Stochastic Frontier Analysis estimator.

Implements ``BARTFrontierSFA``, which models the production frontier using
Bayesian Additive Regression Trees (BART) and simultaneously estimates
inefficiency via a data-augmentation Gibbs sampler in PyMC.

The Bayesian framework naturally provides posterior distributions for
technical efficiency, enabling credible interval estimation.

Requires ``pymc`` and ``pymc-bart`` (install via ``pip install ml-sfa[bart]``).
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pymc as pm
import pymc_bart as pmb
from scipy.special import log_ndtr

from ml_sfa._types import FloatArray, InefficiencyType
from ml_sfa.models.base import BaseSFAEstimator, SFASummary
from ml_sfa.utils.distributions import (
    Exponential,
    HalfNormal,
    InefficiencyDistribution,
    TruncatedNormal,
)

__all__ = ["BARTFrontierSFA"]

_FRONTIER_TYPE = "bart"


class BARTFrontierSFA(BaseSFAEstimator):
    """BART-based Stochastic Frontier Analysis estimator.

    Uses PyMC-BART for nonparametric frontier estimation with
    data-augmented MCMC inference for simultaneous error decomposition.

    Parameters
    ----------
    inefficiency : InefficiencyType
        Inefficiency distribution: ``"half-normal"``, ``"truncated-normal"``,
        or ``"exponential"``.
    cost : bool
        If ``True``, estimate a cost frontier.
    n_trees : int
        Number of BART trees.
    n_draws : int
        Number of posterior draws (after tuning).
    n_tune : int
        Number of tuning (burn-in) steps.
    n_chains : int
        Number of MCMC chains.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        inefficiency: InefficiencyType = "half-normal",
        *,
        cost: bool = False,
        n_trees: int = 50,
        n_draws: int = 2000,
        n_tune: int = 1000,
        n_chains: int = 4,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            frontier="cobb-douglas",  # placeholder
            inefficiency=inefficiency,
            cost=cost,
        )
        self.n_trees = n_trees
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.seed = seed

    # -- private helpers ----------------------------------------------------

    def _get_distribution(self) -> InefficiencyDistribution:
        """Return the distribution object for JLMS estimation."""
        if self.inefficiency == "half-normal":
            return HalfNormal()
        if self.inefficiency == "truncated-normal":
            return TruncatedNormal(mu=0.0)
        if self.inefficiency == "exponential":
            return Exponential()
        msg = f"Unsupported inefficiency distribution: {self.inefficiency!r}"
        raise ValueError(msg)

    def _build_model(self, X: np.ndarray, y: np.ndarray) -> pm.Model:
        """Build the PyMC model with BART frontier and SFA error structure.

        Uses data-augmentation approach: latent u_i is explicitly sampled.

        Parameters
        ----------
        X : np.ndarray
            Input matrix of shape ``(n, p)``.
        y : np.ndarray
            Observed output of shape ``(n,)``.

        Returns
        -------
        pm.Model
            The PyMC model ready for sampling.
        """
        n = X.shape[0]
        sign = 1.0 if self.cost else -1.0

        with pm.Model() as model:
            # Variance priors
            sigma_v = pm.HalfNormal("sigma_v", sigma=1.0)
            sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)

            # BART frontier
            mu = pmb.BART("mu", X=X, Y=y, m=self.n_trees)

            # Latent inefficiency
            if self.inefficiency == "half-normal":
                u = pm.HalfNormal("u", sigma=sigma_u, shape=n)
            elif self.inefficiency == "exponential":
                u = pm.Exponential("u", lam=1.0 / sigma_u, shape=n)
            elif self.inefficiency == "truncated-normal":
                u = pm.TruncatedNormal("u", mu=0.0, sigma=sigma_u, lower=0.0, shape=n)
            else:
                msg = f"Unsupported: {self.inefficiency!r}"
                raise ValueError(msg)

            # Observation: y = mu + sign*u + v
            # Production: y = f(x) - u + v → mu - u
            # Cost:       y = f(x) + u + v → mu + u
            pm.Normal("y_obs", mu=mu + sign * u, sigma=sigma_v, observed=y)

        return model

    # -- public interface ---------------------------------------------------

    def fit(self, X: FloatArray, y: FloatArray) -> Self:
        """Fit the BART-SFA model via MCMC.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        Self
            The fitted estimator.
        """
        X_val, y_val = self._validate_data(X, y)
        n_obs = X_val.shape[0]

        model = self._build_model(X_val, y_val)

        random_seed = self.seed

        with model:
            trace = pm.sample(
                draws=self.n_draws,
                tune=self.n_tune,
                chains=self.n_chains,
                cores=1,
                random_seed=random_seed,
                progressbar=False,
            )

        # Store results
        self._trace = trace
        self._X_train = X_val.copy()
        self._y_train = y_val.copy()

        # Extract posterior means
        posterior = trace.posterior
        self.sigma_v_ = float(posterior["sigma_v"].mean().values)
        self.sigma_u_ = float(posterior["sigma_u"].mean().values)

        # Frontier posterior mean
        self._mu_posterior_mean: FloatArray = (
            posterior["mu"].mean(dim=("chain", "draw")).values
        )

        # u posterior for TE computation
        u_posterior = posterior["u"].values  # (chains, draws, n)
        self._u_posterior = u_posterior.reshape(-1, n_obs)  # (total_draws, n)
        te_posterior = np.exp(-self._u_posterior)
        self._te_posterior = te_posterior

        # Approximate log-likelihood (plug-in estimate using posterior means).
        # Not a proper Bayesian marginal likelihood; AIC/BIC from summary()
        # should be interpreted with caution for cross-model comparison.
        eps = y_val - self._mu_posterior_mean
        sigma_sq = self.sigma_v_**2 + self.sigma_u_**2
        sigma = np.sqrt(sigma_sq)
        lam = self.sigma_u_ / self.sigma_v_
        ll = float(
            -n_obs * np.log(sigma)
            + np.sum(log_ndtr(-eps * lam / sigma))
            - np.sum(eps**2) / (2.0 * sigma_sq)
            + 0.5 * n_obs * np.log(2.0 / np.pi)
        )
        self.log_likelihood_ = ll

        self.n_features_in_ = X_val.shape[1]
        self.is_fitted_ = True
        self._n_obs = n_obs

        # Precompute mean efficiency
        self._mean_efficiency = float(np.mean(te_posterior))

        return self

    def predict(self, X: FloatArray) -> FloatArray:
        """Predict frontier values (posterior mean).

        For in-sample data, returns the stored posterior mean.
        For out-of-sample, uses JLMS with posterior mean sigmas.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        FloatArray
            Predicted frontier values of shape ``(n_samples,)``.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)

        # Check if in-sample (same data as training)
        if X_arr.shape == self._X_train.shape and np.allclose(X_arr, self._X_train):
            return self._mu_posterior_mean.copy()

        # Out-of-sample prediction not yet supported for BART
        msg = (
            "Out-of-sample prediction is not yet supported for "
            "BARTFrontierSFA. Pass the training data."
        )
        raise NotImplementedError(msg)

    def efficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate technical efficiency (posterior mean).

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        FloatArray
            Technical efficiency in ``(0, 1]`` of shape ``(n_samples,)``.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        # In-sample: use direct posterior mean of exp(-u)
        if (
            X_arr.shape == self._X_train.shape
            and np.allclose(X_arr, self._X_train)
            and np.allclose(y_arr, self._y_train)
        ):
            result: FloatArray = np.mean(self._te_posterior, axis=0)
            return result

        # Out-of-sample: JLMS with posterior mean sigmas
        u_hat = self.get_inefficiency(X_arr, y_arr)
        te: FloatArray = np.exp(-u_hat)
        return te

    def get_inefficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate inefficiency u (posterior mean or JLMS).

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        FloatArray
            Non-negative inefficiency estimates of shape ``(n_samples,)``.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        # In-sample: posterior mean of u
        if (
            X_arr.shape == self._X_train.shape
            and np.allclose(X_arr, self._X_train)
            and np.allclose(y_arr, self._y_train)
        ):
            u_mean: FloatArray = np.mean(self._u_posterior, axis=0)
            return u_mean

        # Out-of-sample: JLMS
        pred = self.predict(X_arr)
        epsilon = y_arr - pred
        dist = self._get_distribution()
        if self.cost:
            return dist.conditional_mean(-epsilon, self.sigma_v_, self.sigma_u_)
        return dist.conditional_mean(epsilon, self.sigma_v_, self.sigma_u_)

    def get_noise(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate noise component v.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        FloatArray
            Estimated noise values of shape ``(n_samples,)``.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        pred = self.predict(X_arr)
        epsilon = y_arr - pred
        u_hat = self.get_inefficiency(X_arr, y_arr)

        if self.cost:
            return epsilon - u_hat
        return epsilon + u_hat

    def log_likelihood(self) -> float:
        """Return the approximate log-likelihood.

        Returns
        -------
        float
            Log-likelihood computed from posterior mean estimates.
        """
        self._check_fitted()
        return self.log_likelihood_

    def summary(self) -> SFASummary:
        """Return a summary of the fitted model.

        Returns
        -------
        SFASummary
            Frozen dataclass with model diagnostics.
        """
        self._check_fitted()

        # Rough approximation: one parameter per tree + sigma_v, sigma_u.
        # BART's effective DoF is better captured by WAIC/LOO-CV.
        n_params = self.n_trees + 2
        n_obs = self._n_obs
        ll = self.log_likelihood_

        aic_val = -2.0 * ll + 2.0 * n_params
        bic_val = -2.0 * ll + n_params * float(np.log(n_obs))

        return SFASummary(
            n_obs=n_obs,
            n_params=n_params,
            log_likelihood=ll,
            aic=aic_val,
            bic=bic_val,
            sigma_v=self.sigma_v_,
            sigma_u=self.sigma_u_,
            mean_efficiency=self._mean_efficiency,
            frontier_type=_FRONTIER_TYPE,
            inefficiency_dist=self.inefficiency,
        )

    # -- Bayesian-specific methods ------------------------------------------

    def credible_interval(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        alpha: float = 0.05,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute credible intervals for technical efficiency.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.
        alpha : float
            Significance level. Default 0.05 gives a 95% credible interval.

        Returns
        -------
        tuple[FloatArray, FloatArray]
            ``(lower, upper)`` bounds of shape ``(n_samples,)``.
        """
        self._check_fitted()
        if not (0.0 < alpha < 1.0):
            msg = f"alpha must be in (0, 1), got {alpha!r}"
            raise ValueError(msg)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        # In-sample: use posterior TE draws directly
        if (
            X_arr.shape == self._X_train.shape
            and np.allclose(X_arr, self._X_train)
            and np.allclose(y_arr, self._y_train)
        ):
            lower = np.quantile(self._te_posterior, alpha / 2, axis=0)
            upper = np.quantile(self._te_posterior, 1 - alpha / 2, axis=0)
            return np.maximum(lower, 0.0), np.minimum(upper, 1.0)

        # Out-of-sample: JLMS point estimate with no posterior
        te = self.efficiency(X_arr, y_arr)
        return te, te
