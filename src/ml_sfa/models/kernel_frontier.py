"""Local polynomial Stochastic Frontier Analysis estimator.

Implements ``KernelSFA``, a nonparametric local polynomial SFA estimator
based on Fan, Li & Weersink (1996) and Kumbhakar et al. (2007).  For each
observation, a kernel-weighted SFA MLE is solved using nearby observations.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

from ml_sfa._types import FloatArray, InefficiencyType
from ml_sfa.models.base import BaseSFAEstimator, SFASummary
from ml_sfa.utils.distributions import HalfNormal

__all__ = ["KernelSFA"]

_LOG_2_OVER_PI = float(np.log(2.0 / np.pi))
_MIN_SIGMA = 1e-8
_MAX_EXP_ARG = 20.0


def _gaussian_kernel(u: FloatArray) -> FloatArray:
    """Evaluate the standard Gaussian kernel (product form)."""
    p = u.shape[1]
    norm = (2.0 * np.pi) ** (p / 2.0)
    result: FloatArray = np.exp(-0.5 * np.sum(u**2, axis=1)) / norm
    return result


def _scott_bandwidth(X: FloatArray) -> FloatArray:
    """Scott's rule bandwidth: ``h_j = n^{-1/(p+4)} * std(X_j)``."""
    n, p = X.shape
    stds = np.maximum(np.std(X, axis=0), _MIN_SIGMA)
    bw: FloatArray = n ** (-1.0 / (p + 4)) * stds
    return bw


def _nll_half_normal_weighted(
    params: FloatArray,
    Z: FloatArray,
    y: FloatArray,
    weights: FloatArray,
    cost: bool,
) -> float:
    """Kernel-weighted negative log-likelihood for half-normal SFA."""
    k = Z.shape[1]
    beta = params[:k]
    sv_raw = np.clip(params[k], -_MAX_EXP_ARG, _MAX_EXP_ARG)
    su_raw = np.clip(params[k + 1], -_MAX_EXP_ARG, _MAX_EXP_ARG)
    sv = max(float(np.exp(sv_raw)), _MIN_SIGMA)
    su = max(float(np.exp(su_raw)), _MIN_SIGMA)

    sigma_sq = sv**2 + su**2
    sigma = np.sqrt(sigma_sq)
    lam = su / sv
    epsilon = y - Z @ beta
    phi_sign = 1.0 if cost else -1.0

    ll_i = (
        -np.log(sigma)
        + log_ndtr(phi_sign * epsilon * lam / sigma)
        - epsilon**2 / (2.0 * sigma_sq)
        + 0.5 * _LOG_2_OVER_PI
    )
    return -float(np.sum(weights * ll_i))


class KernelSFA(BaseSFAEstimator):
    """Local polynomial SFA estimator with kernel-weighted MLE.

    For each observation, solves a kernel-weighted SFA maximum likelihood
    problem using nearby observations, producing observation-specific
    frontier predictions and variance parameters.

    Parameters
    ----------
    inefficiency : InefficiencyType
        Inefficiency distribution (only ``"half-normal"`` supported).
    cost : bool
        If ``True``, estimate a cost frontier.
    bandwidth : float or str
        Kernel bandwidth. ``"scott"`` uses Scott's rule.
    kernel : str
        Kernel function (only ``"gaussian"`` supported).
    seed : int or None
        Random seed for reproducibility.
    """

    _FRONTIER_TYPE = "kernel"

    def __init__(
        self,
        inefficiency: InefficiencyType = "half-normal",
        *,
        cost: bool = False,
        bandwidth: float | str = "scott",
        kernel: str = "gaussian",
        seed: int | None = None,
    ) -> None:
        super().__init__(frontier="cobb-douglas", inefficiency=inefficiency, cost=cost)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.seed = seed

    def _compute_bandwidth(self, X: FloatArray) -> FloatArray:
        """Return bandwidth vector from the configured strategy."""
        if isinstance(self.bandwidth, str) and self.bandwidth == "scott":
            return _scott_bandwidth(X)
        return np.full(X.shape[1], float(self.bandwidth))

    def _fit_local(
        self,
        X: FloatArray,
        y: FloatArray,
        idx: int,
        h: FloatArray,
        rng: np.random.Generator,
    ) -> tuple[float, float, float, float]:
        """Solve a single local kernel-weighted SFA MLE.

        Returns ``(frontier_hat, sigma_v, sigma_u, local_ll)``.
        """
        x_i = X[idx]
        weights = _gaussian_kernel((X - x_i) / h)
        weights = weights / (np.sum(weights) + _MIN_SIGMA)

        n_obs = X.shape[0]
        ones = np.ones((n_obs, 1), dtype=X.dtype)
        Z = np.hstack([ones, X - x_i])
        k = Z.shape[1]

        # Weighted OLS initialisation
        W = np.diag(weights)
        ZtW = Z.T @ W
        try:
            beta_init = np.linalg.solve(ZtW @ Z + 1e-6 * np.eye(k), ZtW @ y)
        except np.linalg.LinAlgError:
            beta_init = np.zeros(k)
            beta_init[0] = y[idx]

        var_e = max(float(np.average((y - Z @ beta_init) ** 2, weights=weights)), 1e-6)
        log_s = np.log(max(np.sqrt(var_e / 2.0), _MIN_SIGMA))
        params0 = np.concatenate([beta_init, [log_s, log_s]])
        params0 += rng.normal(0, 0.01, size=params0.shape)

        result = minimize(
            _nll_half_normal_weighted,
            params0,
            args=(Z, y, weights, self.cost),
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-6},
        )

        sv_r = np.clip(result.x[k], -_MAX_EXP_ARG, _MAX_EXP_ARG)
        su_r = np.clip(result.x[k + 1], -_MAX_EXP_ARG, _MAX_EXP_ARG)
        sv = max(float(np.exp(sv_r)), _MIN_SIGMA)
        su = max(float(np.exp(su_r)), _MIN_SIGMA)
        return float(result.x[0]), sv, su, -float(result.fun)

    # -- public interface ---------------------------------------------------

    def fit(self, X: FloatArray, y: FloatArray) -> Self:
        """Fit the local polynomial SFA model.

        Parameters
        ----------
        X : FloatArray
            Input matrix ``(n_samples, n_features)``.
        y : FloatArray
            Observed output ``(n_samples,)``.

        Returns
        -------
        Self
            The fitted estimator.
        """
        X_val, y_val = self._validate_data(X, y)
        n, p = X_val.shape
        h = self._compute_bandwidth(X_val)
        rng = np.random.default_rng(self.seed)

        frontier_hat = np.empty(n)
        sigma_v_local = np.empty(n)
        sigma_u_local = np.empty(n)
        local_ll = np.empty(n)

        for i in range(n):
            f_hat, sv, su, ll_i = self._fit_local(X_val, y_val, i, h, rng)
            frontier_hat[i] = f_hat
            sigma_v_local[i] = sv
            sigma_u_local[i] = su
            local_ll[i] = ll_i

        self._X_fit = X_val.copy()
        self._y_fit = y_val.copy()
        self._frontier_hat = frontier_hat
        self._sigma_v_local = sigma_v_local
        self._sigma_u_local = sigma_u_local
        self._local_ll = local_ll

        self.sigma_v_ = float(np.median(sigma_v_local))
        self.sigma_u_ = float(np.median(sigma_u_local))
        self.log_likelihood_ = float(np.sum(local_ll))
        self.n_features_in_ = p
        self.is_fitted_ = True
        self._n_obs = n
        self._bandwidth = h

        te = self.efficiency(X_val, y_val)
        self._mean_efficiency = float(np.mean(te))
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        """Return frontier predictions (in-sample only).

        Raises ``NotImplementedError`` for out-of-sample data.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape != self._X_fit.shape or not np.allclose(X_arr, self._X_fit):
            msg = (
                "KernelSFA does not support out-of-sample prediction. "
                "Pass the same X used during fit()."
            )
            raise NotImplementedError(msg)
        return self._frontier_hat.copy()

    def efficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate technical efficiency ``exp(-u_hat)`` via JLMS."""
        u_hat = self.get_inefficiency(X, y)
        return np.exp(-u_hat)

    def get_inefficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate inefficiency via JLMS with per-observation sigmas."""
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        frontier = self.predict(X_arr)
        epsilon = y_arr - frontier

        dist = HalfNormal()
        n = len(epsilon)
        u_hat = np.empty(n)
        for i in range(n):
            eps_i = np.array([epsilon[i]])
            sv_i, su_i = self._sigma_v_local[i], self._sigma_u_local[i]
            if self.cost:
                u_hat[i] = dist.conditional_mean(-eps_i, sv_i, su_i)[0]
            else:
                u_hat[i] = dist.conditional_mean(eps_i, sv_i, su_i)[0]
        return u_hat

    def get_noise(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate noise component v."""
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        frontier = self.predict(X_arr)
        epsilon = y_arr - frontier
        u_hat = self.get_inefficiency(X_arr, y_arr)
        if self.cost:
            return epsilon - u_hat
        return epsilon + u_hat

    def log_likelihood(self) -> float:
        """Return the sum of local log-likelihoods."""
        self._check_fitted()
        return self.log_likelihood_

    def summary(self) -> SFASummary:
        """Return a summary of the fitted kernel SFA model."""
        self._check_fitted()
        p = self.n_features_in_
        n_params = p + 1 + 2  # local betas + sigma_v + sigma_u
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
            frontier_type=self._FRONTIER_TYPE,
            inefficiency_dist=self.inefficiency,
        )
