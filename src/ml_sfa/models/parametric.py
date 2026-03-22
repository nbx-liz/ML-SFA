"""Parametric Stochastic Frontier Analysis estimator.

Implements traditional parametric SFA with Cobb-Douglas and Translog frontier
specifications, half-normal / truncated-normal / exponential inefficiency
distributions, and maximum likelihood estimation via ``scipy.optimize``.

Technical efficiency is estimated using the JLMS (Jondrow et al., 1982)
conditional expectation formula.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import skew

from ml_sfa._types import FloatArray, FrontierType, InefficiencyType
from ml_sfa.models.base import BaseSFAEstimator, SFASummary
from ml_sfa.utils.distributions import (
    Exponential,
    HalfNormal,
    InefficiencyDistribution,
    TruncatedNormal,
)

__all__ = ["ParametricSFA", "build_design_matrix"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_2_OVER_PI = float(np.log(2.0 / np.pi))
_LOG_2PI = float(np.log(2.0 * np.pi))

# Clipping bounds for numerical stability
_MIN_SIGMA = 1e-8
_MAX_EXP_ARG = 20.0


# ---------------------------------------------------------------------------
# Module-level helper: design matrix construction
# ---------------------------------------------------------------------------


def build_design_matrix(X: FloatArray, frontier_type: FrontierType) -> FloatArray:
    """Build the regression design matrix from raw inputs.

    Parameters
    ----------
    X : FloatArray
        Raw input matrix of shape ``(n, p)`` with **positive** values.
    frontier_type : FrontierType
        ``"cobb-douglas"`` or ``"translog"``.

    Returns
    -------
    FloatArray
        Design matrix *Z* with an intercept column prepended.

        * Cobb-Douglas: ``(n, p + 1)`` -- ``[1, ln x_1, ..., ln x_p]``
        * Translog: ``(n, 1 + p + p*(p+1)/2)`` -- intercept, log terms,
          ``0.5 * ln(x_j)^2`` squared terms, and ``ln(x_j)*ln(x_k)``
          cross-product terms for ``j < k``.
    """
    if np.any(X <= 0):
        msg = "All values in X must be strictly positive (log-transformed internally)."
        raise ValueError(msg)

    n, p = X.shape
    ln_x = np.log(X)
    ones = np.ones((n, 1), dtype=X.dtype)

    if frontier_type == "cobb-douglas":
        return np.hstack([ones, ln_x])

    # Translog: intercept + linear + 0.5*squared + cross-products
    squared = 0.5 * ln_x**2

    cross_terms: list[FloatArray] = []
    for j in range(p):
        for k in range(j + 1, p):
            cross_terms.append((ln_x[:, j] * ln_x[:, k]).reshape(-1, 1))

    parts = [ones, ln_x, squared]
    if cross_terms:
        parts.append(np.hstack(cross_terms))
    return np.hstack(parts)


# ---------------------------------------------------------------------------
# Negative log-likelihood functions (one per distribution family)
# ---------------------------------------------------------------------------


def _unpack_sigmas(params: FloatArray, k: int) -> tuple[float, float]:
    """Extract sigma_v and sigma_u from the parameter vector.

    Parameters
    ----------
    params : FloatArray
        Full parameter vector.
    k : int
        Number of beta coefficients.

    Returns
    -------
    tuple[float, float]
        ``(sigma_v, sigma_u)`` both clipped to ``[MIN_SIGMA, inf)``.
    """
    sv = float(np.exp(np.clip(params[k], -_MAX_EXP_ARG, _MAX_EXP_ARG)))
    su = float(np.exp(np.clip(params[k + 1], -_MAX_EXP_ARG, _MAX_EXP_ARG)))
    return max(sv, _MIN_SIGMA), max(su, _MIN_SIGMA)


def _nll_half_normal(
    params: FloatArray,
    Z: FloatArray,
    y: FloatArray,
    cost: bool,
) -> float:
    """Negative log-likelihood for the half-normal SFA model.

    Parameters
    ----------
    params : FloatArray
        ``[beta_0, ..., beta_k, ln_sigma_v, ln_sigma_u]``.
    Z : FloatArray
        Design matrix ``(n, k+1)``.
    y : FloatArray
        Observed output ``(n,)``.
    cost : bool
        ``True`` for cost frontier.

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    k = Z.shape[1]
    beta = params[:k]
    sigma_v, sigma_u = _unpack_sigmas(params, k)

    sigma_sq = sigma_v**2 + sigma_u**2
    sigma = np.sqrt(sigma_sq)
    lam = sigma_u / sigma_v

    epsilon = y - Z @ beta  # v - u (production) or v + u (cost)

    # Sign factor: for production Phi(-eps*lam/sigma), for cost Phi(eps*lam/sigma)
    phi_sign = 1.0 if cost else -1.0

    n = len(y)
    ll = (
        -n * np.log(sigma)
        + np.sum(log_ndtr(phi_sign * epsilon * lam / sigma))
        - np.sum(epsilon**2) / (2.0 * sigma_sq)
        + 0.5 * n * _LOG_2_OVER_PI
    )
    return -float(ll)


def _nll_truncated_normal(
    params: FloatArray,
    Z: FloatArray,
    y: FloatArray,
    cost: bool,
    mu: float,
) -> float:
    """Negative log-likelihood for the truncated-normal SFA model.

    Parameters
    ----------
    params : FloatArray
        ``[beta_0, ..., beta_k, ln_sigma_v, ln_sigma_u]``.
    Z : FloatArray
        Design matrix ``(n, k+1)``.
    y : FloatArray
        Observed output ``(n,)``.
    cost : bool
        ``True`` for cost frontier.
    mu : float
        Location parameter of the pre-truncation normal.

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    k = Z.shape[1]
    beta = params[:k]
    sigma_v, sigma_u = _unpack_sigmas(params, k)

    sigma_sq = sigma_v**2 + sigma_u**2
    sigma = np.sqrt(sigma_sq)
    sigma_star = np.sqrt(sigma_u**2 * sigma_v**2 / sigma_sq)

    epsilon = y - Z @ beta
    # For production: epsilon = v - u, for JLMS mu_star uses -epsilon
    # For cost: epsilon = v + u, need to flip sign
    eps_adj = epsilon if not cost else -epsilon

    mu_star = (-eps_adj * sigma_u**2 + mu * sigma_v**2) / sigma_sq

    n = len(y)
    ll = (
        -n * float(log_ndtr(mu / sigma_u))
        - n * np.log(sigma)
        + np.sum(log_ndtr(mu_star / sigma_star))
        - np.sum(eps_adj**2) / (2.0 * sigma_sq)
        + mu * np.sum(eps_adj) / sigma_sq
        - 0.5 * n * mu**2 / sigma_sq
        - 0.5 * n * _LOG_2PI
        + 0.5 * n * np.log(2.0)
    )
    return -float(ll)


def _nll_exponential(
    params: FloatArray,
    Z: FloatArray,
    y: FloatArray,
    cost: bool,
) -> float:
    """Negative log-likelihood for the exponential SFA model.

    Parameters
    ----------
    params : FloatArray
        ``[beta_0, ..., beta_k, ln_sigma_v, ln_sigma_u]``.
    Z : FloatArray
        Design matrix ``(n, k+1)``.
    y : FloatArray
        Observed output ``(n,)``.
    cost : bool
        ``True`` for cost frontier.

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    k = Z.shape[1]
    beta = params[:k]
    sigma_v, sigma_u = _unpack_sigmas(params, k)

    epsilon = y - Z @ beta
    # For production: epsilon = v - u.  For cost: flip sign.
    eps_adj = epsilon if not cost else -epsilon

    n = len(y)
    ll = (
        -n * np.log(sigma_u)
        + n * sigma_v**2 / (2.0 * sigma_u**2)
        + np.sum(log_ndtr(-eps_adj / sigma_v - sigma_v / sigma_u) + eps_adj / sigma_u)
    )
    return -float(ll)


# ---------------------------------------------------------------------------
# ParametricSFA estimator
# ---------------------------------------------------------------------------


class ParametricSFA(BaseSFAEstimator):
    """Traditional parametric Stochastic Frontier Analysis estimator.

    Fits a stochastic frontier model using maximum likelihood estimation.
    Supports Cobb-Douglas and Translog frontier specifications with
    half-normal, truncated-normal, or exponential inefficiency distributions.

    Parameters
    ----------
    frontier : FrontierType
        Frontier function specification: ``"cobb-douglas"`` or ``"translog"``.
    inefficiency : InefficiencyType
        Inefficiency distribution: ``"half-normal"``, ``"truncated-normal"``,
        or ``"exponential"``.
    cost : bool
        If ``True``, estimates a cost frontier where inefficiency *increases*
        the dependent variable.
    max_iter : int
        Maximum number of optimiser iterations.
    tol : float
        Convergence tolerance for the optimiser.

    Attributes (set after ``fit``)
    ----------
    coef_ : FloatArray
        Estimated coefficients of the design matrix.
    sigma_v_ : float
        Estimated noise standard deviation.
    sigma_u_ : float
        Estimated inefficiency scale parameter.
    log_likelihood_ : float
        Maximised log-likelihood.
    n_features_in_ : int
        Number of raw input features seen during fit.
    is_fitted_ : bool
        ``True`` after a successful call to ``fit``.
    """

    def __init__(
        self,
        frontier: FrontierType = "cobb-douglas",
        inefficiency: InefficiencyType = "half-normal",
        *,
        cost: bool = False,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> None:
        super().__init__(frontier=frontier, inefficiency=inefficiency, cost=cost)
        self.max_iter = max_iter
        self.tol = tol

    # -- private helpers ----------------------------------------------------

    def _get_distribution(self) -> InefficiencyDistribution:
        """Return the distribution object for the configured inefficiency type.

        Returns
        -------
        InefficiencyDistribution
            A frozen dataclass instance implementing the distribution protocol.
        """
        if self.inefficiency == "half-normal":
            return HalfNormal()
        if self.inefficiency == "truncated-normal":
            return TruncatedNormal(mu=0.0)
        if self.inefficiency == "exponential":
            return Exponential()
        msg = f"Unsupported inefficiency distribution: {self.inefficiency!r}"
        raise ValueError(msg)

    def _neg_log_likelihood(
        self, params: FloatArray, Z: FloatArray, y: FloatArray
    ) -> float:
        """Compute the negative log-likelihood for the current distribution.

        Parameters
        ----------
        params : FloatArray
            Packed parameter vector ``[beta, ln_sigma_v, ln_sigma_u]``.
        Z : FloatArray
            Design matrix.
        y : FloatArray
            Observed output.

        Returns
        -------
        float
            Negative log-likelihood (for minimisation).
        """
        if self.inefficiency == "half-normal":
            return _nll_half_normal(params, Z, y, self.cost)
        if self.inefficiency == "truncated-normal":
            return _nll_truncated_normal(params, Z, y, self.cost, mu=0.0)
        if self.inefficiency == "exponential":
            return _nll_exponential(params, Z, y, self.cost)

        msg = f"Unsupported inefficiency: {self.inefficiency!r}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def _init_params(self, Z: FloatArray, y: FloatArray) -> FloatArray:
        """Compute initial parameter estimates via OLS and method of moments.

        Parameters
        ----------
        Z : FloatArray
            Design matrix.
        y : FloatArray
            Observed output.

        Returns
        -------
        FloatArray
            Initial ``[beta_init, ln_sigma_v_init, ln_sigma_u_init]``.
        """
        # OLS: beta = (Z'Z)^{-1} Z'y
        beta_init = np.linalg.lstsq(Z, y, rcond=None)[0]
        residuals = y - Z @ beta_init

        var_e = max(float(np.var(residuals)), 1e-6)

        # Method of moments using skewness to split sigma_v / sigma_u
        m3 = float(skew(residuals, bias=True))
        c = np.sqrt(2.0 / np.pi) * (4.0 / np.pi - 1.0)

        # For production: E[skew(epsilon)] < 0; for cost: > 0
        raw_sigma_u_cubed = (-m3 / c) if not self.cost else (m3 / c)

        if raw_sigma_u_cubed > 0 and c > 0:
            sigma_u_init = float(raw_sigma_u_cubed ** (1.0 / 3.0))
            sigma_v_sq = var_e - (1.0 - 2.0 / np.pi) * sigma_u_init**2
            if sigma_v_sq > 0:
                sigma_v_init = float(np.sqrt(sigma_v_sq))
            else:
                sigma_v_init = float(np.sqrt(var_e / 2.0))
                sigma_u_init = float(np.sqrt(var_e / 2.0))
        else:
            sigma_v_init = float(np.sqrt(var_e / 2.0))
            sigma_u_init = float(np.sqrt(var_e / 2.0))

        sigma_v_init = max(sigma_v_init, _MIN_SIGMA)
        sigma_u_init = max(sigma_u_init, _MIN_SIGMA)

        return np.concatenate(
            [
                beta_init,
                [np.log(sigma_v_init), np.log(sigma_u_init)],
            ]
        )

    def _compute_epsilon(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Compute composed error residuals ``epsilon = y - Z @ beta``.

        Parameters
        ----------
        X : FloatArray
            Raw input matrix.
        y : FloatArray
            Observed output.

        Returns
        -------
        FloatArray
            Composed error. Production: ``v - u``.  Cost: ``v + u``.
        """
        Z = build_design_matrix(X, self.frontier)
        result: FloatArray = y - Z @ self.coef_
        return result

    # -- public interface ---------------------------------------------------

    def fit(self, X: FloatArray, y: FloatArray) -> Self:
        """Fit the parametric SFA model via maximum likelihood.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)`` with positive
            values (will be log-transformed internally).
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        Self
            The fitted estimator.
        """
        X_val, y_val = self._validate_data(X, y)
        Z = build_design_matrix(X_val, self.frontier)
        n_obs = X_val.shape[0]

        params0 = self._init_params(Z, y_val)

        # Primary optimiser: L-BFGS-B
        result = minimize(
            self._neg_log_likelihood,
            params0,
            args=(Z, y_val),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        # Fallback: Nelder-Mead when L-BFGS-B fails to converge
        if not result.success:
            result_nm = minimize(
                self._neg_log_likelihood,
                params0,
                args=(Z, y_val),
                method="Nelder-Mead",
                options={"maxiter": self.max_iter * 2, "xatol": self.tol},
            )
            if result_nm.fun < result.fun:
                result = result_nm

        # Unpack results
        k = Z.shape[1]
        self.coef_ = result.x[:k].copy()
        self.sigma_v_ = float(np.exp(np.clip(result.x[k], -_MAX_EXP_ARG, _MAX_EXP_ARG)))
        self.sigma_u_ = float(
            np.exp(np.clip(result.x[k + 1], -_MAX_EXP_ARG, _MAX_EXP_ARG))
        )
        self.log_likelihood_ = -float(result.fun)
        self.n_features_in_ = X_val.shape[1]
        self.is_fitted_ = True
        self._n_obs = n_obs

        # Precompute mean efficiency for summary
        te = self.efficiency(X_val, y_val)
        self._mean_efficiency = float(np.mean(te))

        return self

    def predict(self, X: FloatArray) -> FloatArray:
        """Predict frontier values for the given inputs.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        FloatArray
            Predicted frontier values ``Z @ coef_``.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        Z = build_design_matrix(X_arr, self.frontier)
        prediction: FloatArray = Z @ self.coef_
        return prediction

    def efficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate technical efficiency for each observation.

        Uses the JLMS conditional expectation ``E[u | epsilon]`` to compute
        ``TE = exp(-u_hat)``.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        FloatArray
            Technical efficiency in ``(0, 1]``.
        """
        u_hat = self.get_inefficiency(X, y)
        return np.exp(-u_hat)

    def get_inefficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate the one-sided inefficiency ``u_hat`` via JLMS.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        FloatArray
            Non-negative inefficiency estimates.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        epsilon = self._compute_epsilon(X_arr, y_arr)

        dist = self._get_distribution()
        # epsilon = y - f.
        # Production: epsilon = v - u → pass epsilon directly to JLMS.
        # Cost: epsilon = v + u → negate so conditional_mean sees -(v+u)
        #   which mimics the production convention.
        if self.cost:
            return dist.conditional_mean(-epsilon, self.sigma_v_, self.sigma_u_)
        return dist.conditional_mean(epsilon, self.sigma_v_, self.sigma_u_)

    def get_noise(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate the symmetric noise component ``v_hat``.

        For production: ``epsilon = v - u`` so ``v_hat = epsilon + u_hat``.
        For cost: ``epsilon = v + u`` so ``v_hat = epsilon - u_hat``.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        FloatArray
            Estimated noise values.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        epsilon = self._compute_epsilon(X_arr, y_arr)
        u_hat = self.get_inefficiency(X_arr, y_arr)

        if self.cost:
            return epsilon - u_hat
        return epsilon + u_hat

    def log_likelihood(self) -> float:
        """Return the maximised log-likelihood of the fitted model.

        Returns
        -------
        float
            Log-likelihood value.
        """
        self._check_fitted()
        return self.log_likelihood_

    def summary(self) -> SFASummary:
        """Return a summary of the fitted model.

        Returns
        -------
        SFASummary
            Frozen dataclass with model diagnostics including AIC, BIC,
            sigma estimates, and mean efficiency.
        """
        self._check_fitted()

        n_params = len(self.coef_) + 2  # betas + sigma_v + sigma_u
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
            frontier_type=self.frontier,
            inefficiency_dist=self.inefficiency,
        )
