"""Simulation data generator for Stochastic Frontier Analysis.

Provides data-generating processes (DGPs) for validating SFA model
implementations, including Cobb-Douglas and Translog frontiers with
configurable inefficiency distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

import numpy as np
from scipy.stats import truncnorm

from ml_sfa._types import FloatArray, FrontierType, InefficiencyType

__all__ = ["SFADataset", "simulate_sfa"]


@dataclass(frozen=True, slots=True)
class SFADataset:
    """Immutable container for simulated SFA data.

    Attributes:
        X: Input matrix of shape ``(n_obs, n_inputs)``.
        y: Log-output vector of shape ``(n_obs,)``.
        te: True technical efficiency ``exp(-u)`` of shape ``(n_obs,)``.
        u: True inefficiency of shape ``(n_obs,)``.
        v: True noise of shape ``(n_obs,)``.
        beta: True coefficient vector.
        frontier_type: Frontier function type used for generation.
        inefficiency_dist: Inefficiency distribution used for generation.
        sigma_v: Standard deviation of noise.
        sigma_u: Scale parameter for inefficiency distribution.
        n_obs: Number of observations.
        n_inputs: Number of input variables.
    """

    X: FloatArray
    y: FloatArray
    te: FloatArray
    u: FloatArray
    v: FloatArray
    beta: FloatArray
    frontier_type: str
    inefficiency_dist: str
    sigma_v: float
    sigma_u: float
    n_obs: int
    n_inputs: int


def _generate_inefficiency(
    rng: np.random.Generator,
    dist: InefficiencyType,
    sigma_u: float,
    n: int,
) -> FloatArray:
    """Draw inefficiency terms from the specified distribution.

    Args:
        rng: NumPy random generator instance.
        dist: Name of the inefficiency distribution.
        sigma_u: Scale parameter.
        n: Number of draws.

    Returns:
        Non-negative array of shape ``(n,)``.
    """
    if dist == "half-normal":
        return np.abs(rng.normal(0.0, sigma_u, n))
    if dist == "truncated-normal":
        # N(0, sigma_u^2) truncated at 0 from below
        a_clip = 0.0 / sigma_u  # lower bound in standard normal units
        b_clip = np.inf
        result: FloatArray = truncnorm.rvs(
            a_clip,
            b_clip,
            loc=0.0,
            scale=sigma_u,
            size=n,
            random_state=rng,
        )
        return result
    if dist == "exponential":
        return rng.exponential(sigma_u, n)

    # Should be unreachable after validation, but keeps mypy happy.
    msg = f"Unsupported inefficiency_dist: {dist!r}"
    raise ValueError(msg)


def _build_cobb_douglas_beta(
    rng: np.random.Generator,
    n_inputs: int,
) -> FloatArray:
    """Generate sensible Cobb-Douglas coefficients.

    Intercept ~ 1.0, slopes sum to ~ 0.8-0.9 (decreasing returns).

    Args:
        rng: NumPy random generator instance.
        n_inputs: Number of input variables.

    Returns:
        Coefficient array of length ``n_inputs + 1``.
    """
    intercept = np.array([1.0])
    # Generate slopes that sum to roughly 0.85
    raw = rng.uniform(0.3, 0.5, n_inputs)
    target_sum = 0.85
    slopes = raw * (target_sum / raw.sum())
    return np.concatenate([intercept, slopes])


def _build_translog_beta(
    rng: np.random.Generator,
    n_inputs: int,
) -> FloatArray:
    """Generate Translog coefficients.

    Includes intercept, linear terms, squared terms, and cross-product terms.

    Args:
        rng: NumPy random generator instance.
        n_inputs: Number of input variables.

    Returns:
        Coefficient array of length ``1 + n_inputs + n_inputs + C(n_inputs, 2)``.
    """
    intercept = np.array([1.0])
    linear = rng.uniform(0.2, 0.4, n_inputs)
    squared = rng.uniform(-0.05, 0.05, n_inputs)
    n_cross = n_inputs * (n_inputs - 1) // 2
    cross = rng.uniform(-0.03, 0.03, n_cross) if n_cross > 0 else np.array([])
    return np.concatenate([intercept, linear, squared, cross])


def _build_nonlinear_beta(
    rng: np.random.Generator,
    n_inputs: int,
) -> FloatArray:
    """Generate coefficients for the nonlinear frontier DGP.

    The nonlinear frontier uses a mix of log, power, and interaction terms
    that cannot be captured by Cobb-Douglas or Translog specifications.

    Args:
        rng: NumPy random generator instance.
        n_inputs: Number of input variables.

    Returns:
        Coefficient array of length ``n_inputs + 1``.
    """
    intercept = np.array([1.5])
    slopes = rng.uniform(0.3, 0.7, n_inputs)
    return np.concatenate([intercept, slopes])


def _compute_nonlinear_frontier(
    x: FloatArray,
    beta: FloatArray,
    n_inputs: int,
) -> FloatArray:
    """Compute a nonlinear frontier that parametric models cannot capture.

    ``f(x) = beta_0 + sum_j beta_j * x_j^0.6 + 0.3 * sin(ln(x_1) * ln(x_2))``

    The sine interaction term introduces non-monotone local behaviour that
    Cobb-Douglas and Translog cannot represent.

    Args:
        x: Raw inputs of shape ``(n_obs, n_inputs)``.
        beta: Coefficient vector.
        n_inputs: Number of input variables.

    Returns:
        Frontier values of shape ``(n_obs,)``.
    """
    frontier: FloatArray = np.full(x.shape[0], beta[0])
    for j in range(n_inputs):
        frontier = frontier + beta[j + 1] * np.power(x[:, j], 0.6)

    # Add non-parametric interaction if at least 2 inputs
    if n_inputs >= 2:
        frontier = frontier + 0.3 * np.sin(np.log(x[:, 0]) * np.log(x[:, 1]))

    return frontier


def _compute_frontier(
    ln_x: FloatArray,
    beta: FloatArray,
    frontier_type: FrontierType,
    n_inputs: int,
) -> FloatArray:
    """Compute the deterministic frontier for each observation.

    Args:
        ln_x: Log-transformed inputs of shape ``(n_obs, n_inputs)``.
        beta: Coefficient vector.
        frontier_type: ``"cobb-douglas"`` or ``"translog"``
            (``"nonlinear"`` is handled separately before this function).
        n_inputs: Number of input variables.

    Returns:
        Frontier values of shape ``(n_obs,)``.
    """
    if frontier_type not in ("cobb-douglas", "translog"):
        msg = f"_compute_frontier: unsupported frontier_type {frontier_type!r}"
        raise ValueError(msg)

    if frontier_type == "cobb-douglas":
        cd_frontier: FloatArray = beta[0] + ln_x @ beta[1:]
        return cd_frontier

    # Translog: intercept + linear + squared + cross
    offset = 1
    linear_beta = beta[offset : offset + n_inputs]
    offset += n_inputs
    squared_beta = beta[offset : offset + n_inputs]
    offset += n_inputs
    cross_beta = beta[offset:]

    frontier = beta[0] + ln_x @ linear_beta

    # Squared terms: 0.5 * beta_jj * ln(x_j)^2
    frontier = frontier + 0.5 * (ln_x**2) @ squared_beta

    # Cross-product terms: beta_jk * ln(x_j) * ln(x_k) for j < k
    idx = 0
    for j in range(n_inputs):
        for k in range(j + 1, n_inputs):
            frontier = frontier + cross_beta[idx] * ln_x[:, j] * ln_x[:, k]
            idx += 1

    tl_frontier: FloatArray = frontier
    return tl_frontier


def simulate_sfa(
    n_obs: int = 500,
    n_inputs: int = 3,
    frontier_type: FrontierType = "cobb-douglas",
    inefficiency_dist: InefficiencyType = "half-normal",
    sigma_v: float = 0.1,
    sigma_u: float = 0.2,
    seed: int = 42,
    cost: bool = False,
) -> SFADataset:
    """Generate simulated data from a stochastic frontier model.

    Args:
        n_obs: Number of observations.
        n_inputs: Number of input variables.
        frontier_type: ``"cobb-douglas"`` or ``"translog"``.
        inefficiency_dist: ``"half-normal"``, ``"truncated-normal"``,
            or ``"exponential"``.
        sigma_v: Standard deviation of symmetric noise.
        sigma_u: Scale parameter for the inefficiency distribution.
        seed: Random seed for reproducibility.
        cost: If ``True``, generates a cost frontier (``y = frontier + v + u``).
            If ``False`` (default), generates a production frontier
            (``y = frontier + v - u``).

    Returns:
        An :class:`SFADataset` containing the simulated data and true
        parameters.

    Raises:
        ValueError: If *frontier_type* or *inefficiency_dist* is not a
            recognised value.
    """
    # --- Validate inputs ---
    valid_frontiers = get_args(FrontierType)
    if frontier_type not in valid_frontiers:
        msg = f"frontier_type must be one of {valid_frontiers}, got {frontier_type!r}"
        raise ValueError(msg)

    valid_dists = get_args(InefficiencyType)
    if inefficiency_dist not in valid_dists:
        msg = (
            f"inefficiency_dist must be one of {valid_dists}, got {inefficiency_dist!r}"
        )
        raise ValueError(msg)

    rng = np.random.default_rng(seed)

    # --- Generate inputs ---
    x = rng.uniform(1.0, 10.0, size=(n_obs, n_inputs))
    ln_x = np.log(x)

    # --- Generate coefficients ---
    if frontier_type == "cobb-douglas":
        beta = _build_cobb_douglas_beta(rng, n_inputs)
    elif frontier_type == "nonlinear":
        beta = _build_nonlinear_beta(rng, n_inputs)
    else:
        beta = _build_translog_beta(rng, n_inputs)

    # --- Compute deterministic frontier ---
    if frontier_type == "nonlinear":
        frontier = _compute_nonlinear_frontier(x, beta, n_inputs)
    else:
        frontier = _compute_frontier(ln_x, beta, frontier_type, n_inputs)

    # --- Generate error components ---
    v = rng.normal(0.0, sigma_v, n_obs)
    u = _generate_inefficiency(rng, inefficiency_dist, sigma_u, n_obs)

    # --- Composite output ---
    y = frontier + v + u if cost else frontier + v - u

    te = np.exp(-u)

    return SFADataset(
        X=x,
        y=y,
        te=te,
        u=u,
        v=v,
        beta=beta,
        frontier_type=frontier_type,
        inefficiency_dist=inefficiency_dist,
        sigma_v=sigma_v,
        sigma_u=sigma_u,
        n_obs=n_obs,
        n_inputs=n_inputs,
    )
