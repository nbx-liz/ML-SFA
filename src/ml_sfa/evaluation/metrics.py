"""Evaluation metrics for SFA model assessment.

Provides pure functions for computing efficiency estimation accuracy,
model selection criteria, and prediction quality metrics.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ml_sfa.models._types import FloatArray

__all__ = [
    "aic",
    "bic",
    "coverage_rate",
    "frontier_mse",
    "rank_correlation",
    "rmse_efficiency",
]


def _validate_paired_arrays(a: FloatArray, b: FloatArray) -> None:
    """Validate that two arrays have the same length and are non-empty.

    Parameters
    ----------
    a : FloatArray
        First array.
    b : FloatArray
        Second array.

    Raises
    ------
    ValueError
        If arrays are empty or have different lengths.
    """
    if a.size == 0 or b.size == 0:
        msg = "Input arrays must not be empty."
        raise ValueError(msg)
    if a.shape[0] != b.shape[0]:
        msg = (
            f"Input arrays must have the same length, "
            f"got {a.shape[0]} and {b.shape[0]}."
        )
        raise ValueError(msg)


def rmse_efficiency(true_te: FloatArray, pred_te: FloatArray) -> float:
    """Root Mean Squared Error of technical efficiency estimates.

    Parameters
    ----------
    true_te : FloatArray
        True technical efficiency values.
    pred_te : FloatArray
        Predicted technical efficiency values.

    Returns
    -------
    float
        RMSE between true and predicted efficiency.

    Raises
    ------
    ValueError
        If arrays are empty or have different lengths.
    """
    _validate_paired_arrays(true_te, pred_te)
    return float(np.sqrt(np.mean((true_te - pred_te) ** 2)))


def rank_correlation(true_te: FloatArray, pred_te: FloatArray) -> float:
    """Spearman rank correlation between true and predicted efficiency.

    Parameters
    ----------
    true_te : FloatArray
        True technical efficiency values.
    pred_te : FloatArray
        Predicted technical efficiency values.

    Returns
    -------
    float
        Spearman rank correlation coefficient.

    Raises
    ------
    ValueError
        If arrays are empty or have different lengths.
    """
    _validate_paired_arrays(true_te, pred_te)
    correlation, _ = spearmanr(true_te, pred_te)
    return float(correlation)


def aic(log_likelihood: float, n_params: int) -> float:
    """Akaike Information Criterion: -2*LL + 2*k.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the fitted model.
    n_params : int
        Number of estimated parameters.

    Returns
    -------
    float
        AIC value (lower is better).
    """
    return -2.0 * log_likelihood + 2.0 * n_params


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Bayesian Information Criterion: -2*LL + k*ln(n).

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the fitted model.
    n_params : int
        Number of estimated parameters.
    n_obs : int
        Number of observations.

    Returns
    -------
    float
        BIC value (lower is better).
    """
    return -2.0 * log_likelihood + n_params * float(np.log(n_obs))


def frontier_mse(true_frontier: FloatArray, pred_frontier: FloatArray) -> float:
    """Mean Squared Error between true and estimated frontier values.

    Parameters
    ----------
    true_frontier : FloatArray
        True frontier values.
    pred_frontier : FloatArray
        Predicted frontier values.

    Returns
    -------
    float
        MSE between true and predicted frontier.

    Raises
    ------
    ValueError
        If arrays are empty or have different lengths.
    """
    _validate_paired_arrays(true_frontier, pred_frontier)
    return float(np.mean((true_frontier - pred_frontier) ** 2))


def coverage_rate(
    true_values: FloatArray,
    lower_bound: FloatArray,
    upper_bound: FloatArray,
) -> float:
    """Fraction of true values falling within [lower_bound, upper_bound].

    Parameters
    ----------
    true_values : FloatArray
        True values to check coverage for.
    lower_bound : FloatArray
        Lower bounds of the confidence interval.
    upper_bound : FloatArray
        Upper bounds of the confidence interval.

    Returns
    -------
    float
        Coverage rate in [0, 1].

    Raises
    ------
    ValueError
        If arrays are empty or have different lengths.
    """
    _validate_paired_arrays(true_values, lower_bound)
    _validate_paired_arrays(true_values, upper_bound)
    covered = (true_values >= lower_bound) & (true_values <= upper_bound)
    return float(np.mean(covered))
