"""Model comparison framework for SFA estimators.

Provides ``compare_models`` to evaluate multiple SFA estimators on simulated
data and ``run_benchmark`` to run a grid of DGP scenarios.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml_sfa._types import FloatArray
from ml_sfa.data.simulator import SFADataset
from ml_sfa.models.base import BaseSFAEstimator

__all__ = ["ComparisonResult", "compare_models", "run_benchmark"]


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Immutable result of a single model evaluation.

    Attributes
    ----------
    model_name : str
        Name identifier for the model.
    rmse_te : float
        RMSE of technical efficiency estimates vs true TE.
    rank_corr : float
        Spearman rank correlation of efficiency estimates.
    frontier_mse : float
        Mean squared error of frontier predictions.
    mean_efficiency : float
        Mean estimated technical efficiency.
    sigma_v : float
        Estimated noise standard deviation.
    sigma_u : float
        Estimated inefficiency scale parameter.
    log_likelihood : float
        Maximised (or approximate) log-likelihood.
    elapsed_seconds : float
        Wall-clock time for fitting in seconds.
    """

    model_name: str
    rmse_te: float
    rank_corr: float
    frontier_mse: float
    mean_efficiency: float
    sigma_v: float
    sigma_u: float
    log_likelihood: float
    elapsed_seconds: float


def _compute_true_frontier(ds: SFADataset) -> FloatArray:
    """Recover the deterministic frontier from dataset components.

    ``true_frontier = y - v + u`` (production) or ``y - v - u`` (cost).
    Since we have ``y = frontier + v - u`` for production, frontier = y - v + u.
    """
    return ds.y - ds.v + ds.u


def compare_models(
    models: dict[str, BaseSFAEstimator],
    dataset: SFADataset,
) -> list[ComparisonResult]:
    """Evaluate multiple SFA models on a single dataset.

    Parameters
    ----------
    models : dict[str, BaseSFAEstimator]
        Mapping of model name to unfitted estimator instance.
    dataset : SFADataset
        Simulated dataset with ground-truth TE.

    Returns
    -------
    list[ComparisonResult]
        One result per model, in insertion order.
    """
    from sklearn.base import clone

    true_frontier = _compute_true_frontier(dataset)
    results: list[ComparisonResult] = []

    for name, model in models.items():
        fitted = clone(model)
        t0 = time.perf_counter()
        fitted.fit(dataset.X, dataset.y)
        elapsed = time.perf_counter() - t0

        te_hat = fitted.efficiency(dataset.X, dataset.y)
        pred = fitted.predict(dataset.X)

        rmse_te = float(np.sqrt(np.mean((dataset.te - te_hat) ** 2)))
        rank_corr = float(spearmanr(dataset.te, te_hat).statistic)
        f_mse = float(np.mean((true_frontier - pred) ** 2))

        results.append(
            ComparisonResult(
                model_name=name,
                rmse_te=rmse_te,
                rank_corr=rank_corr,
                frontier_mse=f_mse,
                mean_efficiency=float(np.mean(te_hat)),
                sigma_v=fitted.sigma_v_,
                sigma_u=fitted.sigma_u_,
                log_likelihood=fitted.log_likelihood(),
                elapsed_seconds=elapsed,
            )
        )

    return results


def results_to_dataframe(results: list[ComparisonResult]) -> pd.DataFrame:
    """Convert comparison results to a pandas DataFrame.

    Parameters
    ----------
    results : list[ComparisonResult]
        Results from ``compare_models`` or ``run_benchmark``.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per result.
    """
    records: list[dict[str, Any]] = []
    for r in results:
        records.append(
            {
                "model": r.model_name,
                "rmse_te": r.rmse_te,
                "rank_corr": r.rank_corr,
                "frontier_mse": r.frontier_mse,
                "mean_efficiency": r.mean_efficiency,
                "sigma_v": r.sigma_v,
                "sigma_u": r.sigma_u,
                "log_likelihood": r.log_likelihood,
                "elapsed_s": r.elapsed_seconds,
            }
        )
    return pd.DataFrame(records)


def run_benchmark(
    model_factories: dict[str, dict[str, Any]],
    dgp_configs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Run a grid of model × DGP comparisons.

    Parameters
    ----------
    model_factories : dict[str, dict[str, Any]]
        Mapping of model name to dict with keys ``"class"`` (the estimator
        class) and ``"kwargs"`` (constructor keyword arguments).
    dgp_configs : list[dict[str, Any]]
        Each dict is passed as keyword arguments to ``simulate_sfa``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for DGP parameters, model name, and all
        comparison metrics.
    """
    from ml_sfa.data.simulator import simulate_sfa

    # Validate factory entries
    for name, factory in model_factories.items():
        if "class" not in factory:
            msg = (
                f"model_factories entry {name!r} is missing required "
                f"key 'class'. Got keys: {list(factory.keys())}"
            )
            raise ValueError(msg)

    all_frames: list[pd.DataFrame] = []

    for dgp_kwargs in dgp_configs:
        ds = simulate_sfa(**dgp_kwargs)

        models: dict[str, BaseSFAEstimator] = {}
        for name, factory in model_factories.items():
            cls = factory["class"]
            kwargs = factory.get("kwargs", {})
            models[name] = cls(**kwargs)

        results = compare_models(models, ds)
        df = results_to_dataframe(results)

        # Add DGP metadata columns
        for key, val in dgp_kwargs.items():
            df[f"dgp_{key}"] = val

        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)
