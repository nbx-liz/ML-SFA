"""Model evaluation and comparison metrics."""

from ml_sfa.evaluation.comparison import (
    ComparisonResult,
    compare_models,
    results_to_dataframe,
    run_benchmark,
)
from ml_sfa.evaluation.metrics import (
    aic,
    bic,
    coverage_rate,
    frontier_mse,
    rank_correlation,
    rmse_efficiency,
)

__all__ = [
    "ComparisonResult",
    "aic",
    "bic",
    "compare_models",
    "coverage_rate",
    "frontier_mse",
    "rank_correlation",
    "results_to_dataframe",
    "rmse_efficiency",
    "run_benchmark",
]
