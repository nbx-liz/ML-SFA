"""Unit tests for model comparison framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_sfa.data.simulator import simulate_sfa
from ml_sfa.evaluation.comparison import (
    ComparisonResult,
    compare_models,
    results_to_dataframe,
    run_benchmark,
)
from ml_sfa.models.parametric import ParametricSFA


@pytest.fixture()
def simple_dataset() -> object:
    """Small simulated dataset."""
    return simulate_sfa(
        n_obs=100,
        n_inputs=2,
        frontier_type="cobb-douglas",
        inefficiency_dist="half-normal",
        sigma_v=0.15,
        sigma_u=0.3,
        seed=42,
    )


class TestCompareModels:
    """Tests for compare_models function."""

    def test_returns_results(self, simple_dataset: object) -> None:
        """compare_models returns a list of ComparisonResult."""
        ds = simple_dataset  # type: ignore[assignment]
        models = {"parametric": ParametricSFA()}
        results = compare_models(models, ds)
        assert len(results) == 1
        assert isinstance(results[0], ComparisonResult)

    def test_result_fields(self, simple_dataset: object) -> None:
        """ComparisonResult has correct field types."""
        ds = simple_dataset  # type: ignore[assignment]
        models = {"psfa": ParametricSFA()}
        r = compare_models(models, ds)[0]
        assert r.model_name == "psfa"
        assert 0 < r.rmse_te < 1.0
        assert -1 <= r.rank_corr <= 1
        assert r.frontier_mse >= 0
        assert 0 < r.mean_efficiency <= 1.0
        assert r.sigma_v > 0
        assert r.sigma_u > 0
        assert np.isfinite(r.log_likelihood)
        assert r.elapsed_seconds >= 0

    def test_multiple_models(self, simple_dataset: object) -> None:
        """Handles multiple models."""
        ds = simple_dataset  # type: ignore[assignment]
        models = {
            "cd": ParametricSFA(frontier="cobb-douglas"),
            "tl": ParametricSFA(frontier="translog"),
        }
        results = compare_models(models, ds)
        assert len(results) == 2
        names = [r.model_name for r in results]
        assert names == ["cd", "tl"]


class TestResultsToDataframe:
    """Tests for results_to_dataframe."""

    def test_returns_dataframe(self) -> None:
        """Converts results to pandas DataFrame."""
        results = [
            ComparisonResult(
                model_name="test",
                rmse_te=0.1,
                rank_corr=0.9,
                frontier_mse=0.05,
                mean_efficiency=0.8,
                sigma_v=0.15,
                sigma_u=0.3,
                log_likelihood=-100.0,
                elapsed_seconds=1.5,
            )
        ]
        df = results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "model" in df.columns
        assert "rmse_te" in df.columns

    def test_empty_results(self) -> None:
        """Empty results produce empty DataFrame."""
        df = results_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestRunBenchmark:
    """Tests for run_benchmark grid evaluation."""

    def test_single_dgp(self) -> None:
        """Runs benchmark with one DGP config."""
        factories = {
            "psfa": {
                "class": ParametricSFA,
                "kwargs": {"frontier": "cobb-douglas"},
            },
        }
        dgps = [
            {
                "n_obs": 100,
                "n_inputs": 2,
                "frontier_type": "cobb-douglas",
                "sigma_v": 0.15,
                "sigma_u": 0.3,
                "seed": 42,
            },
        ]
        df = run_benchmark(factories, dgps)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "dgp_n_obs" in df.columns
        assert "dgp_frontier_type" in df.columns

    def test_grid_expansion(self) -> None:
        """Multiple DGPs × models produce correct row count."""
        factories = {
            "cd": {"class": ParametricSFA, "kwargs": {"frontier": "cobb-douglas"}},
            "tl": {"class": ParametricSFA, "kwargs": {"frontier": "translog"}},
        }
        dgps = [
            {"n_obs": 50, "n_inputs": 2, "sigma_v": 0.15, "sigma_u": 0.3, "seed": 1},
            {"n_obs": 50, "n_inputs": 2, "sigma_v": 0.15, "sigma_u": 0.3, "seed": 2},
        ]
        df = run_benchmark(factories, dgps)
        assert len(df) == 4  # 2 models × 2 DGPs


class TestNonlinearDGP:
    """Test that the nonlinear DGP works with simulate_sfa."""

    def test_simulate_nonlinear(self) -> None:
        """simulate_sfa accepts frontier_type='nonlinear'."""
        ds = simulate_sfa(
            n_obs=100,
            n_inputs=3,
            frontier_type="nonlinear",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        assert ds.X.shape == (100, 3)
        assert ds.y.shape == (100,)
        assert ds.frontier_type == "nonlinear"

    def test_nonlinear_different_from_cd(self) -> None:
        """Nonlinear DGP produces different frontiers from Cobb-Douglas."""
        ds_cd = simulate_sfa(
            n_obs=100, n_inputs=2, frontier_type="cobb-douglas", seed=42
        )
        ds_nl = simulate_sfa(n_obs=100, n_inputs=2, frontier_type="nonlinear", seed=42)
        # Same seed but different frontier type → different y
        assert not np.allclose(ds_cd.y, ds_nl.y)
