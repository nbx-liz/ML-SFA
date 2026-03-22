"""Integration tests for systematic model comparison.

Tests the full comparison pipeline with ParametricSFA and KernelSFA
(NN and BART require optional deps and are tested separately).
"""

from __future__ import annotations

import pandas as pd

from ml_sfa.data.simulator import simulate_sfa
from ml_sfa.evaluation.comparison import compare_models, run_benchmark
from ml_sfa.models.kernel_frontier import KernelSFA
from ml_sfa.models.parametric import ParametricSFA


class TestFullComparison:
    """End-to-end comparison of Parametric + Kernel models."""

    def test_parametric_vs_kernel_on_cd_dgp(self) -> None:
        """Both models run on Cobb-Douglas DGP and produce valid results."""
        ds = simulate_sfa(
            n_obs=200,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        models = {
            "parametric_cd": ParametricSFA(frontier="cobb-douglas"),
            "kernel": KernelSFA(),
        }
        results = compare_models(models, ds)
        assert len(results) == 2
        for r in results:
            assert r.rmse_te < 0.3
            assert r.rank_corr > 0.3
            assert r.elapsed_seconds >= 0

    def test_parametric_advantage_on_cd(self) -> None:
        """Parametric model has lower RMSE on correct-form DGP."""
        ds = simulate_sfa(
            n_obs=300,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        models = {
            "parametric": ParametricSFA(frontier="cobb-douglas"),
            "kernel": KernelSFA(),
        }
        results = compare_models(models, ds)
        param_rmse = results[0].rmse_te
        kernel_rmse = results[1].rmse_te
        # Parametric should be competitive on its own DGP
        assert param_rmse < kernel_rmse + 0.05


class TestBenchmarkGrid:
    """Test run_benchmark with multiple DGPs."""

    def test_multi_dgp_benchmark(self) -> None:
        """Benchmark produces results for each DGP × model combo."""
        factories = {
            "parametric": {
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
            {
                "n_obs": 100,
                "n_inputs": 2,
                "frontier_type": "nonlinear",
                "sigma_v": 0.15,
                "sigma_u": 0.3,
                "seed": 42,
            },
        ]
        df = run_benchmark(factories, dgps)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(df["dgp_frontier_type"]) == {"cobb-douglas", "nonlinear"}
