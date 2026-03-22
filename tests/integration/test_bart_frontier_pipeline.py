"""Integration tests for BARTFrontierSFA pipeline.

End-to-end tests: simulate data → fit BART-SFA → evaluate TE recovery.
Uses minimal MCMC settings for speed; thresholds are relaxed accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

pymc = pytest.importorskip("pymc")

from ml_sfa.data.simulator import simulate_sfa  # noqa: E402
from ml_sfa.evaluation.metrics import rmse_efficiency  # noqa: E402
from ml_sfa.models.bart_frontier import BARTFrontierSFA  # noqa: E402

_FAST_KWARGS = dict(
    n_trees=20,
    n_draws=200,
    n_tune=200,
    n_chains=1,
    seed=42,
)


# ---------------------------------------------------------------------------
# Efficiency recovery
# ---------------------------------------------------------------------------


class TestEfficiencyRecovery:
    """Test that BART-SFA recovers technical efficiency."""

    def test_half_normal_recovery(self) -> None:
        """BART-SFA recovers TE with half-normal DGP."""
        ds = simulate_sfa(
            n_obs=200,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        model = BARTFrontierSFA(**_FAST_KWARGS)
        model.fit(ds.X, ds.y)
        te_hat = model.efficiency(ds.X, ds.y)
        rmse = rmse_efficiency(ds.te, te_hat)
        # Relaxed threshold for MCMC with few draws
        assert rmse < 0.20, f"RMSE too high: {rmse:.4f}"

    def test_rank_correlation(self) -> None:
        """Efficiency ranking is correlated with true ranking."""
        ds = simulate_sfa(
            n_obs=200,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        model = BARTFrontierSFA(**_FAST_KWARGS)
        model.fit(ds.X, ds.y)
        te_hat = model.efficiency(ds.X, ds.y)
        rho = spearmanr(ds.te, te_hat).statistic
        assert rho > 0.5, f"Rank correlation too low: {rho:.4f}"


# ---------------------------------------------------------------------------
# Credible interval coverage
# ---------------------------------------------------------------------------


class TestCredibleIntervalCoverage:
    """Test that credible intervals have reasonable coverage."""

    def test_coverage_rate(self) -> None:
        """95% CI should cover true TE for a significant fraction."""
        ds = simulate_sfa(
            n_obs=200,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        model = BARTFrontierSFA(**_FAST_KWARGS)
        model.fit(ds.X, ds.y)
        lower, upper = model.credible_interval(ds.X, ds.y, alpha=0.05)
        covered = np.mean((ds.te >= lower) & (ds.te <= upper))
        # With few draws, coverage may be imperfect; expect > 50%
        assert covered > 0.50, f"Coverage too low: {covered:.2%}"


# ---------------------------------------------------------------------------
# Sklearn compatibility
# ---------------------------------------------------------------------------


class TestSklearnCompat:
    """Test sklearn API compatibility."""

    def test_clone(self) -> None:
        """sklearn clone works on BARTFrontierSFA."""
        from sklearn.base import clone

        model = BARTFrontierSFA(n_trees=30, n_draws=100)
        cloned = clone(model)
        assert cloned.get_params() == model.get_params()
        assert cloned is not model


# ---------------------------------------------------------------------------
# AIC/BIC consistency
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Test AIC/BIC consistency."""

    def test_aic_bic_finite(self) -> None:
        """AIC and BIC are finite values."""
        ds = simulate_sfa(
            n_obs=100,
            n_inputs=2,
            frontier_type="cobb-douglas",
            inefficiency_dist="half-normal",
            sigma_v=0.15,
            sigma_u=0.3,
            seed=42,
        )
        model = BARTFrontierSFA(n_trees=10, n_draws=50, n_tune=50, n_chains=1, seed=42)
        model.fit(ds.X, ds.y)
        s = model.summary()
        assert np.isfinite(s.aic)
        assert np.isfinite(s.bic)
