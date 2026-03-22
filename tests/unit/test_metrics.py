"""Unit tests for evaluation metrics module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ml_sfa.evaluation.metrics import (
    aic,
    bic,
    coverage_rate,
    frontier_mse,
    rank_correlation,
    rmse_efficiency,
)


class TestRMSEEfficiency:
    """Tests for rmse_efficiency function."""

    def test_rmse_efficiency_perfect(self) -> None:
        """RMSE is 0.0 when predicted == true."""
        true_te = np.array([0.9, 0.8, 0.7, 0.6])
        pred_te = np.array([0.9, 0.8, 0.7, 0.6])
        result = rmse_efficiency(true_te, pred_te)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_rmse_efficiency_known_value(self) -> None:
        """Hand-computed case: true=[0.8, 0.6], pred=[0.7, 0.5] -> RMSE=0.1."""
        true_te = np.array([0.8, 0.6])
        pred_te = np.array([0.7, 0.5])
        # RMSE = sqrt(mean([0.01, 0.01])) = sqrt(0.01) = 0.1
        result = rmse_efficiency(true_te, pred_te)
        assert result == pytest.approx(0.1, abs=1e-12)

    def test_rmse_efficiency_symmetric(self) -> None:
        """rmse(a, b) == rmse(b, a)."""
        a = np.array([0.9, 0.7, 0.5])
        b = np.array([0.8, 0.6, 0.4])
        assert rmse_efficiency(a, b) == pytest.approx(rmse_efficiency(b, a), abs=1e-12)


class TestRankCorrelation:
    """Tests for rank_correlation function."""

    def test_rank_correlation_perfect(self) -> None:
        """Spearman rho = 1.0 for identical ranking."""
        true_te = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        pred_te = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = rank_correlation(true_te, pred_te)
        assert result == pytest.approx(1.0, abs=1e-12)

    def test_rank_correlation_reversed(self) -> None:
        """rho = -1.0 for completely reversed ranking."""
        true_te = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        pred_te = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        result = rank_correlation(true_te, pred_te)
        assert result == pytest.approx(-1.0, abs=1e-12)

    def test_rank_correlation_known_value(self) -> None:
        """Use a small array with known Spearman correlation.

        true ranks: [1, 2, 3, 4, 5]
        pred ranks: [1, 3, 2, 5, 4]
        d^2 = [0, 1, 1, 1, 1] = 4
        rho = 1 - 6*4 / (5*(25-1)) = 1 - 24/120 = 1 - 0.2 = 0.8
        """
        true_te = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        pred_te = np.array([10.0, 30.0, 20.0, 50.0, 40.0])
        result = rank_correlation(true_te, pred_te)
        assert result == pytest.approx(0.8, abs=1e-10)


class TestAIC:
    """Tests for aic function."""

    def test_aic_formula(self) -> None:
        """AIC = -2*LL + 2*k, verify with known values."""
        # LL = -100.0, k = 5 -> AIC = 200 + 10 = 210
        result = aic(log_likelihood=-100.0, n_params=5)
        assert result == pytest.approx(210.0, abs=1e-12)


class TestBIC:
    """Tests for bic function."""

    def test_bic_formula(self) -> None:
        """BIC = -2*LL + k*ln(n), verify with known values."""
        # LL = -100.0, k = 5, n = 100 -> BIC = 200 + 5*ln(100)
        expected = 200.0 + 5.0 * math.log(100.0)
        result = bic(log_likelihood=-100.0, n_params=5, n_obs=100)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_aic_less_than_bic_for_large_n(self) -> None:
        """For large n and same LL/k, BIC > AIC (since ln(n) > 2 for n > e^2 ~ 7.4)."""
        ll = -50.0
        k = 3
        n = 1000
        aic_val = aic(log_likelihood=ll, n_params=k)
        bic_val = bic(log_likelihood=ll, n_params=k, n_obs=n)
        assert bic_val > aic_val


class TestFrontierMSE:
    """Tests for frontier_mse function."""

    def test_frontier_mse_perfect(self) -> None:
        """MSE is 0.0 when identical."""
        true_f = np.array([1.0, 2.0, 3.0])
        pred_f = np.array([1.0, 2.0, 3.0])
        result = frontier_mse(true_f, pred_f)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_frontier_mse_known_value(self) -> None:
        """Hand-computed: true=[1,2,3], pred=[2,3,4] -> MSE=1.0."""
        true_f = np.array([1.0, 2.0, 3.0])
        pred_f = np.array([2.0, 3.0, 4.0])
        result = frontier_mse(true_f, pred_f)
        assert result == pytest.approx(1.0, abs=1e-12)


class TestCoverageRate:
    """Tests for coverage_rate function."""

    def test_coverage_rate_perfect(self) -> None:
        """1.0 when all values within bounds."""
        true_vals = np.array([0.5, 1.0, 1.5])
        lower = np.array([0.0, 0.5, 1.0])
        upper = np.array([1.0, 1.5, 2.0])
        result = coverage_rate(true_vals, lower, upper)
        assert result == pytest.approx(1.0, abs=1e-12)

    def test_coverage_rate_none(self) -> None:
        """0.0 when no values within bounds."""
        true_vals = np.array([5.0, 6.0, 7.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        result = coverage_rate(true_vals, lower, upper)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_coverage_rate_partial(self) -> None:
        """Known fraction: 2 out of 4 = 0.5."""
        true_vals = np.array([0.5, 5.0, 0.8, 10.0])
        lower = np.array([0.0, 0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0, 1.0])
        result = coverage_rate(true_vals, lower, upper)
        assert result == pytest.approx(0.5, abs=1e-12)


class TestInputValidation:
    """Tests for input validation across all functions."""

    def test_mismatched_array_lengths_raises(self) -> None:
        """ValueError for all functions with mismatched array lengths."""
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="same length"):
            rmse_efficiency(a, b)

        with pytest.raises(ValueError, match="same length"):
            rank_correlation(a, b)

        with pytest.raises(ValueError, match="same length"):
            frontier_mse(a, b)

        with pytest.raises(ValueError, match="same length"):
            coverage_rate(a, a, b)

        with pytest.raises(ValueError, match="same length"):
            coverage_rate(b, a, a)

    def test_empty_arrays_raises(self) -> None:
        """ValueError for empty inputs."""
        empty = np.array([])
        non_empty = np.array([1.0])

        with pytest.raises(ValueError, match="empty"):
            rmse_efficiency(empty, empty)

        with pytest.raises(ValueError, match="empty"):
            rank_correlation(empty, empty)

        with pytest.raises(ValueError, match="empty"):
            frontier_mse(empty, empty)

        with pytest.raises(ValueError, match="empty"):
            coverage_rate(empty, empty, empty)

        # Also test that one empty and one non-empty raises
        with pytest.raises(ValueError, match="(empty|same length)"):
            rmse_efficiency(empty, non_empty)
