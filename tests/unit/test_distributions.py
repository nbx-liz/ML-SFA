"""Tests for inefficiency distribution module.

Covers HalfNormal, TruncatedNormal, and Exponential distributions
with JLMS conditional moment formulas.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from ml_sfa.utils.distributions import (
    Exponential,
    HalfNormal,
    InefficiencyDistribution,
    TruncatedNormal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def half_normal() -> HalfNormal:
    """HalfNormal distribution instance."""
    return HalfNormal()


@pytest.fixture
def truncated_normal() -> TruncatedNormal:
    """TruncatedNormal distribution instance (mu=0)."""
    return TruncatedNormal(mu=0.0)


@pytest.fixture
def truncated_normal_nonzero() -> TruncatedNormal:
    """TruncatedNormal distribution instance (mu=0.5)."""
    return TruncatedNormal(mu=0.5)


@pytest.fixture
def exponential() -> Exponential:
    """Exponential distribution instance."""
    return Exponential()


@pytest.fixture
def u_values() -> np.ndarray:
    """Positive test values for u (inefficiency)."""
    return np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0])


@pytest.fixture
def epsilon_values() -> np.ndarray:
    """Composed error values (can be negative or positive)."""
    return np.array([-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0])


# ---------------------------------------------------------------------------
# HalfNormal tests
# ---------------------------------------------------------------------------


class TestHalfNormal:
    """Tests for HalfNormal inefficiency distribution."""

    @pytest.mark.parametrize("sigma_u", [0.1, 0.3, 0.5, 1.0, 2.0])
    def test_half_normal_log_pdf_matches_scipy(
        self,
        half_normal: HalfNormal,
        u_values: np.ndarray,
        sigma_u: float,
    ) -> None:
        """log_pdf should match scipy.stats.halfnorm.logpdf."""
        result = half_normal.log_pdf(u_values, sigma_u)
        expected = stats.halfnorm.logpdf(u_values, scale=sigma_u)
        assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("sigma_u", [0.1, 0.3, 0.5, 1.0, 2.0])
    def test_half_normal_cdf_matches_scipy(
        self,
        half_normal: HalfNormal,
        u_values: np.ndarray,
        sigma_u: float,
    ) -> None:
        """cdf should match scipy.stats.halfnorm.cdf."""
        result = half_normal.cdf(u_values, sigma_u)
        expected = stats.halfnorm.cdf(u_values, scale=sigma_u)
        assert_allclose(result, expected, rtol=1e-10)

    def test_half_normal_conditional_mean_jlms(
        self,
        half_normal: HalfNormal,
        epsilon_values: np.ndarray,
    ) -> None:
        """JLMS conditional mean E[u|epsilon] for half-normal.

        mu_star = -epsilon * sigma_u^2 / sigma^2
        sigma_star^2 = sigma_u^2 * sigma_v^2 / sigma^2
        E[u|eps] = mu_star + sigma_star * phi(r) / Phi(r)
        where r = mu_star / sigma_star.
        """
        sigma_v = 0.2
        sigma_u = 0.3
        sigma_sq = sigma_v**2 + sigma_u**2

        mu_star = -epsilon_values * sigma_u**2 / sigma_sq
        sigma_star = np.sqrt(sigma_u**2 * sigma_v**2 / sigma_sq)

        ratio = mu_star / sigma_star
        expected = mu_star + sigma_star * (
            stats.norm.pdf(ratio) / stats.norm.cdf(ratio)
        )

        result = half_normal.conditional_mean(epsilon_values, sigma_v, sigma_u)
        assert_allclose(result, expected, rtol=1e-8)
        # Must be non-negative
        assert np.all(result >= 0.0)

    def test_half_normal_conditional_mode_jlms(
        self,
        half_normal: HalfNormal,
        epsilon_values: np.ndarray,
    ) -> None:
        """JLMS conditional mode is max(0, mu_star)."""
        sigma_v = 0.2
        sigma_u = 0.3
        sigma_sq = sigma_v**2 + sigma_u**2
        mu_star = -epsilon_values * sigma_u**2 / sigma_sq

        expected = np.maximum(0.0, mu_star)
        result = half_normal.conditional_mode(epsilon_values, sigma_v, sigma_u)
        assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# TruncatedNormal tests
# ---------------------------------------------------------------------------


class TestTruncatedNormal:
    """Tests for TruncatedNormal inefficiency distribution."""

    @pytest.mark.parametrize("sigma_u", [0.3, 0.5, 1.0])
    def test_truncated_normal_log_pdf_matches_scipy(
        self,
        truncated_normal_nonzero: TruncatedNormal,
        u_values: np.ndarray,
        sigma_u: float,
    ) -> None:
        """log_pdf should match scipy.stats.truncnorm for lower=0."""
        mu = truncated_normal_nonzero.mu
        # scipy truncnorm parameterization: a = (lower - mu) / sigma, b = inf
        a = (0.0 - mu) / sigma_u
        b = np.inf
        expected = stats.truncnorm.logpdf(u_values, a, b, loc=mu, scale=sigma_u)
        result = truncated_normal_nonzero.log_pdf(u_values, sigma_u)
        assert_allclose(result, expected, rtol=1e-10)

    def test_truncated_normal_reduces_to_half_normal_when_mu_zero(
        self,
        half_normal: HalfNormal,
        truncated_normal: TruncatedNormal,
        u_values: np.ndarray,
    ) -> None:
        """TruncatedNormal(mu=0) should produce identical results to HalfNormal."""
        sigma_u = 0.3
        assert_allclose(
            truncated_normal.log_pdf(u_values, sigma_u),
            half_normal.log_pdf(u_values, sigma_u),
            rtol=1e-10,
        )
        assert_allclose(
            truncated_normal.cdf(u_values, sigma_u),
            half_normal.cdf(u_values, sigma_u),
            rtol=1e-10,
        )

        epsilon = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        sigma_v = 0.2
        assert_allclose(
            truncated_normal.conditional_mean(epsilon, sigma_v, sigma_u),
            half_normal.conditional_mean(epsilon, sigma_v, sigma_u),
            rtol=1e-8,
        )
        assert_allclose(
            truncated_normal.conditional_mode(epsilon, sigma_v, sigma_u),
            half_normal.conditional_mode(epsilon, sigma_v, sigma_u),
            rtol=1e-10,
        )

    def test_truncated_normal_conditional_mean_jlms(
        self,
        truncated_normal_nonzero: TruncatedNormal,
        epsilon_values: np.ndarray,
    ) -> None:
        """JLMS conditional mean for truncated normal with mu != 0.

        mu_star = (-epsilon * sigma_u^2 + mu * sigma_v^2) / sigma^2
        sigma_star^2 = sigma_u^2 * sigma_v^2 / sigma^2
        E[u|eps] = mu_star + sigma_star * phi(r) / Phi(r)
        where r = mu_star / sigma_star.
        """
        sigma_v = 0.2
        sigma_u = 0.3
        mu = truncated_normal_nonzero.mu
        sigma_sq = sigma_v**2 + sigma_u**2

        mu_star = (-epsilon_values * sigma_u**2 + mu * sigma_v**2) / sigma_sq
        sigma_star = np.sqrt(sigma_u**2 * sigma_v**2 / sigma_sq)

        ratio = mu_star / sigma_star
        expected = mu_star + sigma_star * (
            stats.norm.pdf(ratio) / stats.norm.cdf(ratio)
        )

        result = truncated_normal_nonzero.conditional_mean(
            epsilon_values, sigma_v, sigma_u
        )
        assert_allclose(result, expected, rtol=1e-8)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# Exponential tests
# ---------------------------------------------------------------------------


class TestExponential:
    """Tests for Exponential inefficiency distribution."""

    @pytest.mark.parametrize("sigma_u", [0.1, 0.3, 0.5, 1.0, 2.0])
    def test_exponential_log_pdf_matches_scipy(
        self,
        exponential: Exponential,
        u_values: np.ndarray,
        sigma_u: float,
    ) -> None:
        """log_pdf should match scipy.stats.expon.logpdf."""
        result = exponential.log_pdf(u_values, sigma_u)
        expected = stats.expon.logpdf(u_values, scale=sigma_u)
        assert_allclose(result, expected, rtol=1e-10)

    def test_exponential_conditional_mean(
        self,
        exponential: Exponential,
        epsilon_values: np.ndarray,
    ) -> None:
        """JLMS conditional mean for exponential distribution.

        mu_star = -epsilon - sigma_v^2 / sigma_u
        E[u|epsilon] = mu_star + sigma_v * phi(mu_star/sigma_v) / Phi(mu_star/sigma_v)
        """
        sigma_v = 0.2
        sigma_u = 0.3

        mu_star = -epsilon_values - sigma_v**2 / sigma_u
        ratio = mu_star / sigma_v
        expected = mu_star + sigma_v * (stats.norm.pdf(ratio) / stats.norm.cdf(ratio))

        result = exponential.conditional_mean(epsilon_values, sigma_v, sigma_u)
        assert_allclose(result, expected, rtol=1e-8)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# Protocol and edge case tests
# ---------------------------------------------------------------------------


class TestProtocolAndEdgeCases:
    """Cross-cutting tests for protocol conformance and numerical stability."""

    def test_all_distributions_satisfy_protocol(self) -> None:
        """All distribution classes must satisfy InefficiencyDistribution protocol."""
        assert isinstance(HalfNormal(), InefficiencyDistribution)
        assert isinstance(TruncatedNormal(), InefficiencyDistribution)
        assert isinstance(Exponential(), InefficiencyDistribution)

    def test_numerical_stability_large_epsilon(self) -> None:
        """No NaN or Inf for extreme epsilon values."""
        epsilon = np.linspace(-10.0, 10.0, 201)
        sigma_v = 0.2
        sigma_u = 0.3

        for dist in [HalfNormal(), TruncatedNormal(mu=0.5), Exponential()]:
            mean = dist.conditional_mean(epsilon, sigma_v, sigma_u)
            mode = dist.conditional_mode(epsilon, sigma_v, sigma_u)
            name = type(dist).__name__
            assert np.all(np.isfinite(mean)), f"{name} mean has non-finite values"
            assert np.all(np.isfinite(mode)), f"{name} mode has non-finite values"

    def test_immutability_input_arrays(self) -> None:
        """Input arrays must not be mutated by any method."""
        epsilon = np.array([-1.0, 0.0, 1.0])
        u = np.array([0.1, 0.5, 1.0])
        epsilon_orig = epsilon.copy()
        u_orig = u.copy()

        for dist in [HalfNormal(), TruncatedNormal(mu=0.5), Exponential()]:
            dist.log_pdf(u, 0.3)
            assert_allclose(u, u_orig)

            dist.cdf(u, 0.3)
            assert_allclose(u, u_orig)

            dist.conditional_mean(epsilon, 0.2, 0.3)
            assert_allclose(epsilon, epsilon_orig)

            dist.conditional_mode(epsilon, 0.2, 0.3)
            assert_allclose(epsilon, epsilon_orig)

    def test_conditional_mean_non_negative(self) -> None:
        """E[u|epsilon] must be >= 0 for all distributions."""
        epsilon = np.linspace(-5.0, 5.0, 101)
        sigma_v = 0.2
        sigma_u = 0.3

        dists = [
            HalfNormal(),
            TruncatedNormal(mu=0.0),
            TruncatedNormal(mu=0.5),
            Exponential(),
        ]
        for dist in dists:
            result = dist.conditional_mean(epsilon, sigma_v, sigma_u)
            assert np.all(result >= 0.0), (
                f"{type(dist).__name__} produced negative conditional mean"
            )
