"""Inefficiency distribution implementations for SFA models.

Provides HalfNormal, TruncatedNormal, and Exponential distributions
with JLMS (Jondrow et al., 1982) conditional moment formulas for
estimating technical inefficiency from composed error residuals.

Numerical stability is ensured via ``scipy.special.log_ndtr`` and
Mills-ratio approximations for extreme values.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm

from ml_sfa._types import FloatArray

__all__ = [
    "Exponential",
    "HalfNormal",
    "InefficiencyDistribution",
    "TruncatedNormal",
]

# Constants
_LOG_2 = np.log(2.0)
_LOG_2PI = np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InefficiencyDistribution(Protocol):
    """Protocol for one-sided inefficiency distributions.

    All methods accept and return numpy arrays.  Implementations must
    be stateless with respect to the sigma parameters (passed per-call)
    so that the same instance can be reused across fits.
    """

    def log_pdf(self, u: FloatArray, sigma_u: float) -> FloatArray:
        """Log probability density for *u >= 0*.

        Args:
            u: Non-negative inefficiency values.
            sigma_u: Scale parameter of the distribution.

        Returns:
            Element-wise log-density values.
        """
        ...

    def cdf(self, u: FloatArray, sigma_u: float) -> FloatArray:
        """Cumulative distribution function for *u >= 0*.

        Args:
            u: Non-negative inefficiency values.
            sigma_u: Scale parameter of the distribution.

        Returns:
            Element-wise CDF values.
        """
        ...

    def conditional_mean(
        self,
        epsilon: FloatArray,
        sigma_v: float,
        sigma_u: float,
    ) -> FloatArray:
        """JLMS conditional expectation E[u | epsilon].

        Args:
            epsilon: Composed error residuals (y - frontier).
            sigma_v: Noise standard deviation.
            sigma_u: Inefficiency scale parameter.

        Returns:
            Non-negative conditional mean estimates.
        """
        ...

    def conditional_mode(
        self,
        epsilon: FloatArray,
        sigma_v: float,
        sigma_u: float,
    ) -> FloatArray:
        """JLMS conditional mode of u | epsilon.

        Args:
            epsilon: Composed error residuals (y - frontier).
            sigma_v: Noise standard deviation.
            sigma_u: Inefficiency scale parameter.

        Returns:
            Non-negative conditional mode estimates.
        """
        ...


# ---------------------------------------------------------------------------
# Helper: numerically stable Mills ratio  phi(x) / Phi(x)
# ---------------------------------------------------------------------------


def _mills_ratio(x: FloatArray) -> FloatArray:
    """Compute phi(x) / Phi(x) in a numerically stable way.

    Uses ``log_ndtr`` to avoid log(0) for large negative *x*.

    Args:
        x: Input array.

    Returns:
        Element-wise Mills ratio values.
    """
    log_phi = -0.5 * (x * x + _LOG_2PI)
    log_Phi = log_ndtr(x)
    return np.exp(log_phi - log_Phi)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# HalfNormal
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class HalfNormal:
    """Half-normal inefficiency distribution.

    The density for u >= 0 is:
        f(u) = (2 / sigma_u) * phi(u / sigma_u)

    where phi is the standard normal PDF.
    """

    def log_pdf(self, u: FloatArray, sigma_u: float) -> FloatArray:
        """Log-density of half-normal distribution.

        ``log f(u) = -(u^2)/(2*sigma_u^2) - ln(sigma_u) + ln(2) - 0.5*ln(2*pi)``

        Args:
            u: Non-negative inefficiency values.
            sigma_u: Scale parameter (> 0).

        Returns:
            Element-wise log-density values.
        """
        z = u / sigma_u
        return -(z * z) / 2.0 - np.log(sigma_u) + _LOG_2 - 0.5 * _LOG_2PI  # type: ignore[no-any-return]

    def cdf(self, u: FloatArray, sigma_u: float) -> FloatArray:
        """CDF of half-normal: ``2*Phi(u/sigma_u) - 1`` for u >= 0.

        Args:
            u: Non-negative inefficiency values.
            sigma_u: Scale parameter (> 0).

        Returns:
            Element-wise CDF values.
        """
        return 2.0 * norm.cdf(u / sigma_u) - 1.0  # type: ignore[no-any-return]

    def conditional_mean(
        self,
        epsilon: FloatArray,
        sigma_v: float,
        sigma_u: float,
    ) -> FloatArray:
        """JLMS E[u | epsilon] for half-normal inefficiency.

        Args:
            epsilon: Composed error residuals.
            sigma_v: Noise standard deviation (> 0).
            sigma_u: Inefficiency scale parameter (> 0).

        Returns:
            Non-negative conditional mean array.
        """
        sigma_sq = sigma_v**2 + sigma_u**2
        mu_star = -epsilon * (sigma_u**2 / sigma_sq)
        sigma_star = np.sqrt(sigma_u**2 * sigma_v**2 / sigma_sq)

        ratio = mu_star / sigma_star
        result = mu_star + sigma_star * _mills_ratio(ratio)
        return np.maximum(result, 0.0)  # type: ignore[no-any-return]

    def conditional_mode(
        self,
        epsilon: FloatArray,
        sigma_v: float,
        sigma_u: float,
    ) -> FloatArray:
        """JLMS conditional mode: ``max(0, mu_star)``.

        Args:
            epsilon: Composed error residuals.
            sigma_v: Noise standard deviation (> 0).
            sigma_u: Inefficiency scale parameter (> 0).

        Returns:
            Non-negative conditional mode array.
        """
        sigma_sq = sigma_v**2 + sigma_u**2
        mu_star = -epsilon * (sigma_u**2 / sigma_sq)
        return np.maximum(0.0, mu_star)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# TruncatedNormal
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class TruncatedNormal:
    """Truncated-normal inefficiency distribution (truncated at 0 from below).

    When ``mu=0`` this reduces to the half-normal distribution.

    The density for u >= 0 is:
        f(u) = phi((u - mu) / sigma_u) / (sigma_u * Phi(mu / sigma_u))

    Args:
        mu: Location parameter of the pre-truncation normal.
    """

    mu: float = 0.0

    def log_pdf(self, u: FloatArray, sigma_u: float) -> FloatArray:
        """Log-density of truncated normal.

        Args:
            u: Non-negative inefficiency values.
            sigma_u: Scale parameter (> 0).

        Returns:
            Element-wise log-density values.
        """
        z = (u - self.mu) / sigma_u
        # log phi(z) - log sigma_u - log Phi(mu/sigma_u)
        log_phi_z = -0.5 * (z * z + _LOG_2PI)
        log_norm = log_ndtr(self.mu / sigma_u)
        return log_phi_z - np.log(sigma_u) - log_norm  # type: ignore[no-any-return]

    def cdf(self, u: FloatArray, sigma_u: float) -> FloatArray:
        """CDF of truncated normal for u >= 0.

        ``F(u) = [Phi((u - mu)/sigma_u) - Phi(-mu/sigma_u)] / Phi(mu/sigma_u)``

        Args:
            u: Non-negative inefficiency values.
            sigma_u: Scale parameter (> 0).

        Returns:
            Element-wise CDF values.
        """
        Phi_upper = norm.cdf((u - self.mu) / sigma_u)
        Phi_lower = norm.cdf(-self.mu / sigma_u)
        Phi_norm = norm.cdf(self.mu / sigma_u)
        return (Phi_upper - Phi_lower) / Phi_norm  # type: ignore[no-any-return]

    def conditional_mean(
        self,
        epsilon: FloatArray,
        sigma_v: float,
        sigma_u: float,
    ) -> FloatArray:
        """JLMS E[u | epsilon] for truncated-normal inefficiency.

        Generalized formula:
            mu_star = (-epsilon * sigma_u^2 + mu * sigma_v^2) / sigma^2
            sigma_star^2 = sigma_u^2 * sigma_v^2 / sigma^2

        Args:
            epsilon: Composed error residuals.
            sigma_v: Noise standard deviation (> 0).
            sigma_u: Inefficiency scale parameter (> 0).

        Returns:
            Non-negative conditional mean array.
        """
        sigma_sq = sigma_v**2 + sigma_u**2
        mu_star = (-epsilon * sigma_u**2 + self.mu * sigma_v**2) / sigma_sq
        sigma_star = np.sqrt(sigma_u**2 * sigma_v**2 / sigma_sq)

        ratio = mu_star / sigma_star
        result = mu_star + sigma_star * _mills_ratio(ratio)
        return np.maximum(result, 0.0)  # type: ignore[no-any-return]

    def conditional_mode(
        self,
        epsilon: FloatArray,
        sigma_v: float,
        sigma_u: float,
    ) -> FloatArray:
        """JLMS conditional mode: ``max(0, mu_star)``.

        Args:
            epsilon: Composed error residuals.
            sigma_v: Noise standard deviation (> 0).
            sigma_u: Inefficiency scale parameter (> 0).

        Returns:
            Non-negative conditional mode array.
        """
        sigma_sq = sigma_v**2 + sigma_u**2
        mu_star = (-epsilon * sigma_u**2 + self.mu * sigma_v**2) / sigma_sq
        return np.maximum(0.0, mu_star)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Exponential:
    """Exponential inefficiency distribution.

    The density for u >= 0 is:
        f(u) = (1 / sigma_u) * exp(-u / sigma_u)
    """

    def log_pdf(self, u: FloatArray, sigma_u: float) -> FloatArray:
        """Log-density of exponential distribution.

        ``log f(u) = -ln(sigma_u) - u / sigma_u``

        Args:
            u: Non-negative inefficiency values.
            sigma_u: Scale parameter (> 0).

        Returns:
            Element-wise log-density values.
        """
        return -np.log(sigma_u) - u / sigma_u  # type: ignore[no-any-return]

    def cdf(self, u: FloatArray, sigma_u: float) -> FloatArray:
        """CDF of exponential distribution for u >= 0.

        ``F(u) = 1 - exp(-u / sigma_u)``

        Args:
            u: Non-negative inefficiency values.
            sigma_u: Scale parameter (> 0).

        Returns:
            Element-wise CDF values.
        """
        return 1.0 - np.exp(-u / sigma_u)  # type: ignore[no-any-return]

    def conditional_mean(
        self,
        epsilon: FloatArray,
        sigma_v: float,
        sigma_u: float,
    ) -> FloatArray:
        """JLMS E[u | epsilon] for exponential inefficiency.

        ``mu_star = -epsilon - sigma_v^2 / sigma_u``
        ``E[u|e] = mu_star + sigma_v * phi(mu_star/sigma_v) / Phi(mu_star/sigma_v)``

        Args:
            epsilon: Composed error residuals.
            sigma_v: Noise standard deviation (> 0).
            sigma_u: Inefficiency scale parameter (> 0).

        Returns:
            Non-negative conditional mean array.
        """
        mu_star = -epsilon - sigma_v**2 / sigma_u
        ratio = mu_star / sigma_v
        result = mu_star + sigma_v * _mills_ratio(ratio)
        return np.maximum(result, 0.0)

    def conditional_mode(
        self,
        epsilon: FloatArray,
        sigma_v: float,
        sigma_u: float,
    ) -> FloatArray:
        """JLMS conditional mode for exponential: ``max(0, mu_star)``.

        Args:
            epsilon: Composed error residuals.
            sigma_v: Noise standard deviation (> 0).
            sigma_u: Inefficiency scale parameter (> 0).

        Returns:
            Non-negative conditional mode array.
        """
        mu_star = -epsilon - sigma_v**2 / sigma_u
        return np.maximum(0.0, mu_star)  # type: ignore[no-any-return]
