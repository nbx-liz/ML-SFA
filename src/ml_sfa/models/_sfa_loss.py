"""PyTorch SFA negative log-likelihood functions.

Provides differentiable NLL functions for half-normal, exponential, and
truncated-normal inefficiency distributions.  These are used by
``NNFrontierSFA`` for joint estimation via autograd.

Numerical stability is ensured via ``torch.special.log_ndtr``.

Requires ``torch`` (install via ``pip install ml-sfa[nn]``).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "sfa_nll_exponential",
    "sfa_nll_half_normal",
    "sfa_nll_truncated_normal",
]

_LOG_2_OVER_PI = math.log(2.0 / math.pi)
_LOG_2PI = math.log(2.0 * math.pi)


def sfa_nll_half_normal(
    epsilon: torch.Tensor,
    log_sigma_v: float | torch.Tensor,
    log_sigma_u: float | torch.Tensor,
    *,
    cost: bool = False,
) -> torch.Tensor:
    """Negative log-likelihood for half-normal SFA model.

    Parameters
    ----------
    epsilon : torch.Tensor
        Composed error residuals ``y - f(x)`` of shape ``(n,)``.
    log_sigma_v : float or torch.Tensor
        Log of noise standard deviation.
    log_sigma_u : float or torch.Tensor
        Log of inefficiency scale parameter.
    cost : bool
        If ``True``, use cost frontier convention.

    Returns
    -------
    torch.Tensor
        Scalar negative log-likelihood.
    """
    sigma_v = torch.exp(torch.as_tensor(log_sigma_v, dtype=epsilon.dtype))
    sigma_u = torch.exp(torch.as_tensor(log_sigma_u, dtype=epsilon.dtype))

    sigma_sq = sigma_v**2 + sigma_u**2
    sigma = torch.sqrt(sigma_sq)
    lam = sigma_u / sigma_v

    phi_sign = 1.0 if cost else -1.0

    n = epsilon.shape[0]
    ll = (
        -n * torch.log(sigma)
        + torch.sum(torch.special.log_ndtr(phi_sign * epsilon * lam / sigma))
        - torch.sum(epsilon**2) / (2.0 * sigma_sq)
        + 0.5 * n * _LOG_2_OVER_PI
    )
    return -ll


def sfa_nll_exponential(
    epsilon: torch.Tensor,
    log_sigma_v: float | torch.Tensor,
    log_sigma_u: float | torch.Tensor,
    *,
    cost: bool = False,
) -> torch.Tensor:
    """Negative log-likelihood for exponential SFA model.

    Parameters
    ----------
    epsilon : torch.Tensor
        Composed error residuals ``y - f(x)`` of shape ``(n,)``.
    log_sigma_v : float or torch.Tensor
        Log of noise standard deviation.
    log_sigma_u : float or torch.Tensor
        Log of inefficiency scale parameter.
    cost : bool
        If ``True``, use cost frontier convention.

    Returns
    -------
    torch.Tensor
        Scalar negative log-likelihood.
    """
    sigma_v = torch.exp(torch.as_tensor(log_sigma_v, dtype=epsilon.dtype))
    sigma_u = torch.exp(torch.as_tensor(log_sigma_u, dtype=epsilon.dtype))

    eps_adj = -epsilon if cost else epsilon

    n = epsilon.shape[0]
    ll = (
        -n * torch.log(sigma_u)
        + n * sigma_v**2 / (2.0 * sigma_u**2)
        + torch.sum(
            torch.special.log_ndtr(-eps_adj / sigma_v - sigma_v / sigma_u)
            + eps_adj / sigma_u
        )
    )
    return -ll


def sfa_nll_truncated_normal(
    epsilon: torch.Tensor,
    log_sigma_v: float | torch.Tensor,
    log_sigma_u: float | torch.Tensor,
    *,
    mu: float = 0.0,
    cost: bool = False,
) -> torch.Tensor:
    """Negative log-likelihood for truncated-normal SFA model.

    Parameters
    ----------
    epsilon : torch.Tensor
        Composed error residuals ``y - f(x)`` of shape ``(n,)``.
    log_sigma_v : float or torch.Tensor
        Log of noise standard deviation.
    log_sigma_u : float or torch.Tensor
        Log of inefficiency scale parameter.
    mu : float
        Location parameter of the pre-truncation normal.
    cost : bool
        If ``True``, use cost frontier convention.

    Returns
    -------
    torch.Tensor
        Scalar negative log-likelihood.
    """
    sigma_v = torch.exp(torch.as_tensor(log_sigma_v, dtype=epsilon.dtype))
    sigma_u = torch.exp(torch.as_tensor(log_sigma_u, dtype=epsilon.dtype))

    sigma_sq = sigma_v**2 + sigma_u**2
    sigma = torch.sqrt(sigma_sq)
    sigma_star = torch.sqrt(sigma_u**2 * sigma_v**2 / sigma_sq)

    eps_adj = -epsilon if cost else epsilon

    mu_star = (-eps_adj * sigma_u**2 + mu * sigma_v**2) / sigma_sq

    n = epsilon.shape[0]
    ll = (
        -n * torch.special.log_ndtr(torch.as_tensor(mu / sigma_u, dtype=epsilon.dtype))
        - n * torch.log(sigma)
        + torch.sum(torch.special.log_ndtr(mu_star / sigma_star))
        - torch.sum(eps_adj**2) / (2.0 * sigma_sq)
        + mu * torch.sum(eps_adj) / sigma_sq
        - 0.5 * n * mu**2 / sigma_sq
        - 0.5 * n * _LOG_2PI
        + 0.5 * n * math.log(2.0)
    )
    result: torch.Tensor = -ll
    return result
