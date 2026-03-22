"""Shared utility functions."""

from ml_sfa.utils.distributions import (
    Exponential,
    HalfNormal,
    InefficiencyDistribution,
    TruncatedNormal,
)

__all__ = [
    "Exponential",
    "HalfNormal",
    "InefficiencyDistribution",
    "MonotonicMLP",
    "TruncatedNormal",
]


def __getattr__(name: str) -> object:
    """Lazy import for optional-dependency modules."""
    if name == "MonotonicMLP":
        from ml_sfa.utils.constraints import MonotonicMLP

        return MonotonicMLP
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
