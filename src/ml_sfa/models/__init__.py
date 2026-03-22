"""SFA models with ML extensions."""

from ml_sfa.models.base import BaseSFAEstimator, SFASummary
from ml_sfa.models.parametric import ParametricSFA

__all__ = ["BaseSFAEstimator", "NNFrontierSFA", "ParametricSFA", "SFASummary"]


def __getattr__(name: str) -> object:
    """Lazy import for optional-dependency models."""
    if name == "NNFrontierSFA":
        from ml_sfa.models.nn_frontier import NNFrontierSFA

        return NNFrontierSFA
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
