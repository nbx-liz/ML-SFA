"""Shared type definitions for SFA models."""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

# Array type for float64 arrays used throughout the library
FloatArray = NDArray[np.floating[Any]]

# Frontier function types
FrontierType = Literal["cobb-douglas", "translog"]

# Inefficiency distribution types
InefficiencyType = Literal["half-normal", "truncated-normal", "exponential"]

__all__ = ["FloatArray", "FrontierType", "InefficiencyType"]
