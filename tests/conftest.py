"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Reproducible random number generator."""
    return np.random.default_rng(seed=42)
