"""Unit tests for the SFA simulation data generator."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ml_sfa.data.simulator import simulate_sfa


class TestSFADatasetFrozen:
    """Tests for the SFADataset frozen dataclass."""

    def test_sfa_dataset_is_frozen_dataclass(self) -> None:
        """SFADataset instances must be immutable (frozen)."""
        ds = simulate_sfa(n_obs=10, n_inputs=2, seed=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ds.n_obs = 999  # type: ignore[misc]


class TestCobbDouglasShapes:
    """Tests for Cobb-Douglas DGP output shapes."""

    def test_simulate_cobb_douglas_shapes(self) -> None:
        """X is (n, p), y is (n,), te/u/v are (n,), beta has correct length."""
        n, p = 100, 3
        ds = simulate_sfa(
            n_obs=n,
            n_inputs=p,
            frontier_type="cobb-douglas",
            seed=42,
        )
        assert ds.X.shape == (n, p)
        assert ds.y.shape == (n,)
        assert ds.te.shape == (n,)
        assert ds.u.shape == (n,)
        assert ds.v.shape == (n,)
        # Cobb-Douglas: intercept + p slopes
        assert len(ds.beta) == p + 1
        assert ds.n_obs == n
        assert ds.n_inputs == p


class TestTranslogShapes:
    """Tests for Translog DGP output shapes."""

    def test_simulate_translog_shapes(self) -> None:
        """Beta includes intercept, linear, squared, and interaction terms."""
        n, p = 100, 3
        ds = simulate_sfa(
            n_obs=n,
            n_inputs=p,
            frontier_type="translog",
            seed=42,
        )
        assert ds.X.shape == (n, p)
        assert ds.y.shape == (n,)
        # Translog: intercept + p linear + p squared + C(p,2) interactions
        n_cross = p * (p - 1) // 2
        expected_beta_len = 1 + p + p + n_cross
        assert len(ds.beta) == expected_beta_len


class TestReproducibility:
    """Tests for seed-based reproducibility."""

    def test_simulate_cobb_douglas_reproducible(self) -> None:
        """Same seed produces identical output."""
        ds1 = simulate_sfa(n_obs=50, seed=123)
        ds2 = simulate_sfa(n_obs=50, seed=123)
        np.testing.assert_array_equal(ds1.X, ds2.X)
        np.testing.assert_array_equal(ds1.y, ds2.y)
        np.testing.assert_array_equal(ds1.u, ds2.u)
        np.testing.assert_array_equal(ds1.v, ds2.v)
        np.testing.assert_array_equal(ds1.te, ds2.te)

    def test_simulate_different_seeds_differ(self) -> None:
        """Different seeds produce different data."""
        ds1 = simulate_sfa(n_obs=50, seed=1)
        ds2 = simulate_sfa(n_obs=50, seed=2)
        assert not np.array_equal(ds1.X, ds2.X)
        assert not np.array_equal(ds1.y, ds2.y)


class TestEfficiencyAndInefficiency:
    """Tests for TE and u properties."""

    def test_true_te_between_0_and_1(self) -> None:
        """TE = exp(-u) must be in (0, 1]."""
        ds = simulate_sfa(n_obs=500, seed=42)
        assert np.all(ds.te > 0.0)
        assert np.all(ds.te <= 1.0)

    def test_true_u_non_negative(self) -> None:
        """Inefficiency u must be >= 0."""
        ds = simulate_sfa(n_obs=500, seed=42)
        assert np.all(ds.u >= 0.0)


class TestCompositeError:
    """Tests for composite error decomposition."""

    def test_composite_error_decomposition(self) -> None:
        """For production frontier: y = frontier + v - u."""
        ds = simulate_sfa(
            n_obs=200,
            n_inputs=2,
            frontier_type="cobb-douglas",
            cost=False,
            seed=42,
        )
        ln_x = np.log(ds.X)
        frontier = ds.beta[0] + ln_x @ ds.beta[1:]
        expected_y = frontier + ds.v - ds.u
        np.testing.assert_allclose(ds.y, expected_y, atol=1e-12)

    def test_cost_frontier_sign_flip(self) -> None:
        """For cost frontier: y = frontier + v + u."""
        ds = simulate_sfa(
            n_obs=200,
            n_inputs=2,
            frontier_type="cobb-douglas",
            cost=True,
            seed=42,
        )
        ln_x = np.log(ds.X)
        frontier = ds.beta[0] + ln_x @ ds.beta[1:]
        expected_y = frontier + ds.v + ds.u
        np.testing.assert_allclose(ds.y, expected_y, atol=1e-12)


class TestInputValidation:
    """Tests for invalid parameter handling."""

    def test_invalid_frontier_type_raises_valueerror(self) -> None:
        """Unknown frontier_type must raise ValueError."""
        with pytest.raises(ValueError, match="frontier_type"):
            simulate_sfa(frontier_type="unknown")  # type: ignore[arg-type]

    def test_invalid_inefficiency_dist_raises_valueerror(self) -> None:
        """Unknown inefficiency_dist must raise ValueError."""
        with pytest.raises(ValueError, match="inefficiency_dist"):
            simulate_sfa(inefficiency_dist="gamma")  # type: ignore[arg-type]


class TestInefficiencyDistributions:
    """Tests for statistical properties of u distributions."""

    def test_half_normal_u_distribution(self) -> None:
        """Half-normal u has mean ≈ sigma_u * sqrt(2/pi)."""
        sigma_u = 0.3
        ds = simulate_sfa(
            n_obs=50_000,
            inefficiency_dist="half-normal",
            sigma_u=sigma_u,
            seed=42,
        )
        expected_mean = sigma_u * np.sqrt(2.0 / np.pi)
        assert abs(ds.u.mean() - expected_mean) < 0.01

    def test_exponential_u_distribution(self) -> None:
        """Exponential u has mean ≈ sigma_u."""
        sigma_u = 0.25
        ds = simulate_sfa(
            n_obs=50_000,
            inefficiency_dist="exponential",
            sigma_u=sigma_u,
            seed=42,
        )
        assert abs(ds.u.mean() - sigma_u) < 0.01


class TestDefaults:
    """Tests for default parameter handling."""

    def test_default_parameters(self) -> None:
        """simulate_sfa can be called with minimal arguments."""
        ds = simulate_sfa()
        assert ds.n_obs == 500
        assert ds.n_inputs == 3
        assert ds.frontier_type == "cobb-douglas"
        assert ds.inefficiency_dist == "half-normal"
        assert ds.X.shape == (500, 3)
