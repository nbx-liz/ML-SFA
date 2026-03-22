"""Neural Network Stochastic Frontier Analysis with joint estimation.

Implements ``NNFrontierSFA``, which simultaneously optimises an MLP frontier
and SFA error-decomposition parameters (σ_v, σ_u) by maximising the
composed-error log-likelihood via PyTorch autograd.

Two-phase optimisation:
1. **MSE pre-training** — warm-start the NN weights with Adam.
2. **SFA fine-tuning** — switch to the SFA NLL loss with L-BFGS.

Monotonicity (∂f/∂x_j ≥ 0) can be enforced architecturally via
``MonotonicMLP`` (non-negative weights through softplus transform).

Requires ``torch`` (install via ``pip install ml-sfa[nn]``).
"""

from __future__ import annotations

from typing import Self

import numpy as np
import torch
import torch.nn as nn

from ml_sfa._types import FloatArray, InefficiencyType
from ml_sfa.models._sfa_loss import (
    sfa_nll_exponential,
    sfa_nll_half_normal,
    sfa_nll_truncated_normal,
)
from ml_sfa.models.base import BaseSFAEstimator, SFASummary
from ml_sfa.utils.constraints import MonotonicMLP
from ml_sfa.utils.distributions import (
    Exponential,
    HalfNormal,
    InefficiencyDistribution,
    TruncatedNormal,
)

__all__ = ["NNFrontierSFA"]

# ---------------------------------------------------------------------------
# Internal NN module
# ---------------------------------------------------------------------------


class _FrontierNet(nn.Module):
    """Internal MLP + learnable sigma parameters.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    hidden_dims : list[int]
        Hidden layer widths.
    monotonic : bool
        If ``True`` use ``MonotonicMLP``; otherwise a standard MLP.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        monotonic: bool = True,
    ) -> None:
        super().__init__()

        if monotonic:
            self.net: nn.Module = MonotonicMLP(in_dim, hidden_dims)
        else:
            layers: list[nn.Module] = []
            prev = in_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.Softplus())
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        self.log_sigma_v = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_sigma_u = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict frontier values.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(n, p)``.

        Returns
        -------
        torch.Tensor
            Frontier predictions of shape ``(n,)``.
        """
        result: torch.Tensor = self.net(x).squeeze(-1)
        return result


# ---------------------------------------------------------------------------
# NNFrontierSFA estimator
# ---------------------------------------------------------------------------


class NNFrontierSFA(BaseSFAEstimator):
    """Joint-estimation NN-SFA estimator.

    Simultaneously optimises a neural-network frontier and SFA error
    decomposition via maximum likelihood.

    Parameters
    ----------
    inefficiency : InefficiencyType
        Inefficiency distribution: ``"half-normal"``, ``"truncated-normal"``,
        or ``"exponential"``.
    cost : bool
        If ``True``, estimate a cost frontier.
    hidden_dims : list[int]
        Hidden layer widths for the frontier MLP.
    monotonic : bool
        If ``True``, enforce monotonicity via non-negative weight constraints.
    pretrain_epochs : int
        Number of MSE pre-training epochs (Phase 0).
    finetune_epochs : int
        Number of SFA NLL fine-tuning epochs (Phase 1).
    pretrain_lr : float
        Adam learning rate for pre-training.
    weight_decay : float
        L2 regularisation for pre-training.
    n_inits : int
        Number of random initialisations (best log-likelihood wins).
    seed : int or None
        Random seed for reproducibility.
    """

    # frontier param is fixed to "nn" for this class
    _FRONTIER_TYPE = "nn"

    def __init__(
        self,
        inefficiency: InefficiencyType = "half-normal",
        *,
        cost: bool = False,
        hidden_dims: list[int] | None = None,
        monotonic: bool = True,
        pretrain_epochs: int = 200,
        finetune_epochs: int = 20,
        pretrain_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_inits: int = 5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            frontier="cobb-douglas",  # placeholder; overridden in summary
            inefficiency=inefficiency,
            cost=cost,
        )
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 32]
        self.monotonic = monotonic
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.pretrain_lr = pretrain_lr
        self.weight_decay = weight_decay
        self.n_inits = n_inits
        self.seed = seed

    # -- private helpers ----------------------------------------------------

    def _get_distribution(self) -> InefficiencyDistribution:
        """Return the distribution object for JLMS estimation."""
        if self.inefficiency == "half-normal":
            return HalfNormal()
        if self.inefficiency == "truncated-normal":
            return TruncatedNormal(mu=0.0)
        if self.inefficiency == "exponential":
            return Exponential()
        msg = f"Unsupported inefficiency distribution: {self.inefficiency!r}"
        raise ValueError(msg)

    def _compute_nll(
        self,
        epsilon: torch.Tensor,
        log_sigma_v: torch.Tensor,
        log_sigma_u: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NLL for the configured distribution."""
        if self.inefficiency == "half-normal":
            return sfa_nll_half_normal(
                epsilon, log_sigma_v, log_sigma_u, cost=self.cost
            )
        if self.inefficiency == "exponential":
            return sfa_nll_exponential(
                epsilon, log_sigma_v, log_sigma_u, cost=self.cost
            )
        if self.inefficiency == "truncated-normal":
            return sfa_nll_truncated_normal(
                epsilon, log_sigma_v, log_sigma_u, mu=0.0, cost=self.cost
            )
        msg = f"Unsupported: {self.inefficiency!r}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def _build_net(self, in_dim: int) -> _FrontierNet:
        """Build a new _FrontierNet with the configured architecture."""
        return _FrontierNet(
            in_dim=in_dim,
            hidden_dims=self.hidden_dims,
            monotonic=self.monotonic,
        ).double()

    def _init_sigmas(
        self, net: _FrontierNet, X_t: torch.Tensor, y_t: torch.Tensor
    ) -> None:
        """Initialise log_sigma_v/u from MSE residual variance."""
        with torch.no_grad():
            pred = net(X_t)
            residuals = y_t - pred
            var_e = max(float(residuals.var()), 1e-6)
            half_var = float(np.sqrt(var_e / 2.0))
            net.log_sigma_v.fill_(float(np.log(max(half_var, 1e-8))))
            net.log_sigma_u.fill_(float(np.log(max(half_var, 1e-8))))

    def _pretrain(
        self,
        net: _FrontierNet,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
    ) -> None:
        """Phase 0: MSE pre-training with Adam."""
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=self.pretrain_lr,
            weight_decay=self.weight_decay,
        )
        for _ in range(self.pretrain_epochs):
            optimizer.zero_grad()
            pred = net(X_t)
            loss = torch.nn.functional.mse_loss(pred, y_t)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

    def _finetune(
        self,
        net: _FrontierNet,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
    ) -> float:
        """Phase 1: SFA NLL fine-tuning with L-BFGS.

        Returns
        -------
        float
            Final negative log-likelihood value.
        """
        optimizer = torch.optim.LBFGS(
            net.parameters(),
            lr=1.0,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )

        final_nll = float("inf")
        _REL_TOL = 1e-6

        for _ in range(self.finetune_epochs):

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                pred = net(X_t)
                eps = y_t - pred
                nll = self._compute_nll(eps, net.log_sigma_v, net.log_sigma_u)
                nll.backward()  # type: ignore[no-untyped-call]
                return nll

            loss = optimizer.step(closure)  # type: ignore[no-untyped-call]
            if loss is not None:
                val = loss.detach() if isinstance(loss, torch.Tensor) else loss
                current_nll = float(val)
                # Convergence check: relative change in NLL
                if abs(current_nll - final_nll) / (abs(final_nll) + 1e-10) < _REL_TOL:
                    final_nll = current_nll
                    break
                final_nll = current_nll

        return final_nll

    def _fit_single(
        self,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        seed: int,
    ) -> tuple[_FrontierNet, float]:
        """Run a single initialisation: pretrain + finetune.

        Returns
        -------
        tuple[_FrontierNet, float]
            Fitted network and its final NLL.
        """
        torch.manual_seed(seed)
        net = self._build_net(X_t.shape[1])

        self._pretrain(net, X_t, y_t)
        self._init_sigmas(net, X_t, y_t)
        nll = self._finetune(net, X_t, y_t)

        return net, nll

    # -- public interface ---------------------------------------------------

    def fit(self, X: FloatArray, y: FloatArray) -> Self:
        """Fit the joint NN-SFA model.

        Runs ``n_inits`` random initialisations, each with MSE pre-training
        followed by SFA NLL fine-tuning.  The result with the highest
        log-likelihood is retained.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        Self
            The fitted estimator.
        """
        X_val, y_val = self._validate_data(X, y)

        X_t = torch.tensor(X_val, dtype=torch.float64)
        y_t = torch.tensor(y_val, dtype=torch.float64)

        if self.seed is not None:
            base_seed = self.seed
        else:
            import random

            base_seed = random.randint(0, 2**31)

        best_net: _FrontierNet | None = None
        best_nll = float("inf")

        for i in range(self.n_inits):
            net, nll = self._fit_single(X_t, y_t, seed=base_seed + i)
            if nll < best_nll:
                best_nll = nll
                best_net = net

        if best_net is None:
            msg = "No successful initialisation; check n_inits > 0."
            raise RuntimeError(msg)

        # Store fitted attributes
        self._net = best_net
        self.sigma_v_ = float(torch.exp(best_net.log_sigma_v).item())
        self.sigma_u_ = float(torch.exp(best_net.log_sigma_u).item())
        self.log_likelihood_ = -best_nll
        self.n_features_in_ = X_val.shape[1]
        self.is_fitted_ = True
        self._n_obs = X_val.shape[0]

        # Precompute mean efficiency
        te = self.efficiency(X_val, y_val)
        self._mean_efficiency = float(np.mean(te))

        return self

    def predict(self, X: FloatArray) -> FloatArray:
        """Predict frontier values.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        FloatArray
            Predicted frontier values of shape ``(n_samples,)``.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        X_t = torch.tensor(X_arr, dtype=torch.float64)
        with torch.no_grad():
            pred = self._net(X_t).numpy()
        return pred  # type: ignore[no-any-return]

    def efficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate technical efficiency via JLMS.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        FloatArray
            Technical efficiency in ``(0, 1]`` of shape ``(n_samples,)``.
        """
        u_hat = self.get_inefficiency(X, y)
        return np.exp(-u_hat)

    def get_inefficiency(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate inefficiency u via JLMS conditional mean.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        FloatArray
            Non-negative inefficiency estimates of shape ``(n_samples,)``.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        pred = self.predict(X_arr)
        epsilon = y_arr - pred

        dist = self._get_distribution()
        if self.cost:
            return dist.conditional_mean(-epsilon, self.sigma_v_, self.sigma_u_)
        return dist.conditional_mean(epsilon, self.sigma_v_, self.sigma_u_)

    def get_noise(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate noise component v.

        Parameters
        ----------
        X : FloatArray
            Input matrix of shape ``(n_samples, n_features)``.
        y : FloatArray
            Observed output of shape ``(n_samples,)``.

        Returns
        -------
        FloatArray
            Estimated noise values of shape ``(n_samples,)``.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        pred = self.predict(X_arr)
        epsilon = y_arr - pred
        u_hat = self.get_inefficiency(X_arr, y_arr)

        if self.cost:
            return epsilon - u_hat
        return epsilon + u_hat

    def log_likelihood(self) -> float:
        """Return the maximised log-likelihood.

        Returns
        -------
        float
            Log-likelihood value.
        """
        self._check_fitted()
        return self.log_likelihood_

    def summary(self) -> SFASummary:
        """Return a summary of the fitted model.

        Returns
        -------
        SFASummary
            Frozen dataclass with model diagnostics.
        """
        self._check_fitted()

        n_params = sum(p.numel() for p in self._net.parameters())
        n_obs = self._n_obs
        ll = self.log_likelihood_

        aic_val = -2.0 * ll + 2.0 * n_params
        bic_val = -2.0 * ll + n_params * float(np.log(n_obs))

        return SFASummary(
            n_obs=n_obs,
            n_params=n_params,
            log_likelihood=ll,
            aic=aic_val,
            bic=bic_val,
            sigma_v=self.sigma_v_,
            sigma_u=self.sigma_u_,
            mean_efficiency=self._mean_efficiency,
            frontier_type=self._FRONTIER_TYPE,
            inefficiency_dist=self.inefficiency,
        )
