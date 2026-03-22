# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-22

### Added

#### Phase 1: Parametric SFA
- `BaseSFAEstimator` abstract base class with sklearn-compatible API
- `ParametricSFA` estimator with MLE (Cobb-Douglas / Translog frontiers)
- Inefficiency distributions: HalfNormal, TruncatedNormal, Exponential with JLMS formulas
- SFA data simulator (`simulate_sfa`) with configurable DGPs
- Evaluation metrics: `rmse_efficiency`, `rank_correlation`, `aic`, `bic`, `frontier_mse`, `coverage_rate`

#### Phase 2: Neural Network SFA
- `NNFrontierSFA` estimator with joint NN frontier + SFA error decomposition (PyTorch)
- Two-phase optimization: MSE pretraining → SFA NLL fine-tuning (L-BFGS)
- `MonotonicMLP` for architectural monotonicity constraints
- PyTorch SFA loss functions for all 3 distributions

#### Phase 3: BART SFA
- `BARTFrontierSFA` estimator with PyMC-BART and data-augmented MCMC
- Bayesian TE credible intervals via `credible_interval()`
- Support for half-normal, exponential, truncated-normal inefficiency

#### Phase 4: Kernel SFA + Model Comparison
- `KernelSFA` estimator with local polynomial MLE (Gaussian kernel)
- Nonlinear DGP (`frontier_type="nonlinear"`) for ML method validation
- Model comparison framework: `compare_models()`, `run_benchmark()`
- Interactive showcase notebook (`notebooks/model_showcase.ipynb`)

#### Infrastructure
- Project structure with src layout and scikit-learn compatible API design
- Research survey: ML approaches to Stochastic Frontier Analysis
- Model selection analysis (9 models × 7 challenges)
- Detailed design documents for Joint NN-SFA and BART-SFM
- CI/CD pipeline (ci.yml, release.yml, auto-release.yml)
- PyPI publishing via OIDC trusted publisher
- 220 tests, 93% coverage, mypy strict, ruff clean
