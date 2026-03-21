# ML-SFA Project

## Overview

Machine Learning approaches to Stochastic Frontier Analysis (SFA).
Traditional SFA models estimate production/cost frontiers and decompose error terms into noise and inefficiency components. This project extends SFA with ML techniques.

## Project Structure

```
src/ml_sfa/
  models/      # SFA model implementations (traditional + ML-based)
  data/        # Data loading, preprocessing, simulation
  evaluation/  # Metrics, model comparison, diagnostics
  utils/       # Shared utilities
tests/
  unit/        # Unit tests
  integration/ # Integration tests
notebooks/     # Exploratory analysis
configs/       # Configuration files
```

## Development Commands

```bash
# Setup (uv)
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Tests
pytest                          # run all tests
pytest tests/unit/              # unit tests only
pytest --cov-report=html        # coverage report

# Linting & formatting
ruff check src/ tests/          # lint
ruff format src/ tests/         # format

# Type checking
mypy src/
```

## Conventions

- Python 3.11+, type hints required
- src layout (`src/ml_sfa/`)
- Immutable data patterns preferred (return new objects, don't mutate)
- All public functions need docstrings
- Tests follow TDD: write test first, then implement
- Target 80%+ test coverage
