PACKAGE := ml_sfa

.PHONY: install test lint format format-check typecheck ci build clean

install:
	uv sync --frozen --dev

test:
	uv run pytest --cov=$(PACKAGE) --cov-fail-under=80 -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src/$(PACKAGE)/

ci: lint format-check typecheck test

build:
	uv build
	uv run twine check dist/*

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov/
