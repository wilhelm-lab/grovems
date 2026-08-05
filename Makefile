# ── Developer / CI targets ──────────────────────────────────────────────────
# Use `make install` once to set up, then any target below can be run locally.
#
# Usage:
#   make install       – install project + all dev deps into Poetry virtualenv
#   make lint          – run pre-commit hooks (formatting, linting, security checks)
#   make format        – format code with ruff
#   make test          – run unit tests
#   make test-int      – run integration tests
#   make coverage-unit – run unit tests with coverage and export XML
#   make coverage-int  – run integration tests with coverage and export XML
#   make coverage      – run unit + integration tests with coverage and export XML
#   make typecheck     – runtime type checking via typeguard
#   make doctest       – validate inline docstring examples
#   make docs          – build HTML documentation with Sphinx
#   make docs-serve    – build docs and serve locally with live reload
#   make check         – run all quality checks in CI order (lint → test → coverage → typecheck → doctest)
#   make dist          – build source and wheel distributions
# ────────────────────────────────────────────────────────────────────────────
.PHONY: install lint format test test-int coverage coverage-unit coverage-int typecheck doctest docs docs-serve dist check

check: lint test coverage typecheck doctest ## Run all quality checks in CI order

install: ## Install project with all dev and docs dependencies
	poetry install --with dev --extras docs

lint: ## Run pre-commit hooks (formatting, linting, security checks)
	poetry run pre-commit run --all-files

format: ## Format code with ruff
	poetry run ruff format src tests

test: ## Run unit tests
	poetry run pytest tests/unit_tests

test-int: ## Run integration tests
	poetry run pytest tests/integration_tests

coverage-unit: ## Run unit tests with coverage and export XML
	poetry run coverage erase
	poetry run coverage run -m pytest tests/unit_tests
	poetry run coverage report -i
	poetry run coverage xml

coverage-int: ## Run integration tests with coverage and export XML
	poetry run coverage erase
	poetry run coverage run -m pytest tests/integration_tests
	poetry run coverage report -i
	poetry run coverage xml

coverage: ## Run unit + integration tests with coverage and export combined XML
	poetry run coverage erase
	poetry run coverage run -m pytest tests/unit_tests
	poetry run coverage run --append -m pytest tests/integration_tests
	poetry run coverage report -i
	poetry run coverage xml

typecheck: ## Runtime type checking with typeguard
	poetry run pytest --typeguard-packages=grovems tests/unit_tests

doctest: ## Validate inline docstring examples with xdoctest
	poetry run python -m xdoctest grovems all

docs: ## Build HTML documentation with Sphinx
	poetry run sphinx-build -b html docs docs/_build

docs-serve: ## build docs and serve locally with live reload
	poetry run sphinx-autobuild docs docs/_build/html --open-browser

dist: ## Build source and wheel distributions
	poetry build --ansi
