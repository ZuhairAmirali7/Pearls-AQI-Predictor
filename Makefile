# Pearls AQI Predictor — developer commands.
# Uses the local virtualenv at .venv if present, else the ambient python.

VENV ?= .venv
PY := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP := $(PY) -m pip

CITY ?= Karachi
START ?= 2025-01-01
END ?= 2025-12-31

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: venv
venv: ## Create a Python 3.11 virtualenv in .venv
	python3.11 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip wheel setuptools

.PHONY: install
install: ## Install the package (editable) with dev extras
	$(PIP) install -e ".[dev]"

.PHONY: install-all
install-all: ## Install with all optional extras (TensorFlow, SHAP, dev)
	$(PIP) install -e ".[all]"

.PHONY: seed
seed: ## Generate deterministic sample data into the local feature store
	$(PY) scripts/seed_sample_data.py --city $(CITY)

.PHONY: test
test: ## Run the test suite
	$(PY) -m pytest

.PHONY: cov
cov: ## Run tests with coverage report
	$(PY) -m pytest --cov=pearls_aqi --cov-report=term-missing

.PHONY: lint
lint: ## Ruff lint check
	$(PY) -m ruff check src tests scripts api app

.PHONY: format
format: ## Apply black + ruff --fix
	$(PY) -m black src tests scripts api app
	$(PY) -m ruff check --fix src tests scripts api app

.PHONY: format-check
format-check: ## Check formatting without modifying files
	$(PY) -m black --check src tests scripts api app
	$(PY) -m ruff check src tests scripts api app

.PHONY: typecheck
typecheck: ## Run mypy
	$(PY) -m mypy src

.PHONY: backfill
backfill: ## Backfill historical data (CITY, START, END overridable)
	$(PY) scripts/backfill_historical_data.py --city $(CITY) --start-date $(START) --end-date $(END)

.PHONY: features
features: ## Run the hourly feature pipeline once
	$(PY) scripts/run_feature_pipeline.py --city $(CITY)

.PHONY: train
train: ## Run the training pipeline
	$(PY) scripts/run_training_pipeline.py --city $(CITY)

.PHONY: forecast
forecast: ## Generate a 72-hour forecast
	$(PY) scripts/generate_forecast.py --city $(CITY)

.PHONY: api
api: ## Run the FastAPI backend (http://localhost:8000)
	$(PY) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: dashboard
dashboard: ## Run the Streamlit dashboard (http://localhost:8501)
	$(PY) -m streamlit run app/streamlit_app.py

.PHONY: check
check: format-check typecheck test ## Run all quality gates

.PHONY: docker-build
docker-build: ## Build the Docker image
	docker build -t pearls-aqi-predictor:latest .

.PHONY: docker-up
docker-up: ## Start API + dashboard via docker-compose
	docker compose up --build

.PHONY: clean
clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
