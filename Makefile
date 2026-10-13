PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
PROJECT_PYTHON := $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(PYTHON))
K ?= 3

.PHONY: setup install run eval format lint test typecheck build check

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"

install:
	$(PROJECT_PYTHON) -m pip install -e ".[dev]"

run:
	PYTHONPATH=src $(PROJECT_PYTHON) -m refund_agent.main

eval:
	PYTHONPATH=src $(PROJECT_PYTHON) eval/run_eval.py --k $(K)

format:
	$(PROJECT_PYTHON) -m ruff format .
	$(PROJECT_PYTHON) -m ruff check --fix .

lint:
	$(PROJECT_PYTHON) -m ruff format --check .
	$(PROJECT_PYTHON) -m ruff check .

test:
	$(PROJECT_PYTHON) -m pytest

typecheck:
	$(PROJECT_PYTHON) -m mypy

build:
	$(PROJECT_PYTHON) -m build

check: lint typecheck test build
