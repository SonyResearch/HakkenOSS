SHELL := /bin/bash

.PHONY: run_checks serve

run_checks:
	uv run ruff format .
	uv run ruff check --fix .
	uv run mypy .
	uv run pytest

serve:
	NX_CUGRAPH_AUTOCONFIG=True uv run python service.py

explainability_ui:
	uv run streamlit run scripts/app.py
