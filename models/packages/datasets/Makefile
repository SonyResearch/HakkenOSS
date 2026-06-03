.PHONY: run_checks

run_checks:
	uv run ruff check . --fix
	uv run ruff format
	uv run pytest
	uv run mypy

