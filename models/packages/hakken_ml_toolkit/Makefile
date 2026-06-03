.PHONY: run_checks

checks:
	uv run ruff format
	uv run ruff check . --fix
	uv run pytest
	uv run mypy

