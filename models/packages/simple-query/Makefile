.PHONY: prebuild

prebuild:
	uv run mypy
	uv run ruff format --check
	uv run ruff check --fix
	uv run pytest
