.PHONY: prebuild

prebuild:
	uv run mypy --install-types --non-interactive
	uv run mypy
	uv run ruff format --check
	uv run ruff check --fix
	uv run pytest
