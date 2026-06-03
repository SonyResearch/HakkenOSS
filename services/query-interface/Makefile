.PHONY: prebuild

prebuild:
	uv run mypy --install-types --non-interactive
	uv run mypy
	uv run flake8
	uv run pytest