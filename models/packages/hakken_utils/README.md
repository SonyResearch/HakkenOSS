# Hakken Utils

This package is meant to become the reference utils for simplifying handling different operations in python 
like files handling and logging.

# Installation

To install this package please run
```bash
uv sync --index-strategy unsafe-best-match
```

Before committing, please run the following and fix any issue:
```bash
uv run mypy
uv run black .
uv run isort .
uv run flake8
```

And check that tests are not failing
```bash
uv run pytest 
```