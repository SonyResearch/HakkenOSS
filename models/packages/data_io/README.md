# Data IO

Utilities for reading and writing data from various sources and formats.

## Installation
To install this package please run
```bash
uv sync --index-strategy unsafe-best-match
```

## Testing
Before committing please run:
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