# Contributing

Contributions to Hakken Explainer are welcome! This document provides guidelines for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork
3. Install development dependencies:

```bash
uv sync
```

4. Run checks to ensure everything works:

```bash
make run_checks
```

## Code Style

The project uses:
- **ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for testing

Run checks before committing:

```bash
make run_checks
```

## Testing

Write tests for new features:

```bash
uv run pytest tests/
```

## Documentation

When adding new features:

1. Update relevant documentation in `docs/`
2. Add docstrings following Google style
3. Update examples if applicable
4. Build docs locally to verify:

```bash
make docs-serve
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all checks pass: `make run_checks`
4. Update documentation
5. Submit a pull request with a clear description

## Code of Conduct

Be respectful and constructive in all interactions.

