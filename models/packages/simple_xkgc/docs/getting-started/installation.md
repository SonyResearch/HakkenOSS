# Installation

## Requirements

- Python >= 3.11, < 3.13
- CUDA-capable GPU (recommended for performance)
- `uv` package manager

## Install Dependencies

```bash
uv sync
```

This will install all required dependencies including:
- PyTorch
- NetworkX
- hakken_ml_toolkit
- kge
- And other dependencies

## Install Documentation Dependencies (Optional)

To build and serve the documentation locally:

```bash
uv sync --extra docs
```

## Verify Installation

Run the test suite to verify everything is working:

```bash
make run_checks
```

This will run:
- Code formatting checks (ruff)
- Type checking (mypy)
- Unit tests (pytest)

## Environment Setup

The package uses environment variables for configuration. Key variables include:

- `DATA_ROOT_FOLDER`: Base directory for datasets
- `MODEL_FOLDER`: Path to trained KGE model
- `GRAPH_CACHE_FOLDER`: Directory for caching graphs (optional)
- `CONFIG_PATH`: Path to configuration directory (defaults to `config/`)

Set these in your environment or `.env` file before running the explainer.

