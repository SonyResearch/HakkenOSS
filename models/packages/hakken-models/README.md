# Hakken Models

A Python package for training and serving hypothesis generation models using PyTorch Lightning and ZenML pipelines.

## Overview

Hakken Models provides a complete framework for building, training, and deploying knowledge graph embedding models. It includes dataset preparation pipelines, model training with hyperparameter optimization, and support for multiple KGE architectures including ComplEx, DistMult, and RotatE.

## Key Features

- **Pipeline-Based Workflow**: ZenML-powered pipelines for dataset preparation and model training
- **Multiple KGE Models**: Support for ComplEx, DistMult, and RotatE scoring functions
- **Flexible Negative Sampling**: Uniform, Bernoulli, and self-adversarial sampling strategies
- **PyTorch Lightning Integration**: Efficient training with automatic mixed precision and GPU support
- **S3 Integration**: Seamless data loading and artifact storage
- **Data Quality Reports**: Automated quality checks using Evidently AI

## Quick Start

### Installation

```bash
# Install dependencies
set -a && source .env && set +a && uv sync

# uv pip install torch-sparse --no-binary torch-sparse --no-build-isolation
# uv pip install torch-scatter --no-binary torch-scatter --no-build-isolation
# uv pip install torch-cluster --no-binary torch-cluster --no-build-isolation
# uv pip install torch-spline-conv --no-binary torch-spline-conv --no-build-isolation
uv pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-2.7.0+cu118.html
```

### Basic Usage

```bash
# Prepare a dataset
make datasets-pipeline

# Or using the CLI
uv run python scripts/run_pipeline.py prepare-dataset \
    --dataset-name pubtator3-v0.4.0 \
    --dataset-version v2

# Train a model
make train-pipeline

# Or using the CLI
uv run python scripts/run_pipeline.py train-model \
    --dataset-name pubtator3-v0.4.0 \
    --dataset-version v2 \
    --model-name complex


# Hyperparameter tuning
uv run --active python scripts/run_pipeline.py hpo-kge --override "hpo=default" --config-dir <path-to-configs>
```

## Documentation

For comprehensive documentation, see the [full documentation site](docs/index.md) or build it locally:

```bash
# Build and serve documentation
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

## Project Structure

```
hakken_models/
├── models/          # KGE model implementations
├── scores/          # Scoring functions (ComplEx, DistMult, RotatE)
├── pipelines/       # ZenML pipeline definitions
├── steps/           # Pipeline steps (dataset, training)
├── configs/         # Configuration classes
├── negative_samplers/ # Negative sampling strategies
└── fact_validator/   # Fact validation utilities
```

## Requirements

- Python 3.11-3.12
- PyTorch 2.4+ (with CUDA support)
- ZenML 0.91.2+

