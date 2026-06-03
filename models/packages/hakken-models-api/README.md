# Hakken Models API

A FastAPI service for serving models for knowledge graph completion and inference.

## Overview

Hakken Models API provides a RESTful interface to predictive models, enabling prediction, scoring, and fact sampling operations on knowledge graphs. The service is built on the `spaice-inference-api` framework and supports loading models from MLflow or local directories.

## Quick Start

### Installation

```bash
# Install dependencies
set -a && source .env && set +a && uv sync
```

### Configuration

Configure the service via `config/config.yaml` or environment variables. You must specify either:

- **MLflow mode**: `mlflow_run_id` with `tracking_uri` and `artifact_path`
- **Directory mode**: `run_dir` with `relative_ckpt_path` and `relative_params_path`

Example configuration:
```yaml
model: thiger
device: cuda
ckpt_is_lightning: true
run_dir: s3://<your-bucket>/models/prod/thiger-v1.0.0/
relative_ckpt_path: artifacts/checkpoints/last/last.ckpt
relative_params_path: params.json
```

### Running the Service

```bash
# Start the API server
make serve

# Or directly
uv run python service.py
```

The service will start on the default port (typically `8088`).

### Testing

```bash
# Run all checks (format, lint, type check, tests)
make checks

# Test prediction endpoint
make predict
```

## API Endpoints

- `POST /thiger/predict` - Predict relations for entity pairs
- `POST /thiger/score` - Score facts (triples)
- `POST /thiger/entity-pair-indexes` - Map entity IDs to indexes
- `POST /thiger/fact-indexes` - Map fact IDs to indexes
- `POST /thiger/sample-facts` - Sample random facts from dataset splits

## Documentation

For comprehensive documentation including API reference, configuration details, and examples:

```bash
# Serve documentation locally
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

## Requirements

- Python 3.11-3.12
- PyTorch 2.4+ (with CUDA support)
- `hakken_models` package
- `spaice-inference-api` framework

## Secrets and env vars

This repository does not contain real credentials. An example env file is provided at `.env-example` — copy it to `.env` and fill values for local development. Do not commit your `.env` file; it is included in `.gitignore`.

- Use environment variables for credentials (e.g. registry tokens, DB passwords).
- During image builds, use Docker build secret mounts (`--secret`) — no secrets are baked into image layers.
- Before publishing, run a git-history secret scan (e.g. `trufflehog` or `ggshield`) and rotate any exposed credentials.
