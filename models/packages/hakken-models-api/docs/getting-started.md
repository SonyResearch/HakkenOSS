# Getting Started

This guide will help you set up and run the Hakken Models API service.

## Installation

### Prerequisites

- Python 3.11 or 3.12
- UV package manager (recommended) or pip
- Access to the Hakken internal package registry (for `spaice-inference-api`)

### Install Dependencies

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -e .
```

The package will install all required dependencies including:
- `hakken_models` - Core THiGER model implementations
- `spaice-inference-api` - Inference API framework
- `fastapi` - Web framework
- `pydantic` - Data validation
- `dependency-injector` - Dependency injection
- `hydra-core` - Configuration management

## Configuration

The service uses Hydra for configuration management. Configuration files are located in `config/`.

### Basic Configuration

Edit `config/config.yaml` to configure your model loading:

```yaml
model: thiger
device: cuda  # or "cpu"
ckpt_is_lightning: true

# Option 1: Load from directory (S3 or local)
run_dir: s3://<your-bucket>/models/prod/thiger-v1.0.0/
relative_ckpt_path: artifacts/checkpoints/last/last.ckpt
relative_params_path: params.json

# Option 2: Load from MLflow
# mlflow_run_id: 015fbeea19b94fd98c3ee6233773fd30
# artifact_path: checkpoints
# tracking_uri: s3://hakken-mlflow/mlruns
# relative_ckpt_path: last/last.ckpt
```

### Configuration Modes

You must choose **one** of the following loading modes:

#### Directory Mode

Load models from a directory (local or S3):

```yaml
run_dir: s3://bucket/path/to/model/
relative_ckpt_path: artifacts/checkpoints/last/last.ckpt
relative_params_path: params.json
```

#### MLflow Mode

Load models from MLflow tracking server:

```yaml
mlflow_run_id: <run-id>
tracking_uri: s3://hakken-mlflow/mlruns
artifact_path: checkpoints
relative_ckpt_path: last/last.ckpt
```

### Environment Variables

You can override the config path:

```bash
export CONFIG_PATH=/path/to/config
```

## Running the Service

### Development Mode

```bash
# Using Make
make serve

# Or directly
uv run python service.py
```

The service will start on the default port (typically `8088`). Check the console output for the exact URL.

### Docker

Build and run using Docker:

```bash
docker build -t hakken-models-api .
docker run -p 8088:8088 hakken-models-api
```

## Testing the Service

### Health Check

Once the service is running, verify it's working:

```bash
curl http://localhost:8088/health
```

### Test Prediction

Use the Makefile target:

```bash
make predict
```

Or manually:

```bash
curl -X POST http://localhost:8088/thiger/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id_list": ["Chemical|MESH:C494910"],
    "object_id_list": ["Gene|3553"],
    "relation_id_list": ["negative_correlate"]
  }'
```

## Development

### Running Checks

```bash
# Format, lint, type check, and test
make checks
```

This runs:
- `ruff format` - Code formatting
- `ruff check` - Linting
- `mypy` - Type checking
- `pytest` - Unit tests

### Project Structure

```
hakken-models-api/
├── config/              # Hydra configuration files
├── hakken_models_api/   # Main package
│   ├── config.py        # Configuration models
│   ├── container.py     # Dependency injection container
│   ├── loaders/         # Model loaders (thiger, segal, kge)
│   ├── routers/         # FastAPI routes (thiger, segal)
│   └── entities/        # Request/response models
├── tests/               # Test suite
├── service.py           # Service entry point
└── docs/                # Documentation
```

## Next Steps

- Read the [API Reference](api/endpoints.md) to understand available endpoints
- Check [Configuration](configuration.md) for advanced configuration options
- Review [Architecture](architecture.md) to understand the system design

