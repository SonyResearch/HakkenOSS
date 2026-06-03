# Configuration

The Hakken Models API uses Hydra for configuration management. Configuration files are located in the `config/` directory.

## Configuration File

The main configuration file is `config/config.yaml`. It uses Hydra's YAML format with support for environment variable overrides.

## Configuration Parameters

### Model Settings

```yaml
model: thiger              # Model type (currently only "thiger" supported)
device: cuda               # Device: "cuda" or "cpu"
ckpt_is_lightning: true    # Whether checkpoint is PyTorch Lightning format
```

### Model Loading Modes

You must configure **exactly one** of the following loading modes:

#### Directory Mode

Load models from a directory (local filesystem or S3):

```yaml
run_dir: s3://bucket/path/to/model/
relative_ckpt_path: artifacts/checkpoints/last/last.ckpt
relative_params_path: params.json
```

**Parameters:**
- `run_dir` (required): URI to the model directory (S3 or local path)
- `relative_ckpt_path` (required): Relative path to checkpoint file from `run_dir`
- `relative_params_path` (required): Relative path to parameters JSON file

**Example (S3):**
```yaml
run_dir: s3://<your-bucket>/models/prod/thiger-v1.0.0/
relative_ckpt_path: artifacts/checkpoints/last/last.ckpt
relative_params_path: params.json
```

**Example (Local):**
```yaml
run_dir: /path/to/local/model/
relative_ckpt_path: checkpoints/last.ckpt
relative_params_path: params.json
```

#### MLflow Mode

Load models from MLflow tracking server:

```yaml
mlflow_run_id: 015fbeea19b94fd98c3ee6233773fd30
tracking_uri: s3://hakken-mlflow/mlruns
artifact_path: checkpoints
relative_ckpt_path: last/last.ckpt
```

**Parameters:**
- `mlflow_run_id` (required): MLflow run ID
- `tracking_uri` (required): MLflow tracking server URI (S3, file://, or http://)
- `artifact_path` (required): Path within MLflow artifacts
- `relative_ckpt_path` (required): Relative path to checkpoint from `artifact_path`

**Example:**
```yaml
mlflow_run_id: 015fbeea19b94fd98c3ee6233773fd30
tracking_uri: file:///home/user/mlruns
artifact_path: checkpoints
relative_ckpt_path: last/last.ckpt
```

## Environment Variables

### Configuration Path

Override the configuration directory:

```bash
export CONFIG_PATH=/path/to/config
```

### Hydra Settings

Hydra-specific settings can be configured in the config file:

```yaml
hydra:
  run:
    dir: .                    # Output directory for Hydra runs
  output_subdir: null         # Disable Hydra output subdirectories
```

## Configuration Validation

The configuration is validated using Pydantic models. The following rules are enforced:

1. **Exactly one loading mode**: You must provide either `mlflow_run_id` OR `run_dir`, but not both
2. **Required fields**: All required fields for the selected mode must be provided
3. **Type validation**: All values are type-checked

Invalid configurations will raise a `ValueError` at startup with a descriptive error message.

## Example Configurations

### Production (S3 Directory)

```yaml
defaults:
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

model: thiger
ckpt_is_lightning: true
device: cuda

# Directory mode
run_dir: s3://<your-bucket>/models/prod/thiger-v1.0.0/
relative_ckpt_path: artifacts/checkpoints/last/last.ckpt
relative_params_path: params.json

hydra:
  run:
    dir: .
  output_subdir: null
```

### Development (MLflow)

```yaml
defaults:
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

model: thiger
ckpt_is_lightning: true
device: cuda

# MLflow mode
mlflow_run_id: 015fbeea19b94fd98c3ee6233773fd30
tracking_uri: file:///home/user/mlruns
artifact_path: checkpoints
relative_ckpt_path: last/last.ckpt

hydra:
  run:
    dir: .
  output_subdir: null
```

### Local Testing (CPU)

```yaml
defaults:
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

model: thiger
ckpt_is_lightning: true
device: cpu  # Use CPU for local testing

run_dir: /local/path/to/model/
relative_ckpt_path: checkpoints/last.ckpt
relative_params_path: params.json

hydra:
  run:
    dir: .
  output_subdir: null
```

## Configuration Schema

::: hakken_models_api.config.HakkenModelsAPIConfig

## Troubleshooting

### Configuration Errors

**Error: "Configuration is ambiguous"**
- Solution: Remove one of `mlflow_run_id` or `run_dir` - you can only use one mode

**Error: "You must provide either 'mlflow_run_id' OR 'run_dir'"**
- Solution: Add either `mlflow_run_id` (for MLflow) or `run_dir` (for directory mode)

**Error: "Dataset is not loaded in THiGER artifacts"**
- Solution: Ensure the model directory contains the dataset files and `params.json` is correctly configured

### S3 Access

For S3-based configurations, ensure:
- AWS credentials are configured (via `~/.aws/credentials` or environment variables)
- IAM permissions allow read access to the S3 bucket
- The bucket and path exist

### MLflow Access

For MLflow configurations, ensure:
- MLflow tracking server is accessible
- Run ID exists and contains the required artifacts
- Authentication is configured if required

