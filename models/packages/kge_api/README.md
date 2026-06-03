# KGE API Service

A FastAPI-based service for Knowledge Graph Embedding (KGE) model inference.  
It allows you to load trained KGE models and perform inference operations through a RESTful API.

---

## Features

- Load trained KGE models from experiment folders  
- Predict missing entities or relations in triples  
- Score knowledge graph triples  
- Fit and load a score scaler for calibrated outputs  
- Inspect device placement of model parameters  

---

## Repository Structure

```
.
├── Makefile              # Run checks (linting, tests, mypy)
├── README.md
├── config/
│   └── config.yaml       # Hydra configuration
├── init.sh               # Environment setup
├── kge_api/              # Core API implementation
│   ├── config.py         # Pydantic APIConfig (validated paths)
│   ├── container.py      # Dependency injection container
│   ├── exceptions.py     # Custom errors
│   ├── kge_loader.py     # Loader for KGE experiments
│   ├── router.py         # FastAPI endpoints
│   └── __init__.py
├── misc/                 # Example notebooks
├── pyproject.toml        # Project dependencies
├── service.py            # Main service entry point
├── tests/                # Unit tests
└── uv.lock               # Dependency lock file
```

---

## Installation

Initialize the project:

```bash
./init.sh
```

This creates a `.env` file with default environment variables.
Update them if needed, then load them:

```bash
source .env
```

Install dependencies:

```bash
uv sync --index-strategy unsafe-best-match

# or alternatively
set -a && source .env && set +a && uv sync --index-strategy unsafe-best-match
```

---

## Configuration

The service uses **Hydra** + **Pydantic** (`APIConfig`) for configuration.
Main config: `config/config.yaml`.

### Environment Variables

The following variables are set up by `init.sh` in your `.env`:

| Variable                                | Description                                                               |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `DATA_BASE_DIR`                         | Base directory for raw data files                                         |
| `DATA_ROOT_FOLDER`                      | Dataset-specific root folder (derived from `DATA_BASE_DIR`)               |
| `KGE_RUN_PATH`                          | Path to the KGE experiment output (models, checkpoints, logs)             |
| `CACHED_DATA_FOLDER`                    | Path to cached knowledge graph data (usually `${KGE_RUN_PATH}/cached_kg`) |
| `CONFIG_PATH`                           | Path to Hydra config directory (default: `${PWD}/config`)                 |
| `UV_INDEX_HAKKEN_PIP_REGISTRY_USERNAME` | Username for the private package registry                                 |
| `UV_INDEX_HAKKEN_PIP_REGISTRY_PASSWORD` | Password for the private package registry                                 |

> ⚠️ `EXPERIMENT_FOLDER` in `APIConfig` is typically set to `KGE_RUN_PATH` in your `.env`.
> Make sure this folder exists before running the service.

---

## Running the Service
First download a trined KGE model:

```bash
aws s3 sync s3://<your-bucket>/models/prod/kge-v1.0.0/ ./hakken_models/kge-v1.0.0/
```

And make sure all the environmental variables are correctly set. Then run the service:

```bash
uv run python service.py
```

The service runs on **port 8088** by default (or the value of `SPAICE_APPLICATION_PORT` if set).

---

## API Endpoints

### `POST /kge/predict`

Predict missing entities or relations.

Request:

```json
{
  "subject_id_list": ["s1", "s2"],
  "object_id_list": ["o1", "o2"],
  "relation_id_list": ["r1"],
  "inference_config": {}
}
```

Response:

```json
{
  "relations_ids": ["r1"],
  "relations_probs": [[0.8], [0.2]],
  "relations_scores": [[123.23], [-50.3]]
}
```

---

### `POST /kge/score`

Score triples.

Request:

```json
{ "triple_index_list": [[1, 2, 3], [4, 5, 6]] }
```

Response:

```json
{ "scores_list": [0.85, 0.72] }
```

---

### `POST /kge/fit_score_scaler`

Fit and save a score scaler.

Request:

```json
{
  "overwrite": true,
  "loader_kwargs": {}
}
```

Response:

```json
{ "success": true, "message": "Score scaler fitted" }
```

---

### `POST /kge/device`

Check device placement of model parameters.

Response:

```json
["cpu"]
```

---

## Examples

* Interactive notebook: `misc/demo.ipynb`

Quick test with `curl`:

```bash
curl -X POST "http://localhost:8088/kge/predict" \
  -H "Content-Type: application/json" \
  -d '{"subject_id_list":["s1"], "object_id_list":["o1"], "relation_id_list":["r1"], "inference_config":{}}'
```

---

## Development

Run checks and tests:

```bash
make run_checks
```

This runs linting (`ruff`), formatting, tests (`pytest`), and type checks (`mypy`).


--- 

## Model Registry

| KGE Version | Dataset   | Dataset Version | KGE API Version |
|:------------|:----------|---------------:|---------------:|
| 1.0.0       | Pubtator3 | 0.3.0          | 0.1.7          |
| 1.1.0       | Pubtator3 | 0.4.0          | 0.1.8          |
| 1.2.0       | DigitalScience | 2.0.0          | 0.1.8          |

## Secrets and env vars

This repository does not contain real credentials. An example env file is provided at `.env-example` — copy it to `.env` and fill values for local development. Do not commit your `.env` file; it is included in `.gitignore`.

- Use environment variables for credentials (e.g. registry tokens, DB passwords).
- During image builds, use Docker build secret mounts (`--secret`) — no secrets are baked into image layers.
- Before publishing, run a git-history secret scan (e.g. `trufflehog` or `ggshield`) and rotate any exposed credentials.
