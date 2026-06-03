# Simple XKGC API

A FastAPI service + Streamlit demo UI for generating explanations of predictions of knowledge graph completion methods.


## Overview of the project

This package allows probing triples (subject-relation-object) and finding sets of training triples to explain predictions.

The Simple XKGC API Service provides endpoints for:

- Generate explanations for knowledge graph triples
- Compute the length of the explanation
  
## Repository Structure

```
.
|-- Makefile
|-- README.md
|-- config/
|-- misc/
|   |-- *.py
|   `-- demo.ipynb
|-- pyproject.toml
|-- scripts/
|   `-- app.py          # Streamlit App
|-- service.py          # FastAPI server (Hydra entrypoint)
|-- simple_xkgc_api/
|   |-- container.py
|   |-- entities/
|   |-- path_explainer_loader.py
|   `-- router.py       # /path_explainer endpoints
|-- tests/
|   |-- __init__.py
|   `-- test_loader.py
`-- uv.lock
```


## Installation

Initialize the project:

```bash
cp .env.example .env
```

Update the variables in this file. You have to load your env vars by `source .env`.

Install dependencies:

```bash
uv sync --index-strategy unsafe-best-match

# or alternatively
set -a && source .env && set +a && uv sync --index-strategy unsafe-best-match
```


## Configuration

The service is configured using Hydra and YAML files. The main configuration file is located at `config/config.yaml`.



## Usage

### Environment Variables

Before using the API service, you need to set up the following environment variables:

* `DATA_VERSION`: Version tag for the dataset (used to resolve folder paths).

* `GNN_VERSION`: Version tag for the GNNKGE model (used to resolve model and cache paths).

* `DATA_ROOT_FOLDER`: Base directory containing all dataset files including cached data and graph caches.

* `CACHED_DATA_FOLDER`: Directory containing cached/preprocessed dataset files for the specified `GNN_VERSION`.

* `GRAPH_CACHE_FOLDER`: Directory containing precomputed graph cache files for the specified `GNN_VERSION`.

* `MODEL_DIR`: Path to the directory containing the trained GNNKGE model checkpoints.

* `CONFIG_PATH`: Path to the API configuration files (Hydra configs for path finder and explainer).

* `EXPLAIN_API_ENDPOINT`: HTTP endpoint where the Explainability API is served.

* `UV_INDEX_HAKKEN_PIP_REGISTRY_USERNAME`: Username for authenticating to the internal Hakken Python package registry.

* `UV_INDEX_HAKKEN_PIP_REGISTRY_PASSWORD`: Password for authenticating to the internal Hakken Python package registry.

You can set these variables directly in your shell or in the `.env` file (created during initialization).


### Starting the Service

Run the service using:

```bash

make serve
```

### Example Usage

You can interact with the API using curl, Postman, or any HTTP client. For example:

```bash
curl -X POST "http://localhost:8089/test/path_explainer/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "triples_to_probe": [["1000000429", "TREAT", "1000038363"]],
    "num_explanations": 5
  }'
```

Additionally, you can find a Jupyter notebook with usage examples in `misc/demo.ipynb`.

## API Endpoints

To understand the available endpoints and their usage, you can access the automatically generated API documentation at `http://localhost:8089/docs` once the service is running.

## Streamlit App (Development)

This package also includes a streamlit application that provides a visual interface to interact with th  API.

### Features

- Simple user interface built with Streamlit
- Input form for specifying subject, relation, and object identifiers
- Configurable parameters for batch size and number of explanations
- Interactive results
- Raw data view with complete explanation information
- Export functionality to download results as CSV

### Usage

Start the local API service (it must be running at `EXPLAIN_API_ENDPOINT`).

Launch the Streamlit app:
```bash
make explainability_ui
```

## Secrets and env vars

This repository does not contain real credentials. An example env file is provided at `.env-example` — copy it to `.env` and fill values for local development. Do not commit your `.env` file; it is included in `.gitignore`.

- Use environment variables for credentials (e.g. registry tokens, DB passwords).
- During image builds, use Docker build secret mounts (`--secret`) — no secrets are baked into image layers.
- Before publishing, run a git-history secret scan (e.g. `trufflehog` or `ggshield`) and rotate any exposed credentials.
