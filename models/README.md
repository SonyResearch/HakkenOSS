# Models

Python packages implementing the Hakken knowledge graph platform. All packages are managed with [uv](https://docs.astral.sh/uv/) and live under `packages/`. This directory is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) with a shared lockfile.

## Package Overview

| Package | Description |
|---------|-------------|
| [`complex-query`](packages/complex-query/README.md) | Complex DNF query resolution over knowledge graphs |
| [`simple-query`](packages/simple-query/README.md) | Single-link query resolution |
| [`query-common`](packages/query-common/README.md) | Shared entities and utilities for querying packages |
| [`filtering`](packages/filtering/README.md) | Node and triplet filtering to reduce query candidates |
| [`contextualization`](packages/contextualization/README.md) | Publication retrieval and summarisation for triples |
| [`kge`](packages/kge/README.md) | Knowledge graph embedding training (ComplEx, DistMult, ConvE, GNN) |
| [`kge_api`](packages/kge_api/README.md) | REST API for serving KGE models |
| [`gnn`](packages/gnn/README.md) | Graph Neural Networks (GCN, GAT, GIN, RGCN, GraphSAGE) |
| [`embeddings`](packages/embeddings/README.md) | RDF2Vec-based entity embeddings |
| [`datasets`](packages/datasets/README.md) | Dataset management for knowledge graphs |
| [`data_processing`](packages/data_processing/README.md) | Relations data cleaning and processing pipeline |
| [`data_io`](packages/data_io/README.md) | Data I/O utilities |
| [`data-conversion`](packages/data-conversion/README.md) | Data format conversion tools |
| [`data_api`](packages/data_api/README.md) | FastAPI service for data retrieval |
| [`hakken_api_gateway`](packages/hakken_api_gateway/README.md) | API gateway routing to Hakken microservices |
| [`hakken_utils`](packages/hakken_utils/README.md) | Shared Python utilities (file handling, logging) |
| [`hakken_ml_toolkit`](packages/hakken_ml_toolkit/README.md) | Reusable ML components (metrics, losses, trackers) |
| [`hakken-models`](packages/hakken-models/README.md) | Model training pipelines (ZenML) |
| [`hakken-models-api`](packages/hakken-models-api/README.md) | REST API for serving trained models |
| [`hakken-agents`](packages/hakken-agents/README.md) | Document ingestion (Enki) and entity resolution pipelines |
| [`hypgen_pipeline`](packages/hypgen_pipeline/README.md) | Hypothesis generation post-processing pipeline |
| [`simple_xkgc`](packages/simple_xkgc/README.md) | Cross-KG completion |
| [`simple_xkgc_api`](packages/simple_xkgc_api/README.md) | REST API for cross-KG completion |
| [`spaice_inference_api`](packages/spaice_inference_api/README.md) | Base framework for building inference REST APIs |

## Installation

Install all workspace dependencies from the workspace root:

```bash
uv sync
```

To work on a specific package:

```bash
cd packages/<package-name>
uv sync
```

## Code Quality

Before committing, run the following checks in the relevant package:

```bash
uv run mypy
uv run ruff check .
uv run ruff format .
uv run pytest
```

## Environment Variables

Several packages require credentials or configuration via environment variables. Copy the provided `.env.example` to `.env` in the relevant package directory and fill in your values. Never commit `.env` files.
