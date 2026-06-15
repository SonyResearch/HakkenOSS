# HakkenOSS

An open-source knowledge graph platform for scientific hypothesis generation and intelligent querying. It provides a full pipeline from raw biomedical data ingestion through knowledge graph construction, embedding-based reasoning, contextualisation of results, and an interactive web interface.

## Repository Structure

```
HakkenOSS/
├── models/      # Python ML/AI packages (KGE, querying, embeddings, data pipelines)
├── services/    # Docker Compose deployment services (Neo4j loader, query API, PostgreSQL)
└── ui/          # React/TypeScript frontend and FastAPI gateway
```

## Key Capabilities

- **Knowledge Graph Embeddings (KGE)** — Train ComplEx, DistMult, RotatE, and GNN-based models on biomedical knowledge graphs.
- **Intelligent Querying** — Simple (single-link) and complex (DNF) query resolution over a Neo4j knowledge graph.
- **Contextualisation** — Retrieve and summarise relevant publications for a given entity triple using vector search.
- **Hypothesis Generation** — Score and rank candidate entity pairs as potential novel scientific findings.
- **Data Pipeline** — Ingest and process large biomedical datasets (Digital Science, PubTator) into Neo4j.
- **Agents** — Document ingestion (Enki) and entity resolution pipelines backed by PostgreSQL + pgvector.
- **Web Interface** — React UI for interactive querying and exploration of results.

## Getting Started

Each component has its own setup guide. Common entry points:

| Goal | Starting point |
|------|----------------|
| Load the knowledge graph | [services/data-loader](services/data-loader/README.md) |
| Run the query service | [services/query-interface](services/query-interface/README.md) |
| Run the frontend | [ui/README.md](ui/README.md) |
| Train a KGE model | [models/packages/kge](models/packages/kge/README.md) |
| Run document ingestion | [models/packages/hakken-agents](models/packages/hakken-agents/README.md) |

## Prerequisites

- **Python ≥ 3.10** with [uv](https://docs.astral.sh/uv/) — for all model packages.
- **Docker and Docker Compose** — for all services.
- **Node.js ≥ 18** — for the frontend.
- **AWS CLI** configured with appropriate credentials — for S3 data access.

## Security

- **Never commit `.env` files.** Copy `.env.example` to `.env` and populate locally.
- All credentials are injected at runtime via environment variables.
- Docker image builds use secret mounts (`--secret`) — no credentials are baked into layers.
- Before pushing, run a secret scan: `trufflehog git file://.` or `ggshield secret scan repo .`
- See individual package READMEs for service-specific environment variable requirements.

## License

| Asset | License |
|-------|---------|
| **Source code** | [BSD 3-Clause](LICENSE) |
| **Model parameters** | [CC BY 4.0 or CDLA Permissive 2.0](LICENSE-MODEL-PARAMETERS.md) — pending Sony management confirmation |
| **Redistributed open-source components** | Each component retains its original license — see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) |
