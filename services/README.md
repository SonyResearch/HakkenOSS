# Services

Docker Compose-based deployment services for the Hakken platform.

## Services

| Service | Description |
|---------|-------------|
| [`data-loader`](data-loader/README.md) | Preprocesses raw knowledge graph data and loads it into Neo4j |
| [`query-interface`](query-interface/README.md) | REST API exposing the complex and simple querying modules |
| [`sentences_db`](sentences_db/README.md) | PostgreSQL database for storing biomedical publication sentences |

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/) and Docker Compose
- AWS CLI configured with S3 read access (for data download)

## Environment Variables

Each service reads credentials from a `.env` file. Never commit `.env` files.

| Service | Required variables |
|---------|--------------------|
| `query-interface` | `OCTOPUS_HOST`, `OCTOPUS_PORT`, `OCTOPUS_USERNAME`, `OCTOPUS_PASSWORD`, `CORE_MODEL_HOST`, `CORE_MODEL_PORT` |
| `sentences_db` | `POSTGRES_PASSWORD`, `PGPORT`, `PGDATA` |
| `data-loader` | AWS credentials for S3 access |

## Running a Service

```bash
cd <service-name>
docker compose up -d
```

See each service's README for detailed setup and configuration steps.
