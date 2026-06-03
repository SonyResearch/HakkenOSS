# Contextualization Package

This package contains code related to the contextualization component.

Please see also the project documentation for details on algorithms.

## Implemented Algorithms

- Lookup-based
- Vector search

## Components

- `ReferenceReader`
  - Class for _iterating_ publications and publication-concept links.
  - Current primary use case for this class is to encode publication vectors.
- `ReferenceDatabase`
  - Class for retrieve publications and publication-concept links given identifiers.
- `PublicationEncoder`
  - Class for encoding publications into vectors.
- `PublicationVectorDatabase`
  - Class for storing vectors encoded by `PublicationEncoder`s.
- `Retriever`
  - Class for retrieving contexts given triples.
- `PublicationScorer`
  - Class for scoring publications.
- `ContextSummarizer`
  - Class for summarizing the result of `Retriever`s.

## Test

- For tests that don't require Milvus or PostgreSQL, you can run `uv run pytest`.
- For tests with Milvus or PostgreSQL connection, you can run `uv run pytest -m {milvus,postgres}` after running a server.
  - Docker compose files for running servers are available at `scripts/{milvus,postgres}`.
  - Run `docker compose -f test-docker-compose.yaml up` (or with `sudo`).

## Secrets and env vars

This repository does not contain real credentials. An example env file is provided at `.env-example` — copy it to `.env` and fill values for local development. Do not commit your `.env` file; it is included in `.gitignore`.

- Use environment variables for credentials (e.g. `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `POSTGRES_PASSWORD`).
- During image builds, use docker build secret mounts (no secrets in image layers).
- Before publishing, run a git-history secret scan (e.g. `trufflehog` or `ggshield`) and rotate any exposed credentials.

