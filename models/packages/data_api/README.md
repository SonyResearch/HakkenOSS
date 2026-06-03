# Data API Service

A FastAPI-based service for data retrieval.  
## Overview

A FastAPI-based service providing data retrieval endpoints.

## Installation

```bash
uv sync
```

## Usage

Start the service:

```bash
uv run python service.py
```

## Testing

```bash
uv run pytest
```

## Secrets and env vars

This repository does not contain real credentials. An example env file is provided at `.env-example` — copy it to `.env` and fill values for local development. Do not commit your `.env` file; it is included in `.gitignore`.

- Use environment variables for credentials (e.g. registry tokens, DB passwords).
- During image builds, use Docker build secret mounts (`--secret`) — no secrets are baked into image layers.
- Before publishing, run a git-history secret scan (e.g. `trufflehog` or `ggshield`) and rotate any exposed credentials.
