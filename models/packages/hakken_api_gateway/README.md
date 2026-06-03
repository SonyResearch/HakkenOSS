# Hakken API Gateway

A FastAPI-based API gateway service that provides a unified entry point for Hakken services.

## Overview

The API gateway handles routing, authentication, and request dispatching to downstream Hakken services.

Key modules:
- `hakken_api_gateway/api_gateway.py` — main gateway logic
- `hakken_api_gateway/router_v1.py` — v1 API routes
- `hakken_api_gateway/auth/` — authentication middleware
- `hakken_api_gateway/container.py` — dependency injection container
- `hakken_api_gateway/config.py` — configuration management
- `hakken_api_gateway/entities.py` — data models

## Installation

```bash
uv sync
```

## Configuration

The service reads configuration from environment variables or a `.env` file. See `.env-example` for required variables.

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
