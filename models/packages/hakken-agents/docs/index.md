# Hakken Agents

Document ingestion and knowledge extraction (Enki) and entity resolution with semantic search (Element Resolver). Both use PostgreSQL and pgvector.

## Quick start

Choose one of two paths:

| Path | Description |
|------|--------------|
| [Quick start – Enki pipeline](quickstart.md#path-1-enki-pipeline) | Ingest documents → extract entities and facts → store in PostgreSQL |
| [Quick start – Element Resolver](quickstart.md#path-2-element-resolver-cli--api) | Batch-ingest entities from TSV (CLI) and run semantic similarity search (CLI + REST API) |

Enki and the Element Resolver are **independent use cases**: you can run either (or both) depending on your needs.

## Enki (document pipeline)

| Page | Description |
|------|--------------|
| [Overview](enki/overview.md) | Pipeline steps, config layout, document presets |
| [Configuration](enki/configuration.md) | Knowledge-graph profiles, prompts, document source, entity/fact extractor options, LLM/embedder, Hydra overrides |

## Element Resolver (CLI + API)

| Page | Description |
|------|--------------|
| [Overview](element-resolver/overview.md) | CLI commands, API endpoints, table registry |
| [Configuration](element-resolver/configuration.md) | Hydra config groups, env vars, LLM / embedder options (OpenRouter, OpenAI, Ollama) |
| [Examples](element-resolver/cli-examples.md) | CLI and curl examples for ingest and search |
