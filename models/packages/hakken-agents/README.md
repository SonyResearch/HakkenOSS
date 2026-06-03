# Hakken Agents

Document ingestion and knowledge extraction pipelines (Enki) and entity resolution with semantic search (Element Resolver). Both use PostgreSQL + pgvector.

## Quick start

### 1. Start PostgreSQL (Docker)

From the `hakken-agents` package root:

```bash
docker compose up -d
```

This starts a Postgres 17 container with the [pgvector](https://github.com/pgvector/pgvector) extension. Defaults: `POSTGRES_USER=postgres`, `POSTGRES_PASSWORD=postgres`, `POSTGRES_DB=hakken_agents`, port `5432`.

### 2. Environment

Copy the example env and set required variables:

```bash
cp .env.example .env
```

Edit `.env`:

- **LLM**: Set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (and `OPENROUTER_MODEL_NAME` if using OpenRouter).
- **Documents**: Set `DOCS_FOLDER` to the directory that contains your document files. For the bundled examples, use the absolute path to `documents/`, e.g.:
  ```bash
  DOCS_FOLDER=/Users/you/path/to/hakken-agents/documents
  ```
- **PostgreSQL**: Defaults match the Docker Compose service (`POSTGRES_HOST=localhost`, `POSTGRES_PORT=5432`, etc.). Change only if you use a different host/port.

### 3. Install dependencies

From the `hakken-agents` package root:

```bash
uv sync
```

(This installs the default groups, including `graph-builder`, required for the Enki pipeline.)

### 4. Run the Enki pipeline

From the **hakken-agents package root** (so config and prompt paths resolve):

```bash
uv run python scripts/run_enki_ingest.py
```

This runs the pipeline with the default document config (`document=braca1`). Tables are created automatically on first run.

**Use another document preset** (e.g. `paris`, `foxo3`):

```bash
uv run python scripts/run_enki_ingest.py document=paris
```

**Override the document path** (any text file):

```bash
uv run python scripts/run_enki_ingest.py document.path=/path/to/your/doc.txt
```

**Use a different knowledge graph** (separate tables and workspace):

```bash
uv run python scripts/run_enki_ingest.py kg=agro document.path=/path/to/agro_doc.txt
```

**Override other options** (Hydra overrides):

```bash
uv run python scripts/run_enki_ingest.py entity_extractor.use_relevant_domains=false
```

For more Enki options (knowledge-graph profiles, prompts, allowed domains, LLM/embedder per component, fact extractor settings), see [Enki configuration](docs/enki/configuration.md).

### 5. (Optional) Inspect the database

List tables and row counts:

```bash
uv run python scripts/manage_db.py list-tables
```

See `uv run python scripts/manage_db.py --help` for more commands (describe table, query, drop table, etc.).

### 6. Element Resolver

The Element Resolver provides a CLI for batch-ingesting entities from TSV files and a REST API for programmatic ingest and search.

**Ingest from TSV:**

```bash
uv run python scripts/run_element_resolver.py ingest \
  --data-uri s3://bucket/path/nodes.tsv \
  --table-name my_table \
  --content-columns name,context \
  --content-template "{{ name }}"
```

**Query:**

```bash
uv run python scripts/run_element_resolver.py query "bacterium" -t my_table --k 5
```

**Run the API server** (Hydra config, auto-loads embedder from the table registry):

```bash
uv run python scripts/run_element_resolver_api.py
# Or: make element-api  (loads .env.local)
```

See `uv run python scripts/run_element_resolver.py --help` and [Element Resolver docs](docs/element-resolver/overview.md) for full options.

## Summary

| Step            | Command / action                                                       |
|-----------------|-------------------------------------------------------------------------|
| Start DB        | `docker compose up -d`                                                 |
| Configure       | Copy `.env.example` → `.env`, set `DOCS_FOLDER`, LLM keys             |
| Install         | `uv sync` (from package root)                                         |
| Run Enki        | `uv run python scripts/run_enki_ingest.py`                            |
| Change document | `... run_enki_ingest.py document=paris` or `document.path=...`        |
| Different KG    | `... run_enki_ingest.py kg=agro` (separate tables + workspace)         |
| Ingest entities | `uv run python scripts/run_element_resolver.py ingest -i data.tsv ...` |
| Query entities  | `uv run python scripts/run_element_resolver.py query "term" -t table`  |
| Resolver API    | `uv run python scripts/run_element_resolver_api.py` or `make element-api` |

For the full Enki and Element Resolver reference, see the [documentation](docs/index.md).

## Secrets and env vars

This repository does not contain real credentials. An example env file is provided at `.env-example` — copy it to `.env` and fill values for local development. Do not commit your `.env` file; it is included in `.gitignore`.

- Use environment variables for credentials (e.g. registry tokens, DB passwords).
- During image builds, use Docker build secret mounts (`--secret`) — no secrets are baked into image layers.
- Before publishing, run a git-history secret scan (e.g. `trufflehog` or `ggshield`) and rotate any exposed credentials.
