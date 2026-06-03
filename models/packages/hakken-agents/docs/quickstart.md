# Quick start

Two ways to get started: run the **Enki pipeline** (document → entities/facts → DB) or use the **Element Resolver** (CLI for batch TSV ingestion + REST API for programmatic ingest and search). Use one or both; they are independent.

All commands assume you are in the `hakken-agents` package root and use `uv run` for script execution.

---

## Path 1: Enki pipeline

Ingest documents, extract entities and facts, and store them in PostgreSQL.

### 1. Start PostgreSQL (Docker)

```bash
docker compose up -d
```

This starts a Postgres 17 container with the [pgvector](https://github.com/pgvector/pgvector) extension. Defaults: `POSTGRES_USER=postgres`, `POSTGRES_PASSWORD=postgres`, `POSTGRES_DB=hakken_agents`, port `5432`.

### 2. Environment

```bash
cp .env.example .env
```

Edit `.env`:

- **LLM**: Set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (and `OPENROUTER_MODEL_NAME` if using OpenRouter). For local Ollama, no keys are needed — see [Enki configuration](enki/configuration.md#llm--embedder).
- **Documents**: Set `DOCS_FOLDER` to the directory that contains your document files (e.g. the path to `documents/` or where `braca1.txt`, `paris.txt`, etc. live).
- **PostgreSQL**: Defaults match Docker Compose; change only if you use a different host/port.

### 3. Install dependencies

```bash
uv sync
```

(This installs default groups, including `graph-builder`, required for the Enki pipeline.)

### 4. Run the Enki pipeline

```bash
uv run python scripts/run_enki_ingest.py
```

This uses the default document config (`document=braca1`). Tables are created automatically on first run.

**Use another document preset** (e.g. `paris`, `foxo3`):

```bash
uv run python scripts/run_enki_ingest.py document=paris
```

**Override the document path** (any file):

```bash
uv run python scripts/run_enki_ingest.py document.path=/path/to/your/doc.txt
```

**Use a different knowledge graph** (separate tables and workspace, e.g. for agro):

```bash
uv run python scripts/run_enki_ingest.py kg=agro document.path=/path/to/agro_doc.txt
```

More options: [Enki overview](enki/overview.md) and [Enki configuration](enki/configuration.md).

### 5. (Optional) Inspect the database

```bash
uv run python scripts/manage_db.py list-tables
```

See `uv run python scripts/manage_db.py --help` for more commands (describe table, query, drop table, etc.).

---

## Path 2: Element Resolver (CLI + API)

Ingest entities into a vector store and run semantic similarity search. The **CLI** handles batch ingestion from TSV files; the **API** exposes ingest and search over REST.

### 1. Start PostgreSQL (Docker)

```bash
docker compose up -d
```

Same as above: Postgres 17 with pgvector, defaults on port `5432`.

### 2. Environment

```bash
cp .env.example .env
```

Edit `.env`:

- **LLM / Embedder**: Set `OPENAI_API_KEY` or `OPENROUTER_API_KEY`. For **local Ollama**, no API keys are needed — see [Element Resolver configuration](element-resolver/configuration.md#ollama-local).
- **PostgreSQL**: Defaults match Docker Compose; change only if needed.

### 3. Install dependencies

```bash
uv sync
```

### 4a. Ingest via CLI

```bash
uv run python scripts/run_element_resolver.py ingest \
  --data-uri s3://bucket/path/nodes.tsv \
  --table-name my_table \
  --content-columns name,context \
  --content-template "{{ name }}"
```

This reads a TSV file, renders each row through a Jinja2 template, optionally generates LLM descriptions, and inserts into the vector store. A **table registry** entry is created so the API can auto-load the embedder and schema.

Query the ingested data:

```bash
uv run python scripts/run_element_resolver.py query "bacterium" -t my_table --k 5
```

See `uv run python scripts/run_element_resolver.py --help` for all options.

### 4b. Run the Element Resolver API

```bash
# Option 1: Direct script
uv run python scripts/run_element_resolver_api.py

# Option 2: Makefile (loads .env.local)
make element-api
```

The API uses [Hydra](https://hydra.cc/) configuration from `configs/element_resolver_api.yaml`. At startup it reads the table registry (written by the CLI ingest) to auto-configure the embedder model and metadata columns.

Override providers via Hydra:

```bash
uv run python scripts/run_element_resolver_api.py resolver/llm=ollama resolver/embedder=ollama
```

See [Configuration](element-resolver/configuration.md) for Hydra config groups, env-var interpolation, and provider options.
