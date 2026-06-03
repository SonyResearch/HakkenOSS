# Element Resolver

Ingest entities into a PostgreSQL + pgvector vector store and run semantic similarity search. There are two interfaces: a **CLI** for batch ingestion from TSV files and a **REST API** for programmatic ingest and search.

Both interfaces share the same core `ElementResolver` and store configuration in a **table registry** so that embedder model, dimension, and metadata columns are recorded once during ingestion and automatically loaded by the API at startup.

---

## CLI (`run_element_resolver.py`)

A Typer-based CLI with two sub-commands: `ingest` and `query`.

### Ingest

Read a TSV file (S3 or local), render each row through a Jinja2 content template, optionally generate an LLM description, and insert into the vector store.

```bash
uv run python scripts/run_element_resolver.py ingest \
  --data-uri s3://bucket/path/nodes.tsv \
  --table-name my_table \
  --content-columns name,context \
  --content-template "{{ name }}"
```

Key options:

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--data-uri` | `-i` | *(required)* | S3 or local path to a TSV file |
| `--table-name` | `-t` | *(required)* | Target vector-store table name |
| `--content-columns` | | *(required)* | Comma-separated columns to store as metadata |
| `--content-template` | | *(required)* | Jinja2 template rendered per row to produce the `content` field |
| `--model` | `-m` | `openai/gpt-4.1-nano` | LLM for description generation |
| `--embedder` | `-e` | `openai/text-embedding-3-small` | Embedding model |
| `--batch-size` | `-b` | `50` | Elements per LLM / insert batch |
| `--max-concurrency` | `-c` | `10` | Max concurrent LLM calls per batch |
| `--no-description` | | `false` | Skip LLM descriptions; embed raw content only |
| `--limit` | `-n` | *(all rows)* | Max rows to ingest |

The CLI also accepts `--model-temperature`, `--model-base-url`, and `--embedder-base-url` to override provider settings. When a base URL contains `localhost`, the API key is automatically omitted (useful for local Ollama).

On first ingest the CLI validates (or creates) a **table registry** entry that records the embedder model, dimension, base URL, and metadata columns. The API reads this registry at startup so it doesn't need to duplicate that configuration.

### Query

Search the vector store for similar entities.

```bash
uv run python scripts/run_element_resolver.py query "bacterium Desulfurivibrio" \
  --table-name my_table --k 5
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--table-name` | `-t` | *(required)* | Table to search (must match ingest) |
| `--k` | `-k` | `5` | Number of results |
| `--filter` | `-f` | | Metadata filter as JSON (e.g. `'{"context":{"$ilike":"%GENE%"}}'`) |
| `--threshold` | | | Minimum similarity score (0–1) |
| `--json` | `-j` | `false` | Output results as JSON |

Query also accepts `--model`, `--embedder-name`, and provider override flags (`--model-api-key`, `--model-base-url`, `--embedder-api-key`, `--embedder-base-url`).

---

## REST API (`run_element_resolver_api.py`)

A FastAPI server configured via [Hydra](https://hydra.cc/) YAML files.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/element_resolver/ingest` | Ingest elements |
| POST | `/api/v1/element_resolver/search` | Semantic search |
| GET | `/api/v1/element_resolver/filter-columns` | List filterable metadata columns |
| GET | `/api/v1/element_resolver/elements/{id}` | Get element by UUID |

### Start the server

```bash
# Option 1: Direct script
uv run python scripts/run_element_resolver_api.py

# Option 2: Makefile (loads .env.local)
make element-api
```

Override Hydra config groups on the command line:

```bash
uv run python scripts/run_element_resolver_api.py resolver/llm=ollama resolver/embedder=ollama
```

At startup the API reads the **table registry** to load the embedder model, dimension, and metadata columns that were written during CLI ingestion. If no registry entry exists, the YAML defaults are used.

See [Configuration](configuration.md) for Hydra config groups, env-var interpolation, and provider options. See [Examples](cli-examples.md) for curl commands.
