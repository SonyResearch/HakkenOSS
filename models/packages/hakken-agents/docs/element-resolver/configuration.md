# Element Resolver Configuration

The **API** (`run_element_resolver_api.py`) uses [Hydra](https://hydra.cc/) config groups under `configs/`. The **CLI** (`run_element_resolver.py`) accepts the same settings as command-line flags.

---

## Hydra config layout (API)

The top-level config is `configs/element_resolver_api.yaml`:

```yaml
defaults:
  - resolver: default
  - resolver/llm: default
  - resolver/embedder: default
  - resolver/db: default
  - resolver/table: nodes
  - _self_

host: "0.0.0.0"
port: 8000

resolver:
  table:
    name: pubtator3-v0_4_0-nodes_vectors   # override per deployment
```

Config groups live in `configs/resolver/`:

| Group | Files | Purpose |
|-------|-------|---------|
| `resolver/llm` | `default`, `openrouter`, `openai`, `ollama` | LLM provider for description generation |
| `resolver/embedder` | `default`, `openrouter`, `openai`, `ollama` | Embedding provider |
| `resolver/db` | `default` | PostgreSQL connection |
| `resolver/table` | `nodes` | Vector-store table schema (metadata columns, content/embedding) |

### Environment-variable interpolation

The `default` config files use `${oc.env:VAR}` to read values from `.env` at runtime:

```yaml
# resolver/llm/default.yaml
name: ${oc.env:LLM_NAME,openai/gpt-4.1-nano}
temperature: ${oc.env:LLM_TEMPERATURE,0.2}
api_key: ${oc.env:OPENROUTER_API_KEY}
base_url: ${oc.env:OPENROUTER_BASE_URL}
```

```yaml
# resolver/embedder/default.yaml
api_key: ${oc.env:OPENROUTER_API_KEY}
base_url: ${oc.env:OPENROUTER_BASE_URL}
embedding_model: ${oc.env:EMBEDDER_NAME,openai/text-embedding-3-small}
embedding_dim: 1536
```

Set these in `.env` (or export them) before running the API.

### Table registry auto-configuration

When the CLI ingests data it writes a **registry entry** to the `element_resolver_registry` table recording the embedder model, dimension, base URL, and metadata columns. At API startup, the lifespan handler reads this registry and patches the resolver config automatically — so the YAML only needs the `table.name` and operational settings (LLM, DB, credentials).

---

## LLM options

| Config | Provider | Env vars | Default model |
|--------|----------|----------|---------------|
| `default` | From `.env` | `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` | `openai/gpt-4.1-nano` |
| `openrouter` | OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-4.1-nano` |
| `openai` | OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `ollama` | Local Ollama | — | `llama3.2` |

## Embedder options

| Config | Provider | Env vars | Default model | Dim |
|--------|----------|----------|---------------|-----|
| `default` | From `.env` | `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` | `openai/text-embedding-3-small` | 1536 |
| `openrouter` | OpenRouter | `OPENROUTER_API_KEY` | `openai/text-embedding-3-small` | 1536 |
| `openai` | OpenAI | `OPENAI_API_KEY` | `text-embedding-3-small` | 1536 |
| `ollama` | Local Ollama | — | `nomic-embed-text` | 768 |

---

## Override via CLI (API)

```bash
# Use Ollama for both LLM and embedder
uv run python scripts/run_element_resolver_api.py resolver/llm=ollama resolver/embedder=ollama

# Mix providers
uv run python scripts/run_element_resolver_api.py resolver/llm=openrouter resolver/embedder=openai

# Override a single value
uv run python scripts/run_element_resolver_api.py resolver.table.name=my_custom_table
```

## Override via CLI (Ingest/Query)

The CLI script accepts flags directly:

```bash
# Ingest with Ollama
uv run python scripts/run_element_resolver.py ingest \
  -i data.tsv -t my_table \
  --content-columns name --content-template "{{ name }}" \
  -m llama3.2 --model-base-url http://localhost:11434 \
  -e nomic-embed-text --embedder-base-url http://localhost:11434

# Query with a different model
uv run python scripts/run_element_resolver.py query "term" -t my_table \
  -m openai/gpt-4.1-nano --embedder-name openai/text-embedding-3-small
```

---

## Ollama (local)

No API keys required. Install [Ollama](https://ollama.com) and pull the models:

**LLM** (for entity descriptions):

```bash
ollama run llama3.2
```

Or use `mistral`, `llama3.1`, etc.

**Embedder** (for semantic search):

```bash
ollama run nomic-embed-text
```

Then run the API:

```bash
uv run python scripts/run_element_resolver_api.py resolver/llm=ollama resolver/embedder=ollama
```

Or ingest via CLI:

```bash
uv run python scripts/run_element_resolver.py ingest \
  -i data.tsv -t my_table \
  --content-columns name --content-template "{{ name }}" \
  -m llama3.2 --model-base-url http://localhost:11434 \
  -e nomic-embed-text --embedder-base-url http://localhost:11434
```

Ollama must be running (default: `http://localhost:11434`). Override the host via `resolver/llm.base_url=...` / `resolver/embedder.base_url=...` (API) or `--model-base-url` / `--embedder-base-url` (CLI).
