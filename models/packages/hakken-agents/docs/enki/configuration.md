# Enki configuration

This page covers environment variables, document and pipeline options, and advanced Hydra overrides for the Enki pipeline. Run from the hakken-agents package root with:

```bash
uv run python scripts/run_enki_ingest.py [overrides]
```

## Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `DOCS_FOLDER` | Document configs | Directory containing source files (e.g. `braca1.txt`, `paris.txt`). Document YAMLs set `path: ${oc.env:DOCS_FOLDER}/<filename>`. |
| `OPENAI_API_KEY` | LLM / embedder | Used when entity_extractor or fact_extractor (or embedder) is configured for OpenAI. |
| `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL_NAME` | LLM / embedder | Used when resolver or extractor uses OpenRouter. |
| `POSTGRES_*` | DB and resolvers | `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` — same as Element Resolver / Docker Compose defaults. |

## Document source

- **Preset**: Default is `document=braca1`. Switch with e.g. `document=paris`, `document=foxo3`, `document=causalflows`. The preset must exist under `configs/enki/document/<name>.yaml`.
- **Override path**: Use `document.path=/absolute/or/relative/path/to/file.txt` (or `.pdf`) to point to any file without a preset. `DOCS_FOLDER` is not used when you set `document.path` explicitly.

## Knowledge-graph profiles

The `kg` config group sets table names and workspace for a given knowledge graph. Use it to run multiple graphs (e.g. default, agro) against the same database with **separate tables**.

- **Default**: `kg=default` (or omit) — tables `chunks_vectors`, `entities_vectors`, `domains_vectors`, `relations_vectors`, `facts`; workspace `enki-ingest`.
- **Agro**: `kg=agro` — tables `agro_chunks_vectors`, `agro_entities_vectors`, `agro_domains_vectors`, `agro_relations_vectors`, `agro_facts`; workspace `enki-agro`.

**Add a new profile**: Copy `configs/enki/kg/agro.yaml` to e.g. `configs/enki/kg/my_kg.yaml`, set `workspace`, table names under `chunks_table.name`, `entities_table.name`, etc., and `facts_table_name`. Run with `kg=my_kg`.

## Prompts

Entity and fact extractors use prompts from a **prompt registry** (Python modules under `hakken_agents/enki/nodes/*/prompts.py`). Prompts are identified by string IDs (e.g. `entity_extractor.system.default`, `entity_extractor.user.default`, `fact_extractor.user.default`). You can override which prompt is used from the CLI:

```bash
uv run python scripts/run_enki_ingest.py entity_extractor.user_prompt_id=entity_extractor.user.strict
```

Available IDs are listed in the registry; add new prompts by registering them in the corresponding node’s `prompts.py` or at runtime via `prompt_registry.register(...)` (see `hakken_agents.enki.prompts`).

## Entity extractor options

| Option | Config key | Default | Description |
|--------|------------|---------|-------------|
| Relevant domains | `entity_extractor.use_relevant_domains` | `false` | If `true`, the pipeline retrieves relevant domains from the domain resolver (semantic search) and passes them to the entity extractor to bias extraction. |
| Allowed domains | `entity_extractor.allowed_domains` | `null` (none) | Optional list of domain names the extractor is allowed to output. When set (via an allowed_domains config group), only entities in these domains are kept. |

**Allowed domains presets** (optional Hydra group `entity_extractor/allowed_domains`):

- `biomedical_simple` — short list of biomedical entity types (gene, protein, drug, disease, etc.).
- `biomedical` — extended biomedical list.
- `machine_learning` — ML-related entity types.

Example: restrict entities to biomedical domains:

```bash
uv run python scripts/run_enki_ingest.py entity_extractor/allowed_domains=biomedical_simple
```

**LLM for entity extraction**: The entity extractor uses the default `entity_extractor/llm` config (e.g. OpenRouter). To use local Ollama:

```bash
uv run python scripts/run_enki_ingest.py entity_extractor/llm=ollama
```

Ollama must be running (e.g. `ollama run llama3.2`). Override base URL if needed: `entity_extractor/llm.base_url=http://localhost:11434`.

## Fact extractor options

| Option | Config key | Default | Description |
|--------|------------|---------|-------------|
| Preferred relation types | `fact_extractor.preferred_relation_types` | `[]` (none) | List of relation type names to prefer when extracting facts. Can be set via an optional config group `fact_extractor/preferred_relation_types`. |
| Relevant relation types | `fact_extractor.use_relevant_relation_types` | `true` | If `true`, the pipeline retrieves relevant relation types from the fact/relation resolver (semantic search) and passes them to the fact extractor. |

**Preferred relation types presets** (optional Hydra group `fact_extractor/preferred_relation_types`):

- `machine_learning` — ML-oriented relations (e.g. `trained_on`, `outperforms`, `proposed_by`, `uses_dataset`).

Example: use ML relation types and Ollama for both entity and fact extraction:

```bash
uv run python scripts/run_enki_ingest.py entity_extractor/llm=ollama fact_extractor/llm=ollama fact_extractor/preferred_relation_types=machine_learning
```

**LLM for fact extraction**: Same pattern as entity extractor. Default comes from `fact_extractor/llm`. Use Ollama:

```bash
uv run python scripts/run_enki_ingest.py fact_extractor/llm=ollama
```

## LLM and embedder

The pipeline uses shared `llm` and `embedder` configs for resolvers (chunk, domain, entity, relation). Entity and fact **extractors** each have their own LLM config (`entity_extractor/llm`, `fact_extractor/llm`). You can mix providers, for example:

```bash
# Ollama for both extractors; keep default (e.g. OpenRouter) for resolvers
uv run python scripts/run_enki_ingest.py entity_extractor/llm=ollama fact_extractor/llm=ollama

# Ollama for entity extraction only
uv run python scripts/run_enki_ingest.py entity_extractor/llm=ollama
```

Embedder is used by all resolvers; override globally if your config supports it (e.g. via `embedder=ollama` if that group exists).

## Workspace

The **workspace** isolates pipeline cache and status. It is set by the active **knowledge-graph profile** (`kg`): e.g. `enki-ingest` for `kg=default`, `enki-agro` for `kg=agro`. You can still override it from the CLI if needed:

```bash
uv run python scripts/run_enki_ingest.py kg=agro workspace=my_custom_workspace
```

## Hydra override summary

| Override | Example |
|----------|---------|
| Knowledge-graph profile | `kg=agro` |
| Document preset | `document=paris` |
| Document path | `document.path=/path/to/doc.txt` |
| Entity extractor LLM | `entity_extractor/llm=ollama` |
| Fact extractor LLM | `fact_extractor/llm=ollama` |
| Entity extractor prompt | `entity_extractor.user_prompt_id=entity_extractor.user.strict` |
| Fact extractor prompt | `fact_extractor.system_prompt_id=fact_extractor.system.default` |
| Relevant domains | `entity_extractor.use_relevant_domains=true` |
| Allowed domains preset | `entity_extractor/allowed_domains=biomedical_simple` |
| Preferred relation types preset | `fact_extractor/preferred_relation_types=machine_learning` |
| Workspace | `workspace=my_project` (overrides the value from `kg`) |

All overrides can be combined. Run with `--help` or inspect the composed config (e.g. with `print_config: true` in `enki.yaml`) to see the full structure.
