# Enki pipeline overview

Enki is the document-ingestion and knowledge-extraction pipeline in hakken-agents. It parses documents (PDF or text), chunks them, extracts entities and facts with LLMs, resolves them against PostgreSQL-backed vector stores, and writes chunks, entities, domains, relations, and facts to the database.

## Pipeline steps

1. **Parse** — Document parser (with optional cache) reads the source file and produces structured content (e.g. from PDF via MinerU).
2. **Chunk** — Content is split into text chunks (configurable size; default uses a recursive character text splitter).
3. **Chunk resolution** — Each chunk is stored in the chunk resolver table (vector store) for deduplication and retrieval.
4. **Entity extraction** — Per chunk, an LLM extracts entities (name, domain, description). Optional: restrict to **allowed domains** or use **relevant domains** from the domain resolver.
5. **Domain / entity resolution** — Extracted entities are resolved against the domains table and entities table (embedding similarity + optional LLM), producing resolved domain and entity IDs.
6. **Fact extraction** — For chunks with resolved entities, an LLM extracts facts (subject, relation, object). Optional: **preferred** or **relevant relation types** from the fact resolver.
7. **Fact resolution** — Facts are resolved to relation types and stored in the relations and facts tables.

All resolver components use the same PostgreSQL + pgvector backend (tables for chunks, domains, entities, relations, facts). The pipeline is configured via Hydra; the entry script is `scripts/run_enki_ingest.py`.

## Config layout

Configuration is under `configs/enki/`, composed by Hydra from the main entry point `configs/enki/enki.yaml`.

| Area | Config group / path | Purpose |
|------|---------------------|---------|
| **Knowledge graph** | `kg` (e.g. `default`, `agro`) | Table names, facts table name, and workspace for a given KG. One flag switches all storage for that graph. |
| Document | `document` (e.g. `braca1`, `paris`) | Source file path (often `${oc.env:DOCS_FOLDER}/...`), encoding, lang, metadata |
| Chunks / entities / domains / relations | `*_table: default` | Vector table structure (columns, schema). Table **names** are overridden by the `kg` profile. |
| Entity extraction | `entity_extractor` | LLM, prompt IDs (registry), `use_relevant_domains`, optional `allowed_domains` |
| Fact extraction | `fact_extractor` | LLM, prompt IDs (registry), `preferred_relation_types`, `use_relevant_relation_types` |
| Resolvers | Derived from `llm`, `embedder`, `db`, and `*_table` | Chunk, domain, entity, and fact resolvers (not separate YAML groups) |
| Text splitting | `text_splitter` | Chunk size and strategy |

Default document is set in `enki.yaml` with `defaults: [..., document: braca1, ...]`. You override from the CLI with `document=paris` or `document.path=/path/to/file.txt`. To use a different knowledge graph (separate tables and workspace), use `kg=agro` (or another profile defined under `configs/enki/kg/`).

## Document presets

Presets are Hydra config groups under `configs/enki/document/`. Each sets `path` (typically `${oc.env:DOCS_FOLDER}/<filename>`) and optional metadata.

**Common text presets:**

| Preset | File |
|--------|------|
| `braca1` | `braca1.txt` |
| `paris` | `paris.txt` |
| `foxo3` | `foxo3.txt` |
| `causalflows` | `2306.05415v2.pdf` |

Additional presets exist for other PDFs (e.g. `549210v2`, `582168v3`, `596284v1`, `619522v1`, `687483v1`, `701154v1`). Ensure `DOCS_FOLDER` points to a directory that contains the corresponding file for the chosen preset.

## Knowledge-graph profiles

The `kg` config group defines which tables and workspace a run uses. This lets you maintain **multiple knowledge graphs** (e.g. default, agro, biomedical) in the same database with separate tables.

| Profile | Config file | Tables / workspace |
|---------|-------------|--------------------|
| `default` | `configs/enki/kg/default.yaml` | `chunks_vectors`, `entities_vectors`, `domains_vectors`, `relations_vectors`, `facts`; workspace `enki-ingest` |
| `agro` | `configs/enki/kg/agro.yaml` | `agro_chunks_vectors`, `agro_entities_vectors`, etc., and `agro_facts`; workspace `enki-agro` |

Use from the CLI: `kg=agro` (or add a new YAML under `configs/enki/kg/` and use `kg=your_profile`). See [Configuration → Knowledge-graph profiles](configuration.md#knowledge-graph-profiles) for how to add a new profile.

## Tables created

On first run, the pipeline creates (or reuses) tables used by the resolvers and the fact store. **Table names** come from the active `kg` profile; **schema** (columns, metadata) comes from the `*_table: default` configs.

- Chunk, domain, entity, and relation vector/store tables.
- Facts table for resolved facts linked to chunks and entities.

List tables and row counts with:

```bash
uv run python scripts/manage_db.py list-tables
```

See [Configuration](configuration.md) for environment variables, Hydra overrides, and advanced options (allowed domains, relation types, LLM/embedder per component).
