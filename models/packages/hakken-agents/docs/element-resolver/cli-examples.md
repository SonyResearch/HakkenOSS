# Element Resolver Examples

## Prerequisites

For API examples, start the server first:

```bash
# Option 1: Direct script (loads .env from package root)
uv run python scripts/run_element_resolver_api.py

# Option 2: Makefile (loads .env.local)
make element-api
```

Default base URL: `http://localhost:8000`.

---

## CLI — Ingest from TSV

The `nodes` table schema expects metadata columns: `node_name`, `node_id_raw`, `node_id`, `node_domain`, `node_domain_id`. Your TSV must have matching column headers.

```bash
# Basic ingest (S3 source, LLM descriptions enabled)
uv run python scripts/run_element_resolver.py ingest \
  --data-uri s3://bucket/path/nodes.tsv \
  --table-name nodes_vectors \
  --content-columns node_name,node_id_raw,node_domain,node_domain_id,node_id \
  --content-template "{{ node_name }}{% if node_id_raw %} | {{ node_id_raw }}{% endif %} || {{ node_domain }}"

# Ingest with no LLM description (embed raw content only)
uv run python scripts/run_element_resolver.py ingest \
  -i nodes.tsv -t nodes_vectors \
  --content-columns node_name,node_id_raw,node_domain,node_domain_id,node_id \
  --content-template "{{ node_name }}{% if node_id_raw %} | {{ node_id_raw }}{% endif %} || {{ node_domain }}" \
  --no-description

# Limit to first 100 rows, custom batch size
uv run python scripts/run_element_resolver.py ingest \
  -i nodes.tsv -t nodes_vectors \
  --content-columns node_name,node_id_raw,node_domain,node_domain_id,node_id \
  --content-template "{{ node_name }}{% if node_id_raw %} | {{ node_id_raw }}{% endif %} || {{ node_domain }}" \
  --limit 100 --batch-size 25 --max-concurrency 5
```

## CLI — Query

```bash
# Simple search
uv run python scripts/run_element_resolver.py query "bacterium Desulfurivibrio" \
  -t nodes_vectors --k 5

# Search with metadata filter (node_domain, node_name, etc.)
uv run python scripts/run_element_resolver.py query "gene" -t nodes_vectors \
  --filter '{"node_domain": {"$ilike": "%gene%"}}'

# JSON output with similarity threshold
uv run python scripts/run_element_resolver.py query "kinase" -t nodes_vectors \
  --k 10 --threshold 0.7 --json
```

---

## API — Health check

```bash
curl -s http://localhost:8000/health
```

## API — Ingest elements

For the `nodes` table, metadata must use columns: `node_name`, `node_id_raw`, `node_id`, `node_domain`, `node_domain_id`.

```bash
curl -s -X POST http://localhost:8000/api/v1/element_resolver/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "elements": [
      {
        "content": "Desulfurivibrio | DSM 1924 || gene",
        "metadata": {
          "node_name": "Desulfurivibrio",
          "node_id_raw": "DSM 1924",
          "node_id": "ent-001",
          "node_domain": "gene",
          "node_domain_id": "gene:1"
        }
      }
    ]
  }'
```

Skip LLM description generation:

```bash
curl -s -X POST http://localhost:8000/api/v1/element_resolver/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "elements": [
      {
        "content": "Desulfurivibrio || gene",
        "metadata": {
          "node_name": "Desulfurivibrio",
          "node_id_raw": "",
          "node_id": "ent-001",
          "node_domain": "gene",
          "node_domain_id": "gene:1"
        }
      }
    ],
    "no_description": true
  }'
```

Control concurrency for description generation:

```bash
curl -s -X POST http://localhost:8000/api/v1/element_resolver/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "elements": [
      {"content": "Entity A || gene", "metadata": {"node_name": "Entity A", "node_id_raw": "", "node_id": "a-1", "node_domain": "gene", "node_domain_id": "gene:1"}},
      {"content": "Entity B || disease", "metadata": {"node_name": "Entity B", "node_id_raw": "", "node_id": "b-1", "node_domain": "disease", "node_domain_id": "disease:1"}}
    ],
    "max_concurrency": 3
  }'
```

The response reports which elements were newly ingested vs. already present (deduplication by deterministic UUID):

```json
{
  "ingested": ["uuid-1"],
  "skipped": ["uuid-2"]
}
```

## API — Search

```bash
curl -s -X POST http://localhost:8000/api/v1/element_resolver/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "bacterium Desulfurivibrio",
    "k": 5
  }'
```

With metadata filter:

```bash
curl -s -X POST http://localhost:8000/api/v1/element_resolver/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "bacterium",
    "k": 5,
    "filter": {"node_domain": {"$ilike": "%gene%"}}
  }'
```

With similarity threshold:

```bash
curl -s -X POST http://localhost:8000/api/v1/element_resolver/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "kinase",
    "k": 10,
    "threshold": 0.7
  }'
```

With filter and threshold combined:

```bash
curl -s -X POST http://localhost:8000/api/v1/element_resolver/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "nitrogen fertilizer",
    "k": 3,
    "threshold": 0.5,
    "filter": {"node_domain": {"$ilike": "%gene%"}}
  }'
```

## API — Filter columns

List metadata columns available for search filters:

```bash
curl -s http://localhost:8000/api/v1/element_resolver/filter-columns
```

## API — Get element by ID

```bash
# Replace <element_id> with an actual UUID from ingest or search results
curl -s http://localhost:8000/api/v1/element_resolver/elements/<element_id>
```
