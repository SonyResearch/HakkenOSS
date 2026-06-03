
SHELL := /bin/bash

.PHONY: checks element-ingest-nodes element-ingest-relations


checks:
	uv run ruff format
	uv run ruff check . --fix
	uv run pytest
	uv run mypy


enki-ingest-ml:
	HYDRA_FULL_ERROR=1 uv run python scripts/run_enki_ingest.py \
	document=causalflows \
	entity_extractor/llm=ollama \
	entity_extractor/allowed_domains=machine_learning \
	llm=ollama embedder=ollama \
	fact_extractor/preferred_relation_types=none

enki-ingest-bio-simple:
	HYDRA_FULL_ERROR=1 uv run python scripts/run_enki_ingest.py \
	document=596284v1 \
	entity_extractor/allowed_domains=biomedical_simple \
	embedder=ollama \
	fact_extractor/preferred_relation_types=none

enki-ingest-agro:
	HYDRA_FULL_ERROR=1 uv run python scripts/run_enki_ingest.py kg=agro document=maraichere_paris \
	entity_extractor/llm=ollama \
	entity_extractor/allowed_domains=agricultural_simple \
	llm=ollama embedder=ollama \
	fact_extractor/llm=ollama \
	fact_extractor/preferred_relation_types=agriculture

MODEL_NAME = qwen3:14b
MODEL_BASE_URL = http://localhost:11434
EMBEDDER_NAME = "mxbai-embed-large:335m"
EMBEDDER_DIM = 1024
EMBEDDER_BASE_URL = http://localhost:11434


element-ingest-nodes:
	uv run python scripts/run_element_resolver.py ingest \
		--data-uri s3://sai-spaice-ds/data/processed/data_processing/pubtator3-v0.4.0/nodes_corrected.tsv \
		--table-name pubtator3-v0_4_0-nodes_vectors \
		--content-columns node_name,node_id_raw,node_domain,node_domain_id,node_id \
		--content-template "{{ node_name }}{% if node_id_raw %} | {{ node_id_raw }}{% endif %} || {{ node_domain }}" \
		--model ${MODEL_NAME} \
		--model-temperature 0.2 \
		--model-base-url ${MODEL_BASE_URL} \
		--embedder ${EMBEDDER_NAME} \
		--embedding-dim ${EMBEDDER_DIM} \
		--embedder-base-url ${EMBEDDER_BASE_URL} \
		--batch-size 256 \
		--no-description
	
element-ingest-nodes-desc:
	uv run python scripts/run_element_resolver.py ingest \
		--data-uri s3://sai-spaice-ds/data/processed/data_processing/pubtator3-v0.4.0/nodes_corrected.tsv \
		--table-name pubtator3-v0_4_0-nodes_vectors-desc \
		--content-columns node_name,node_id_raw,node_domain,node_domain_id,node_id \
		--content-template "{{ node_name }}{% if node_id_raw %} | {{ node_id_raw }}{% endif %} || {{ node_domain }}" \
		--model ${MODEL_NAME} \
		--model-temperature 0.2 \
		--model-base-url ${MODEL_BASE_URL} \
		--embedder ${EMBEDDER_NAME} \
		--embedder-base-url ${EMBEDDER_BASE_URL} \
		--batch-size 128

element-ingest-relations:
	uv run python scripts/run_element_resolver.py ingest \
		--data-uri s3://sai-spaice-ds/data/processed/data_processing/pubtator3-v0.4.0/edges.tsv \
		--table-name pubtator3-v0_4_0-relations_vectors \
		--content-columns relation_type \
		--content-template "{{ relation_type }}" \
		--model ${MODEL_NAME} \
		--model-temperature 0.2 \
		--model-base-url ${MODEL_BASE_URL} \
		--embedder ${EMBEDDER_NAME} \
		--embedder-base-url ${EMBEDDER_BASE_URL} \
		--batch-size 256 \
		--no-description \
		--unique


element-api:
	set -a && source .env.local && set +a && uv run python scripts/run_element_resolver_api.py