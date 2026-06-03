

.PHONY: checks ingest-temporal-kg


checks:
	uv run ruff format
	uv run ruff check . --fix
	uv run pytest
	uv run mypy

ingest-temporal-kg:
	uv run python -m scripts.temporal_kg_ingest in_memory --recreate

query-temporal-kg:
	uv run python -m scripts.temporal_kg_query in_memory

