SHELL := /bin/bash

# Override if needed: make predict PREDICT_URL=http://host:port/path
PREDICT_URL ?= http://localhost:8088/test/kge/predict

.PHONY: run_checks serve predict

run_checks:
	uv run ruff format .
	uv run ruff check --fix .
	uv run mypy .
	uv run pytest

serve:
	uv run python service.py

predict:
	curl -sS -X POST \
	 -H "Content-Type: application/json" \
	 -d '{"request":{"subject_id_list":["Chemical|MESH:C494910"],"object_id_list":["Gene|3553"],"relation_id_list":["negative_correlate"]}}' \
	 "$(PREDICT_URL)"
	 
