DATASET ?= pubtator3-v0.4.0

.PHONY: explain run_checks

explain:
	HYDRA_FULL_ERROR=1 NX_CUGRAPH_AUTOCONFIG=True uv run python scripts/run_explainer.py  \
	triple_to_probe=$(DATASET)

benchmark:
	HYDRA_FULL_ERROR=1 NX_CUGRAPH_AUTOCONFIG=True uv run python scripts/run_benchmark.py  \
	triple_to_probe=$(DATASET) \
	output.path=time_benchmark.tsv


path_finder:
	NX_CUGRAPH_AUTOCONFIG=True uv run python misc/run_path_finder.py

run_checks:
	uv run ruff format
	uv run ruff check . --fix
	uv run pytest
	uv run mypy

docs-serve:
	uv run --extra docs mkdocs serve

docs-build:
	uv run --extra docs mkdocs build

