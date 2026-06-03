

export DATASET=pubtator3
export DATA_VERSION=0.4.0
export EXPLAINER_VERSION=1.1.0


HYDRA_FULL_ERROR=1 NX_CUGRAPH_AUTOCONFIG=True uv run python scripts/run_explainer.py  \
	triple_to_probe=${DATASET}-v${DATA_VERSION}