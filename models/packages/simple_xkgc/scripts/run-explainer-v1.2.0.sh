

export DATASET=digital_science
export DATA_VERSION=2.0.0
export EXPLAINER_VERSION=1.2.0


HYDRA_FULL_ERROR=1 NX_CUGRAPH_AUTOCONFIG=True uv run python scripts/run_path_explainer.py  \
	triple_to_probe=${DATASET}-v${DATA_VERSION}