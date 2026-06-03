export DATASET=digital_science
export DATA_VERSION=${DATASET}-v2.0.0
export KGE_VERSION=1.2.0

export DATA_ROOT_FOLDER=/home/pablo.sanchez2/Documents/GitHub/data/hakken_bio/${DATA_VERSION}


HYDRA_FULL_ERROR=1 uv run python kge/delivery/cli/inference.py \
	seed=0 \
	model_id=last \
	experiment_folder=outputs/prod/kge-v${KGE_VERSION} \
	task=evaluate \
	query=${DATASET}