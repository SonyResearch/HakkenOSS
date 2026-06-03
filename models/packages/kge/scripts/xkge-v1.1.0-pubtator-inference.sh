
export DATASET=pubtator3
export DATA_VERSION=${DATASET}-v0.4.0
export XKGE_VERSION=1.1.0
export MODEL=sagekge


export DATA_ROOT_FOLDER=/home/pablo.sanchez2/Documents/GitHub/data/hakken_bio/${DATA_VERSION}



HYDRA_FULL_ERROR=1 uv run dotenv run python kge/delivery/cli/inference.py \
	evaluator=mimic_kge \
	model_is_gnn=True \
	seed=0 \
	model_id=last \
	experiment_folder=outputs/prod/xkge-v${XKGE_VERSION} \
	task=evaluate_mimic_kge \
	query=${DATASET}
