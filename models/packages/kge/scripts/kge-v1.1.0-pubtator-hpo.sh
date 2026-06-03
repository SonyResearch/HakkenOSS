
export DATASET=pubtator3
export DATA_VERSION=${DATASET}-v0.4.0
export KGE_VERSION=1.1.0
export MODEL=complex

export DATA_ROOT_FOLDER=/home/pablo.sanchez2/Documents/GitHub/data/hakken_bio/${DATA_VERSION}


HYDRA_FULL_ERROR=1 uv run python kge/delivery/cli/train.py --multirun \
	batch_size_optimization=default \
	data_processor.config.loader.batch_size=2048  \
	data_repo=${DATASET} \
	evaluator=mrr \
	evaluator.config.enable=False \
	hydra=${MODEL} \
	log_level=INFO \
	model=${MODEL} \
	run.project=kge-v${KGE_VERSION}-${DATA_VERSION}-hpo \
	run.save_artifacts=false \
	seed_list="[0]" \
	trainer.max_epochs=10 \
	trainer.limit_train_batches=0.2 \
	trainer.limit_val_batches=1.0