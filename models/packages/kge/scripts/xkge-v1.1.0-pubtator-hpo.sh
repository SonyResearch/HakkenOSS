
export DATASET=pubtator3
export DATA_VERSION=${DATASET}-v0.4.0
export KGE_VERSION=1.1.0
export MODEL=rgcnkge

export DATA_ROOT_FOLDER=/home/pablo.sanchez2/Documents/GitHub/data/hakken_bio/${DATA_VERSION}


HYDRA_FULL_ERROR=1 uv run python kge/delivery/cli/train_gnntkge.py --multirun \
	batch_size_optimization=small \
	data_processor.config.loader.batch_size=2048  \
	data_repo=${DATASET} \
	evaluator=mrr \
	evaluator.config.enable=False \
	hydra=${MODEL} \
	log_level=INFO \
	loss_fn=mse \
	model=${MODEL} \
	run.project=xkge-v${KGE_VERSION}-${DATA_VERSION}-hpo \
	run.save_artifacts=False \
	seed_list="[0]" \
	trained_kge=${DATASET} \
	trainer.max_epochs=10 \
	trainer.limit_train_batches=0.1 \
	trainer.limit_val_batches=1.0