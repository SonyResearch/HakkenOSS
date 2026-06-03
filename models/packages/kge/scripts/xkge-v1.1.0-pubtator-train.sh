
export DATASET=pubtator3
export DATA_VERSION=${DATASET}-v0.4.0
export KGE_VERSION=1.1.0
# export MODEL=rgcnkge
export MODEL=sagekge


export DATA_ROOT_FOLDER=/home/pablo.sanchez2/Documents/GitHub/data/hakken_bio/${DATA_VERSION}



HYDRA_FULL_ERROR=1 uv run dotenv run python kge/delivery/cli/train_mimic_kge.py \
	batch_size_optimization=none \
	data_processor=mimic_kge \
	data_repo=${DATASET} \
	evaluator=mrr \
	evaluator.config.enable=False \
	hydra.run.dir=outputs/prod/xkge-v${KGE_VERSION}/$(date +\%Y\%m\%d-\%H\%M\%S) \
	log_level=DEBUG \
	model=${MODEL} \
	run.project=xkge-prod \
	run.save_artifacts=True \
	seed_list="[0]" \
	trained_kge=${DATASET} \
	trainer.max_epochs=100 \
	trainer.limit_train_batches=0.1 \
	trainer.limit_val_batches=1.0 \
	trainer.gradient_clip_val=50.0
