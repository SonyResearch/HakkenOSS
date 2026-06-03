# Default parameters
DATASET ?= digital_science
LIMIT_TRAIN_BATCHES ?= 1.0
LIMIT_VAL_BATCHES ?= 1.0
MODEL ?= complex

KGE_OUTPUT_DIR ?= outputs/$(DATASET)/$(MODEL)/$$(date +"%Y-%m-%d/%H-%M-%S")
GNN_OUTPUT_DIR ?= outputs/$(DATASET)/$(MODEL)_tkge/$$(date +"%Y-%m-%d/%H-%M-%S")
PROJECT_HPO ?= $(DATASET)_$(MODEL)_hpo
run_hpo:
	HYDRA_FULL_ERROR=1 uv run python kge/delivery/cli/train.py --multirun \
	batch_size_optimization=default \
	data_processor.config.loader.batch_size=2048  \
	data_repo=$(DATASET) \
	evaluator=mrr \
	evaluator.config.enable=False \
	hydra=$(MODEL) \
	log_level=INFO \
	model=$(MODEL) \
	run.project=$(PROJECT_HPO) \
	run.save_artifacts=false \
	seed_list="[0]" \
	trainer.max_epochs=10 \
	trainer.limit_train_batches=0.2 \
	trainer.limit_val_batches=1.0

train:
	HYDRA_FULL_ERROR=1 uv run python kge/delivery/cli/train.py \
	batch_size_optimization=default \
	data_processor.config.loader.batch_size=8192 \
	data_repo=$(DATASET) \
	early_stopping=none \
	evaluator=mrr \
	evaluator.config.enable=False \
	hydra.run.dir=$(KGE_OUTPUT_DIR) \
	log_level=DEBUG \
	loss_fn=margin_ranking \
    loss_fn.config.margin=20 \
	lr_scheduler.kwargs.factor=0.5 \
	model=$(MODEL) \
	model.config.embedding_dim=512 \
	negative_sampler.config.num_negatives=50 \
	optimizer.kwargs.lr=0.04 \
    optimizer.kwargs.weight_decay=4.6e-06 \
	run.project=$(DATASET) \
	run.save_artifacts=true \
	seed_list=[0] \
	trainer.max_epochs=40 \
	trainer.limit_train_batches=0.2 \
	trainer.limit_val_batches=0.2 \


train_mimic_kge:
	HYDRA_FULL_ERROR=1 uv run python kge/delivery/cli/train_mimic_kge.py \
	batch_size_optimization=default \
	data_processor.config.loader.batch_size=2048 \
	data_repo=$(DATASET) \
	evaluator=mrr \
	evaluator.config.enable=False \
	hydra.run.dir=$(GNN_OUTPUT_DIR) \
	log_level=DEBUG \
	loss_fn=mse \
	model=$(MODEL) \
	run.project=$(DATASET)_tkge \
	run.save_artifacts=true \
	seed_list=[0] \
	trained_kge=$(DATASET) \
	trainer.max_epochs=20 \
	trainer.limit_train_batches=0.1 \
	trainer.limit_val_batches=0.1 \


inference:
	HYDRA_FULL_ERROR=1 uv run python kge/delivery/cli/inference.py \
	seed=0 \
	model_id=last \
	experiment_folder=$(KGE_OUTPUT_DIR) \
	task=evaluate \
	query=$(DATASET)

latency-profiler:
	HYDRA_FULL_ERROR=1 uv run python kge/delivery/cli/inference.py \
	seed=0 \
	model_id=last \
	experiment_folder=/home/pablo.sanchez2/Documents/GitHub/project_spaice_ds/packages/pip/kge/outputs/prod/kge-v1.1.0/ \
	task=latency_profiler \
	query=pubtator3-v0.4.0

predict:
	HYDRA_FULL_ERROR=1 uv run python kge/delivery/cli/inference.py \
	seed=0 \
	model_id=last \
	experiment_folder=/home/pablo.sanchez2/Documents/GitHub/project_spaice_ds/packages/pip/kge/outputs/prod/kge-v1.1.0/ \
	task=predict \
	query=pubtator3-v0.4.0



run-checks:
	uv run ruff format
	uv run ruff check . --fix
	uv run pytest
	uv run mypy

run_mlflow:
	uv run mlflow ui --backend-store-uri .ml_flow

.PHONY: run_hpo train run_checks




