

checks:
	uv run ruff format
	uv run ruff check . --fix
	uv run mypy
	uv run pytest
	

DATASET_NAME=pubtator3-v0_4_0
DATASET_VERSION=v2
prepare-dataset: 
	uv run python scripts/run_pipeline.py prepare-dataset \
		--override "dataset_preparator=$(DATASET_NAME)-$(DATASET_VERSION)"

train-kge:
	uv run python scripts/run_pipeline.py train-kge --override "val_metric_hub=kge_v1 \
	loss=kge_ranking_focal"

# Usage: make train-kge-from-run RUN_ID=<mlflow-run-uuid> [MLFLOW_TRACKING_URI=http://127.0.0.1:5000]
train-kge-from-run:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "ERROR: Set RUN_ID: make train-kge-from-run RUN_ID=<mlflow-run-uuid>"; \
		exit 1; \
	fi; \
	if [ -n "$(MLFLOW_TRACKING_URI)" ]; then \
		uv run python scripts/run_pipeline.py train-kge --from-mlflow-run "$(RUN_ID)" --mlflow-tracking-uri "$(MLFLOW_TRACKING_URI)"; \
	else \
		uv run python scripts/run_pipeline.py train-kge --from-mlflow-run "$(RUN_ID)"; \
	fi

train-thiger:
	uv run python scripts/run_pipeline.py train-thiger


# Context graphs: reverse edges per fact + inv_emb on reverse direction (segal.use_inverse_relations).
train-segal:
	CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/run_pipeline.py train-segal --override "val_metric_hub=kge_v1 \
	segal.use_inverse_relations=true \
	learn_embeddings=true \
	embeddings_random_init=true \
	trainer.limit_train_batches=0.05 \
	trainer.limit_val_batches=0.05 \
	optimizer.kwargs.lr=5e-4 \
	optimizer.kwargs.weight_decay=1e-5 \
	scheduler=reduce_lr_on_plateau \
	loss=segal_composite_focal"

# Usage: make train-segal-from-run RUN_ID=<mlflow-run-uuid> [MLFLOW_TRACKING_URI=http://127.0.0.1:5000]
train-segal-from-run:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "ERROR: Set RUN_ID: make train-segal-from-run RUN_ID=<mlflow-run-uuid>"; \
		exit 1; \
	fi; \
	if [ -n "$(MLFLOW_TRACKING_URI)" ]; then \
		CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/run_pipeline.py train-segal --from-mlflow-run "$(RUN_ID)" --mlflow-tracking-uri "$(MLFLOW_TRACKING_URI)"; \
	else \
		CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/run_pipeline.py train-segal --from-mlflow-run "$(RUN_ID)"; \
	fi

hpo-kge:
	uv run python scripts/run_pipeline.py hpo-kge \
	--override "hpo=default \
	val_metric_hub=kge_v1 \
	model_checkpoint.enabled=False \
	logger.log_model=False \
	experiment_tracker.experiment_name=kge-hpo" \
	--config-dir /home/pablo.sanchez2/GitHub/project_spaice_ds/packages/pip/hakken-models/configs


evaluate-thiger:
	uv run python scripts/run_pipeline.py evaluate-thiger --override "model_loader=thiger_mlflow"

evaluate-segal:
	MLFLOW_RUN_ID=2d40ff655a9443ba99dbe0975357fa23 uv run python scripts/run_pipeline.py evaluate-segal \
	--override "model_loader.tracking_uri=file://$(pwd)/mlruns" \
	--override "model_loader.relative_ckpt_path=last-v4.ckpt"


zenml-ui:
	uv run zenml login --local

zenml-login:
	uv run zenml login --local

list-artifacts:
	uv run zenml artifact list

prune-artifacts:
	uv run zenml artifact prune

mlflow-delete-runs:
	@if [ -z "$(EXP)" ]; then \
		echo "ERROR: Please provide an experiment name: make mlflow-delete-runs EXP=<name>"; \
		exit 1; \
	fi; \
	EXP_ID=$$(uv run mlflow experiments search | awk -v name="$(EXP)" '$$2==name {print $$1}'); \
	if [ -z "$$EXP_ID" ]; then \
		echo "ERROR: Experiment \"$(EXP)\" not found."; \
		exit 1; \
	fi; \
	echo "Cleaning runs in experiment $(EXP) (ID: $$EXP_ID)"; \
	find mlruns/$$EXP_ID -mindepth 1 ! -name meta.yaml -exec rm -rf {} +
	
mlflow-ui:
	uv run mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000

mlflow-start:
	uv run mlflow server \
	--backend-store-uri ./mlruns \
	--default-artifact-root ./mlruns \
	--host 127.0.0.1 \
	--port 5000
