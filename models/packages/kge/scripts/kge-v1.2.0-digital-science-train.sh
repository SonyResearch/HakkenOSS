
export DATASET=digital_science
export DATA_VERSION=${DATASET}-v2.0.0
export KGE_VERSION=1.2.0

export DATA_ROOT_FOLDER=/home/pablo.sanchez2/Documents/GitHub/data/hakken_bio/${DATA_VERSION}



HYDRA_FULL_ERROR=1 uv run python kge/delivery/cli/train.py \
    optimizer.kwargs.lr=0.01 \
    lr_scheduler.kwargs.factor=0.5 \
    loss_fn.config.margin=120 \
    model.config.embedding_dim=128 \
    negative_sampler.config.num_negatives=100 \
    batch_size_optimization=default \
    data_processor.config.loader.batch_size=4096 \
    data_repo=${DATASET} \
    evaluator=mrr \
    evaluator.config.enable=False \
	hydra.run.dir=outputs/prod/kge-v${KGE_VERSION} \
    log_level=DEBUG \
    model=complex \
    run.project=kge-prod-${DATA_VERSION} \
    run.save_artifacts=True \
    seed_list=[0] \
    trainer.max_epochs=40 \
    trainer.limit_train_batches=0.2 \
    trainer.limit_val_batches=1.0