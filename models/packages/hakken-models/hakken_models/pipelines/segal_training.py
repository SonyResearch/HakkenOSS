from loguru import logger
from zenml import pipeline

from hakken_models.core.configs.train_segal import TrainSeGALConfig
from hakken_models.core.configs.zenml import ContainerSettings, KubernetesKind, OrchestratorSettings
from hakken_models.steps.dataset import extract_dataset_metadata_step, load_dataset_deployment_step
from hakken_models.steps.segal.load_data_module import load_segal_datamodule_step
from hakken_models.steps.segal.train_segal import train_segal_with_lightning
from hakken_models.steps.tracking import log_hyperparameters_step


@pipeline(
    name="train_segal",
    enable_cache=False,
    settings={
        "orchestrator": OrchestratorSettings.kubernetes(KubernetesKind.IN_CLUSTER),
        "docker": ContainerSettings.docker(),
    },
)
def train_segal_pipeline(config: TrainSeGALConfig) -> float:
    dataset = load_dataset_deployment_step(config.dataset)

    dataset_metadata = extract_dataset_metadata_step(dataset=dataset)

    logger.info(f"dataset_metadata: {dataset_metadata}")

    data_module = load_segal_datamodule_step(
        dataset=dataset,
        data_loader_kwargs=config.data_loader.kwargs.copy(),
        num_negatives=config.num_negatives,
        num_negatives_val=config.num_negatives_val,
        add_reverse_edges=config.segal.use_inverse_relations,
    )

    tune_reporter = None
    if config.hpo is not None:
        tune_reporter = config.hpo.reporter

    model_checkpoint = None
    if config.model_checkpoint.enabled:
        model_checkpoint = config.model_checkpoint

    log_hyperparameters_step(hparams=config.model_dump())

    _mlflow_run_id, _best_ckpt_filename, obj_metric = train_segal_with_lightning(
        dataset=dataset,
        data_module=data_module,
        segal_config=config.segal,
        loss_config=config.loss,
        optimizer_config=config.optimizer,
        scheduler_config=config.scheduler,
        logger_config=config.logger,
        resume_config=config.resume,
        trainer_config=config.trainer,
        checkpoint_config=model_checkpoint,
        early_config=config.early_stopping,
        tune_config=tune_reporter,
        run_config=config.run,
        val_metric_hub=config.val_metric_hub,
        learn_embeddings=config.learn_embeddings,
        embedding_lr_factor=config.embedding_lr_factor,
        embeddings_random_init=config.embeddings_random_init,
    )

    return obj_metric
