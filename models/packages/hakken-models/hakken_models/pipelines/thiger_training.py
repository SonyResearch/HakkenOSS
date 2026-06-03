from zenml import pipeline

from hakken_models.core.configs.train_thiger import TrainTHiGERConfig
from hakken_models.core.configs.zenml import ContainerSettings, KubernetesKind, OrchestratorSettings
from hakken_models.steps.dataset import (
    extract_dataset_metadata_step,
    load_dataset_deployment_step,
)
from hakken_models.steps.thiger import (
    load_datamodule_step,
    train_thiger_with_lightning_step,
)
from hakken_models.steps.tracking.log_hyperparameters import log_hyperparameters_step


@pipeline(
    name="train_thiger",
    enable_cache=True,
    settings={
        "orchestrator": OrchestratorSettings.kubernetes(KubernetesKind.IN_CLUSTER),
        "docker": ContainerSettings.docker(),
    },
)
def train_thiger_pipeline(config: TrainTHiGERConfig) -> str:
    dataset = load_dataset_deployment_step(config.dataset)

    dataset_metadata = extract_dataset_metadata_step(dataset=dataset)

    data_module = load_datamodule_step(
        dataset=dataset,
        data_loader_kwargs=config.data_loader.kwargs.copy(),
    )

    tune_reporter = None
    if config.hpo is not None:
        tune_reporter = config.hpo.reporter

    model_checkpoint = None
    if config.model_checkpoint.enabled:
        model_checkpoint = config.model_checkpoint

    log_hyperparameters_step(hparams=config.model_dump())

    _mlflow_run_id, _best_ckpt_filename, obj_metric = train_thiger_with_lightning_step(
        data_module=data_module,
        dataset_metadata=dataset_metadata,
        init_strategy=config.init_strategy,
        thiger_config=config.thiger,
        loss_config=config.loss,
        optimizer_config=config.optimizer,
        scheduler_config=config.scheduler,
        trainer_config=config.trainer,
        logger_config=config.logger,
        resume_config=config.resume,
        checkpoint_config=model_checkpoint,
        early_config=config.early_stopping,
        tune_config=tune_reporter,
        run_config=config.run,
    )

    return obj_metric
