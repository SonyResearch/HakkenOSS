from loguru import logger
from zenml import pipeline

from hakken_models.core.configs.train_kge import TrainKGEConfig
from hakken_models.core.configs.zenml import ContainerSettings, KubernetesKind, OrchestratorSettings
from hakken_models.steps.dataset import extract_dataset_metadata_step, load_dataset_deployment_step
from hakken_models.steps.kge.load_data_module import load_datamodule_step
from hakken_models.steps.kge.load_negative_sampler import load_negative_sampler_step
from hakken_models.steps.kge.train_kge import train_kge_with_lightning
from hakken_models.steps.tracking import log_hyperparameters_step


@pipeline(
    name="train_kge",
    enable_cache=False,
    settings={
        "orchestrator": OrchestratorSettings.kubernetes(KubernetesKind.IN_CLUSTER),
        "docker": ContainerSettings.docker(),
    },
)
def train_kge_pipeline(config: TrainKGEConfig) -> float:
    dataset = load_dataset_deployment_step(config.dataset)

    dataset_metadata = extract_dataset_metadata_step(dataset=dataset)

    logger.info(f"dataset_metadata: {dataset_metadata}")

    negative_sampler = load_negative_sampler_step(
        dataset=dataset, neg_sampler_config=config.negative_sampler
    )

    data_module = load_datamodule_step(
        dataset=dataset,
        negative_sampler=negative_sampler,
        data_loader_kwargs=config.data_loader.kwargs.copy(),
    )

    tune_reporter = None
    if config.hpo is not None:
        tune_reporter = config.hpo.reporter

    model_checkpoint = None
    if config.model_checkpoint.enabled:
        model_checkpoint = config.model_checkpoint

    log_hyperparameters_step(hparams=config.model_dump())

    _mlflow_run_id, _best_ckpt_filename, obj_metric = train_kge_with_lightning(
        data_module=data_module,
        dataset_metadata=dataset_metadata,
        init_strategy=config.init_strategy,
        kge_config=config.kge,
        loss_config=config.loss,
        negative_strategy=config.negative_strategy,
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
    )

    return obj_metric
