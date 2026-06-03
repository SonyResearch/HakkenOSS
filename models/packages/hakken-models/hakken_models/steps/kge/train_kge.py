from loguru import logger
from zenml import log_metadata, step
from zenml.client import Client

from hakken_models.core.configs.early_stopping import EarlyStoppingConfig
from hakken_models.core.configs.hpo import TuneReporterConfig
from hakken_models.core.configs.model import KGEConfig
from hakken_models.core.configs.negative_strategy import NegativeStrategyConfig
from hakken_models.core.configs.train_common import (
    InitStrategyConfig,
    LoggerConfig,
    LossConfig,
    ModelCheckpointConfig,
    OptimizerConfig,
    ResumeCheckpointConfig,
    RunConfig,
    SchedulerConfig,
    TrainerConfig,
)
from hakken_models.core.configs.zenml import KubernetesKind, OrchestratorSettings
from hakken_models.core.utils.model_checkpoint import (
    maybe_clean_checkpoints,
    maybe_get_checkpoint_path,
)
from hakken_models.core.utils.trainer_pipeline import (
    build_trainer,
    find_max_batch_size,
    set_random_seed,
)
from hakken_models.evaluators.metric_hub import MetricHubConfig
from hakken_models.initialization import init_strategy_registry
from hakken_models.models.kge import KGE
from hakken_models.models.kge.data_module import KGEDataModule
from hakken_models.models.kge.lightning import build_lit_kge_val_metric_hub, create_lit_kge
from hakken_models.scores import score_fn_registry

experiment_tracker = Client().active_stack.experiment_tracker


@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
    settings={
        "orchestrator": OrchestratorSettings.kubernetes(KubernetesKind.GPU),
    },
)
def train_kge_with_lightning(
    data_module: KGEDataModule,
    dataset_metadata: dict,
    init_strategy: InitStrategyConfig,
    kge_config: KGEConfig,
    loss_config: LossConfig,
    negative_strategy: NegativeStrategyConfig,
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig | None,
    trainer_config: TrainerConfig,
    logger_config: LoggerConfig,
    resume_config: ResumeCheckpointConfig | None,
    checkpoint_config: ModelCheckpointConfig | None,
    early_config: EarlyStoppingConfig | None,
    tune_config: TuneReporterConfig | None,
    run_config: RunConfig,
    val_metric_hub: MetricHubConfig,
) -> tuple[str, str | None, float]:
    set_random_seed(run_config.seed, workers=True)

    score_fn = score_fn_registry.create(kge_config.score_fn_name, **kge_config.score_fn_kwargs)

    kge = KGE(
        num_entities=dataset_metadata["num_entities"],
        num_relations=dataset_metadata["num_relations"],
        embedding_dim=kge_config.embedding_dim,
        score_fn=score_fn,
    )

    init_strategy_fn = init_strategy_registry.create(init_strategy.name, **init_strategy.kwargs)

    init_strategy_fn(kge)

    lit_module = create_lit_kge(
        kge=kge,
        loss_config=loss_config.with_kge_negative_strategy(negative_strategy),
        optimizer_config=optimizer_config.model_dump(),
        scheduler_config=scheduler_config.model_dump() if scheduler_config is not None else None,
        val_metric_hub=build_lit_kge_val_metric_hub(
            val_metric_hub,
            num_relations=dataset_metadata["num_relations"],
        ),
    )

    optimal_batch_size = find_max_batch_size(
        module=lit_module, data_module=data_module, trainer_config=trainer_config.model_dump()
    )

    logger.info(f"Optimal batch_size: {optimal_batch_size}")
    data_module.batch_size = optimal_batch_size

    trainer_bundle = build_trainer(
        trainer_config=trainer_config,
        logger_config=logger_config,
        checkpoint_config=checkpoint_config,
        resume_config=resume_config,
        early_config=early_config,
        tune_config=tune_config,
    )

    trainer = trainer_bundle.trainer
    mlflow_logger = trainer_bundle.mlflow_logger

    trainer.fit(lit_module, datamodule=data_module, ckpt_path=trainer_bundle.ckpt_path)

    if run_config.cleanup_local_checkpoints and trainer_bundle.model_checkpoint is not None:
        maybe_clean_checkpoints(trainer_bundle.model_checkpoint)

    run_id = mlflow_logger.run_id if mlflow_logger.run_id is not None else ""
    experiment_id = mlflow_logger.experiment_id if mlflow_logger.experiment_id is not None else ""
    name = mlflow_logger.name if mlflow_logger.name is not None else ""

    logger.info("Inside train_kge_with_lightning pipeline step")
    logger.info(
        f"Trained KGE model with MLFlow run ID: {run_id}"
        f", experiment ID: {experiment_id}, run name: {name}"
    )

    log_metadata(
        metadata={
            "mlflow_run_id": run_id,
            "mlflow_experiment_id": experiment_id,
            "mlflow_run_name": name,
        },
    )

    ckpt_file = maybe_get_checkpoint_path(
        checkpoint_config=checkpoint_config, model_checkpoint=trainer_bundle.model_checkpoint
    )

    obj_metric = trainer.callback_metrics["val/mean_rank"].item()

    return run_id, ckpt_file, obj_metric
