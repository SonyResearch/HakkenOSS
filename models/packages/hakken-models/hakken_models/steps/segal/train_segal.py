import os

import torch
from loguru import logger
from torch import Tensor
from zenml import log_metadata, step
from zenml.client import Client

from hakken_models.core.configs.early_stopping import EarlyStoppingConfig
from hakken_models.core.configs.hpo import TuneReporterConfig
from hakken_models.core.configs.train_common import (
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
from hakken_models.datasets.deployment import DatasetDeployment
from hakken_models.evaluators.metric_hub import MetricHubConfig
from hakken_models.models.kge.lightning import build_lit_kge_val_metric_hub
from hakken_models.models.segal.base import SeGAL
from hakken_models.models.segal.config import SeGALConfig
from hakken_models.models.segal.data_module import SeGALDataModule
from hakken_models.models.segal.lightning import create_lit_segal

experiment_tracker = Client().active_stack.experiment_tracker


def _build_embedding_matrices(
    dataset: DatasetDeployment,
    *,
    segal_config: SeGALConfig,
    learn_embeddings: bool,
    embeddings_random_init: bool = False,
) -> tuple[Tensor, Tensor]:
    """Load or synthesize node/relation tables for SeGAL (see ``learn_embeddings`` semantics)."""
    enc = segal_config.encoder_dim
    if learn_embeddings:
        if embeddings_random_init or not dataset.has_embeddings:
            std = enc**-0.5
            return (
                torch.randn(dataset.num_entities, enc) * std,
                torch.randn(dataset.num_relations, enc) * std,
            )
        node_e = dataset.get_node_embedding_matrix()
        rel_e = dataset.get_relation_embedding_matrix()
        if node_e.shape[1] != enc or rel_e.shape[1] != enc:
            raise ValueError(
                f"learn_embeddings requires on-disk embeddings with last dim == encoder_dim ({enc}); "
                f"got node {node_e.shape[1]}, rel {rel_e.shape[1]}"
            )
        return node_e, rel_e
    if not dataset.has_embeddings:
        raise RuntimeError(
            "SeGAL requires pre-computed embeddings. "
            "Place nodes.npy and relations.npy under "
            f"{dataset.target_root}/embeddings/"
        )
    return dataset.get_node_embedding_matrix(), dataset.get_relation_embedding_matrix()


@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
    settings={
        "orchestrator": OrchestratorSettings.kubernetes(KubernetesKind.GPU),
    },
)
def train_segal_with_lightning(
    dataset: DatasetDeployment,
    data_module: SeGALDataModule,
    segal_config: SeGALConfig,
    loss_config: LossConfig,
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
    learn_embeddings: bool = False,
    embedding_lr_factor: float = 0.1,
    embeddings_random_init: bool = False,
) -> tuple[str, str | None, float]:
    """Train a SeGAL model end-to-end with Lightning."""

    # Reduce CUDA allocator fragmentation (OOM after many epochs when reserved
    # memory is fragmented and no contiguous block is free).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    set_random_seed(run_config.seed, workers=True)

    segal_cfg = segal_config.model_copy(update={"learn_embeddings": learn_embeddings})
    node_embeddings, relation_embeddings = _build_embedding_matrices(
        dataset,
        segal_config=segal_cfg,
        learn_embeddings=learn_embeddings,
        embeddings_random_init=embeddings_random_init,
    )

    segal = SeGAL(config=segal_cfg)

    # Set temporal encoder normalization from training timestamps.
    # When resuming from checkpoint, Lightning overwrites this with saved state.
    ts = data_module.train_kg_data.edge_attr[:, 1].float()
    t_mean = ts.mean().item()
    t_std = max(ts.std().item(), 1e-6)
    segal.temporal_encoder.set_normalization(t_mean, t_std)
    logger.info(f"Temporal encoder normalization: mean={t_mean:.2f}, std={t_std:.2f}")

    lit_module = create_lit_segal(
        segal=segal,
        node_embeddings=node_embeddings,
        relation_embeddings=relation_embeddings,
        loss_config=loss_config.model_dump(),
        optimizer_config=optimizer_config.model_dump(),
        scheduler_config=(scheduler_config.model_dump() if scheduler_config is not None else None),
        dataset=dataset,
        val_metric_hub=build_lit_kge_val_metric_hub(
            val_metric_hub,
            num_relations=relation_embeddings.shape[0],
        ),
        learn_embeddings=learn_embeddings,
        embedding_lr_factor=embedding_lr_factor,
    )

    optimal_batch_size = find_max_batch_size(
        module=lit_module,
        data_module=data_module,
        trainer_config=trainer_config.model_dump(),
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

    trainer.fit(
        lit_module,
        datamodule=data_module,
        ckpt_path=trainer_bundle.ckpt_path,
    )

    if run_config.cleanup_local_checkpoints and trainer_bundle.model_checkpoint is not None:
        maybe_clean_checkpoints(trainer_bundle.model_checkpoint)

    run_id = mlflow_logger.run_id or ""
    experiment_id = mlflow_logger.experiment_id or ""
    name = mlflow_logger.name or ""

    logger.info(
        f"Trained SeGAL model with MLFlow run ID: {run_id}"
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
        checkpoint_config=checkpoint_config,
        model_checkpoint=trainer_bundle.model_checkpoint,
    )

    obj_metric = trainer.callback_metrics["val/mean_rank"].item()

    return run_id, ckpt_file, obj_metric
