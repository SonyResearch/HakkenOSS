import gc

import lightning as pl
import mlflow
import torch
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from loguru import logger
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from hakken_models.callbacks import (
    CUDAMemoryMaintenanceCallback,
    MLFlowLoggerV2,
    TrainingLoopTimingCallback,
)
from hakken_models.core.configs.early_stopping import EarlyStoppingConfig
from hakken_models.core.configs.hpo import TuneReporterConfig
from hakken_models.core.configs.train_common import (
    LoggerConfig,
    ModelCheckpointConfig,
    ResumeCheckpointConfig,
    TrainerConfig,
)
from hakken_models.core.entities.trainer_bundle import TrainerBundle
from hakken_models.core.utils.model_checkpoint import resolve_checkpoint_path


def set_random_seed(seed: int | None, workers: bool = True) -> None:
    if seed is not None:
        seed_everything(seed, workers=workers)
        logger.info(f"Set random seed to {seed} for reproducibility")
    else:
        logger.warning("No seed set - results may not be reproducible")


def build_trainer(
    trainer_config: TrainerConfig,
    logger_config: LoggerConfig | None,
    checkpoint_config: ModelCheckpointConfig | None,
    resume_config: ResumeCheckpointConfig | None,
    early_config: EarlyStoppingConfig | None,
    tune_config: TuneReporterConfig | None,
) -> TrainerBundle:
    callbacks: list[Callback] = []

    mlflow_logger = None
    if logger_config is not None:
        run = mlflow.active_run()
        tracking_uri = mlflow.get_tracking_uri()
        mlflow_logger = MLFlowLoggerV2(
            tracking_uri=tracking_uri,
            run_id=run.info.run_id if run else None,
            **logger_config.model_dump(),
        )

    if tune_config is not None:
        logger.info("Using TuneReportCheckpointCallback!")
        tune_report = TuneReportCheckpointCallback(**tune_config.model_dump())
        callbacks.append(tune_report)

    model_checkpoint: ModelCheckpoint | None = None

    if checkpoint_config is not None:
        logger.info("Using ModelCheckpoint!")
        model_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_config.checkpoint_dir,
            filename=checkpoint_config.filename,
            monitor=checkpoint_config.monitor,
            mode=checkpoint_config.mode,
            save_top_k=checkpoint_config.save_top_k,
            save_last=checkpoint_config.save_last,
            every_n_epochs=checkpoint_config.every_n_epochs,
        )
        callbacks.append(model_checkpoint)

    if early_config is not None:
        logger.info("Using EarlyStopping!")
        early_stopping = EarlyStopping(**early_config.model_dump())
        callbacks.append(early_stopping)

    if torch.cuda.is_available():
        callbacks.append(CUDAMemoryMaintenanceCallback())
    callbacks.append(TrainingLoopTimingCallback())

    trainer = pl.Trainer(
        logger=mlflow_logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=1,
        use_distributed_sampler=False,
        **trainer_config.model_dump(exclude={"auto_batch_size"}),
    )

    local_ckpt_path = None
    if resume_config is not None:
        logger.info(f"Resume from checkpoint URI: {resume_config.uri}")
        local_ckpt_path = resolve_checkpoint_path(resume_config)
        logger.info(f"Resolved local checkpoint path: {local_ckpt_path}")

    return TrainerBundle(
        trainer=trainer,
        mlflow_logger=mlflow_logger,
        model_checkpoint=model_checkpoint,
        ckpt_path=local_ckpt_path,
    )


def find_max_batch_size(
    module: LightningModule,
    data_module: LightningDataModule,
    trainer_config: dict,
    starting_batch_size: int = 32,
    max_batch_size: int = 16384,
):
    if starting_batch_size > max_batch_size:
        return max_batch_size

    batch_size = starting_batch_size

    test_trainer_config = trainer_config.copy()
    test_trainer_config.pop("auto_batch_size", None)

    test_trainer_config["limit_train_batches"] = 2
    test_trainer_config["limit_val_batches"] = 2
    test_trainer_config["max_epochs"] = 1
    test_trainer_config["enable_checkpointing"] = False
    test_trainer_config["devices"] = 1
    test_trainer_config["strategy"] = "auto"
    test_trainer_config["use_distributed_sampler"] = False

    optimal_batch_size = batch_size
    should_continue = True
    while should_continue:
        logger.info(f"Trying batch_size={batch_size} (max={max_batch_size})")

        try:
            data_module.batch_size = batch_size

            trainer = Trainer(**test_trainer_config)
            trainer.fit(module, datamodule=data_module)
            if batch_size >= max_batch_size:
                logger.info(f"Reached the configured max ({max_batch_size}). Using {batch_size}.")
                optimal_batch_size = batch_size
                should_continue = False

            # If successful, try larger batch size
            batch_size *= 2

        except torch.OutOfMemoryError:
            logger.info(f"Batch size {batch_size} failed. Backing off to {batch_size // 2}.")
            optimal_batch_size = batch_size // 2
            should_continue = False
        finally:
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    return optimal_batch_size
