"""
Checkpoint path resolution utilities for resuming training.
"""

import os
import shutil
from pathlib import Path

import mlflow
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger

from hakken_models.core.configs.train_common import ModelCheckpointConfig, ResumeCheckpointConfig


def maybe_clean_checkpoints(model_checkpoint: ModelCheckpoint) -> None:
    best_ckpt = model_checkpoint.best_model_path
    if best_ckpt and Path(best_ckpt).exists():
        shutil.rmtree(Path(best_ckpt).parent, ignore_errors=True)  # Or specific file
        logger.info("Cleaned best checkpoint")

    last_ckpt = model_checkpoint.last_model_path

    if last_ckpt and Path(last_ckpt).exists():
        shutil.rmtree(Path(last_ckpt).parent, ignore_errors=True)
        logger.info("Cleaned last checkpoint")


def resolve_checkpoint_path(config: ResumeCheckpointConfig) -> str | None:
    if config.uri_type == "mlflow":
        return mlflow.artifacts.download_artifacts(
            artifact_uri=config.uri,
            dst_path=config.local_dir,
        )
    if config.uri_type == "local":
        if os.path.exists(config.uri):
            return os.path.abspath(config.uri)
        logger.warning(f"Checkpoint path not found: {config.uri}")
        raise FileNotFoundError(f"Checkpoint file not found: {config.uri}. Please verify the path.")
    logger.warning(f"Unknown URI type: {config.uri_type}")
    raise ValueError(f"Unknown URI type: {config.uri_type}")


def maybe_get_checkpoint_path(
    checkpoint_config: ModelCheckpointConfig | None, model_checkpoint: ModelCheckpoint | None
) -> str | None:
    ckpt_file: str | None = None
    if checkpoint_config is not None:
        best_ckpt = model_checkpoint.best_model_path
        if best_ckpt is not None and len(best_ckpt) > 0:
            ckpt_file = best_ckpt
        else:
            ckpt_file = model_checkpoint.last_model_path

    if ckpt_file is not None:
        logger.info(f"Output checkpoint path: {ckpt_file}")

    return ckpt_file
