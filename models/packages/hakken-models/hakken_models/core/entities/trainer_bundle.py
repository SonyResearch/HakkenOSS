from typing import NamedTuple

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger


class TrainerBundle(NamedTuple):
    trainer: Trainer
    mlflow_logger: MLFlowLogger
    model_checkpoint: ModelCheckpoint | None
    ckpt_path: str | None
