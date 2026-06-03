from __future__ import annotations

from pydantic import BaseModel, Field
from pytorch_lightning.callbacks import EarlyStopping as LightningEarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


class EarlyStoppingConfig(BaseModel):
    patience: int = Field(default=3, description="Number of epochs with no improvement")
    min_delta: float = Field(default=0.0, description="Minimum change to qualify as improvement")
    mode: str = Field(
        default="min",
        description="'min' for reducing loss, 'max' for improving accuracy",
    )
    strict: bool = Field(default=True, description="Raise error if 'monitor' is not found")
    verbose: bool = Field(default=True)
    monitor: str = Field(default="validation/loss")


class EarlyStopping:
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.early_stopping_callback: LightningEarlyStopping
        self.model_checkpoint_callback: ModelCheckpoint

    def set_up(self) -> None:
        self.early_stopping_callback = LightningEarlyStopping(**self.config.model_dump())
        self.model_checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            monitor=self.config.monitor,
            mode=self.config.mode,
        )
