from dataclasses import dataclass

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint


@dataclass
class TrainerCallbacks:
    early_stopping: EarlyStopping
    model_checkpoint: ModelCheckpoint | None = None
    pruning: PyTorchLightningPruningCallback | None = None

    def to_list(self) -> list[Callback]:
        callbacks: list[Callback] = [self.early_stopping]
        if self.model_checkpoint is not None:
            callbacks.append(self.model_checkpoint)

        return callbacks
