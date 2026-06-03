"""Callback to log training-loop wall-clock timings to MLflow (epoch, data wait, step)."""

from __future__ import annotations

import time

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback


class TrainingLoopTimingCallback(Callback):
    """Logs epoch duration, dataloader wait time, and step wall time per epoch.

    All timings are wall-clock (include data loading, transfer, compute).
    Metrics are logged to the trainer logger (e.g. MLflow) at epoch end
    with prefix ``timing/``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._train_epoch_start: float | None = None
        self._val_epoch_start: float | None = None
        self._last_step_end: float | None = None
        self._step_start: float | None = None
        self._train_data_wait_ms: list[float] = []
        self._train_step_ms: list[float] = []
        self._val_data_wait_ms: list[float] = []
        self._val_step_ms: list[float] = []

    def on_train_epoch_start(self, trainer: Trainer, *args, **kwargs) -> None:
        self._train_epoch_start = time.perf_counter()
        self._train_data_wait_ms = []
        self._train_step_ms = []
        self._last_step_end = None

    def on_train_batch_start(self, trainer: Trainer, *args, **kwargs) -> None:
        now = time.perf_counter()
        if self._last_step_end is not None:
            self._train_data_wait_ms.append((now - self._last_step_end) * 1000)
        self._step_start = now

    def on_train_batch_end(self, trainer: Trainer, *args, **kwargs) -> None:
        now = time.perf_counter()
        if self._step_start is not None:
            self._train_step_ms.append((now - self._step_start) * 1000)
        self._last_step_end = now

    def on_train_epoch_end(self, trainer: Trainer, *args, **kwargs) -> None:
        if self._train_epoch_start is None:
            return
        epoch_s = time.perf_counter() - self._train_epoch_start
        metrics = {"timing/train_epoch_s": epoch_s}
        if self._train_data_wait_ms:
            metrics["timing/train_data_wait_ms"] = sum(self._train_data_wait_ms) / len(
                self._train_data_wait_ms
            )
        if self._train_step_ms:
            metrics["timing/train_step_wall_ms"] = sum(self._train_step_ms) / len(
                self._train_step_ms
            )
        trainer.logger.log_metrics(metrics, step=trainer.global_step)
        self._train_epoch_start = None

    def on_validation_epoch_start(self, trainer: Trainer, *args, **kwargs) -> None:
        self._val_epoch_start = time.perf_counter()
        self._val_data_wait_ms = []
        self._val_step_ms = []
        self._last_step_end = None

    def on_validation_batch_start(self, trainer: Trainer, *args, **kwargs) -> None:
        now = time.perf_counter()
        if self._last_step_end is not None:
            self._val_data_wait_ms.append((now - self._last_step_end) * 1000)
        self._step_start = now

    def on_validation_batch_end(self, trainer: Trainer, *args, **kwargs) -> None:
        now = time.perf_counter()
        if self._step_start is not None:
            self._val_step_ms.append((now - self._step_start) * 1000)
        self._last_step_end = now

    def on_validation_epoch_end(self, trainer: Trainer, *args, **kwargs) -> None:
        if self._val_epoch_start is None:
            return
        epoch_s = time.perf_counter() - self._val_epoch_start
        metrics = {"timing/val_epoch_s": epoch_s}
        if self._val_data_wait_ms:
            metrics["timing/val_data_wait_ms"] = sum(self._val_data_wait_ms) / len(
                self._val_data_wait_ms
            )
        if self._val_step_ms:
            metrics["timing/val_step_wall_ms"] = sum(self._val_step_ms) / len(self._val_step_ms)
        trainer.logger.log_metrics(metrics, step=trainer.global_step)
        self._val_epoch_start = None
