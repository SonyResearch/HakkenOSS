from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.classification import MultilabelF1Score

from hakken_models.core.entities.kg_data_with_preds import KGDataWithPreds
from hakken_models.losses import loss_fn_registry

from .base import THiGER


class LitTHiGER(LightningModule):
    def __init__(
        self,
        thiger: THiGER,
        loss_fn: nn.Module,
        optimizer_cls: type[Optimizer] = Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler_cls: type[LRScheduler] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.thiger = thiger
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}

        # Store names for MLFlow tracking
        self.loss_fn_name = loss_fn.__class__.__name__
        self.optimizer_name = optimizer_cls.__name__

        self.save_hyperparameters(
            ignore=[
                "thiger",
                "loss_fn",
                "optimizer_cls",
                "scheduler_cls",
            ],
            logger=False,  # Automatically log to MLFlow
        )

    def forward(self, entity_pair_batch: Tensor) -> Tensor:
        """TODO"""
        return self.thiger.compute_logits(entity_pair_batch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and optionally scheduler.

        Returns:
            Optimizer or dict with optimizer and scheduler configuration
        """
        optimizer = self.optimizer_cls(params=self.thiger.parameters(), **self.optimizer_kwargs)

        if self.scheduler_cls is not None:
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.scheduler_kwargs.get("interval", "epoch"),
                    "frequency": self.scheduler_kwargs.get("frequency", 1),
                },
            }

        return optimizer

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute binary multi-label loss.

        Args:
            logits: Predicted logits from the model.
            targets: Ground truth labels.

        Returns:
            Computed loss value. [num_samples]
        """

        loss: Tensor = self.loss_fn.forward(logits, targets.float())
        return loss.mean()

    def training_step(self, batch: KGDataWithPreds, _batch_idx: int) -> Tensor:
        """Training step for a single batch."""

        entity_pair_batch = batch.edge_label_index.t().contiguous()
        self.thiger.set_context_temporal_kg(batch)

        logits = self.forward(entity_pair_batch)

        edge_labels = batch.edge_label

        loss = self.compute_loss(logits, edge_labels)
        # Log metrics
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, batch_size=entity_pair_batch.shape[0]
        )

        return loss

    def compute_macro_avg_f1(self, logits: Tensor, targets: Tensor) -> Tensor:
        """TODO"""
        f1_metric = MultilabelF1Score(
            num_labels=logits.shape[1], threshold=0.5, average="macro", zero_division=0
        ).to(logits.device)
        probs = torch.sigmoid(logits)
        return f1_metric(probs, targets.long())

    def validation_step(self, batch: KGDataWithPreds, _batch_idx: int) -> Tensor:
        """Validation step."""
        entity_pair_batch = batch.edge_label_index.t().contiguous()

        self.thiger.set_context_temporal_kg(batch)

        logits = self.forward(entity_pair_batch)
        edge_labels = batch.edge_label

        loss = self.compute_loss(logits, edge_labels)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, batch_size=entity_pair_batch.shape[0]
        )

        f1_score = self.compute_macro_avg_f1(logits, edge_labels)
        self.log("val_macro_avg_f1_score", f1_score, on_step=False, on_epoch=True, prog_bar=True)

        return loss


def create_lit_thiger(
    thiger: THiGER,
    loss_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    scheduler_config: dict[str, Any] | None = None,
) -> LitTHiGER:
    """Create and configure a LitTHiGER instance from configuration dictionaries.

    Args:
        thiger: The THiGER model instance to wrap in the Lightning module.
        loss_config: Dictionary containing loss function configuration. Must include:
            - "name": Name of the loss function class (e.g., "BCEWithLogitsLoss", "FLWithLogitsLoss")
            - "kwargs": Optional dictionary of keyword arguments for the loss function
        optimizer_config: Dictionary containing optimizer configuration. Must include:
            - "name": Name of the optimizer class (e.g., "Adam")
            - "kwargs": Dictionary of keyword arguments for the optimizer
        scheduler_config: Optional dictionary containing learning rate scheduler
            configuration. If provided, must include:
            - "name": Name of the scheduler class (e.g., "StepLR")
            - "kwargs": Optional dictionary of keyword arguments for the scheduler

    Returns:
        Configured LitTHiGER instance ready for training with PyTorch Lightning.
    """
    loss_fn = loss_fn_registry.create(loss_config["name"], **loss_config.get("kwargs", {}))

    optimizer_cls = getattr(torch.optim, optimizer_config["name"])

    scheduler_cls = None
    if scheduler_config:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config["name"])

    return LitTHiGER(
        thiger=thiger,
        loss_fn=loss_fn,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_config["kwargs"],
        scheduler_cls=scheduler_cls,
        scheduler_kwargs=scheduler_config.get("kwargs", {}) if scheduler_config else None,
    )
