from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hakken_models.evaluators.metric_bundle import MetricBundle
from hakken_models.evaluators.metric_hub import MetricHub, MetricHubConfig
from hakken_models.losses import loss_fn_registry
from hakken_models.losses.ranking_relation import RankingRelationLoss

from .base import KGE


def build_default_lit_kge_val_metric_hub() -> MetricHub:
    """Return a :class:`MetricHub` with sampled mean-rank validation metric."""
    bundle = MetricBundle(
        name="mean_rank",
        metric_class="hakken_models.evaluators.metrics.mean_rank_metric.MeanRankMetric",
        metric_kwargs={},
        input_bindings={
            "pos_scores": "pos_scores",
            "neg_scores": "neg_scores",
        },
    )
    return MetricHub([bundle])


def build_lit_kge_val_metric_hub(
    cfg: MetricHubConfig,
    *,
    num_relations: int | None = None,
) -> MetricHub | None:
    """Build a validation :class:`MetricHub` from :class:`MetricHubConfig`.

    When ``bundles`` is ``None`` and the config is enabled, returns the same hub as
    :func:`build_default_lit_kge_val_metric_hub`. When ``bundles`` is an empty
    list, returns an empty hub. ``metric_kwargs.num_labels: -1`` is replaced by
    ``num_relations`` when provided.
    """
    if not cfg.enabled:
        return None
    if cfg.bundles is None:
        return build_default_lit_kge_val_metric_hub()

    built: list[MetricBundle] = []
    for raw in cfg.bundles:
        spec = dict(raw)
        mk = dict(spec.get("metric_kwargs", {}))
        if mk.get("num_labels") == -1:
            if num_relations is None:
                raise ValueError(
                    "num_relations is required when a bundle has metric_kwargs.num_labels == -1"
                )
            mk["num_labels"] = num_relations
        spec["metric_kwargs"] = mk
        built.append(MetricBundle(**spec))
    return MetricHub(built)


class LitKGE(LightningModule):
    """Lightning module for KGE training with :class:`RankingRelationLoss`.

    Use ``rel_loss_weight=0`` in the loss for pure margin ranking; set
    ``rel_loss_weight>0`` and supply ``relation_labels`` in the batch for the
    relation multi-label term. For KGE training, negative aggregation
    (hardest/mean) is set on ``TrainKGEConfig.negative_strategy`` and injected
    into loss kwargs before this module is constructed.

    Optional ``val_metric_hub`` accumulates validation metrics during
    ``validation_step``; values are computed and logged under ``val/<name>`` in
    ``on_validation_epoch_end``. Use :func:`build_default_lit_kge_val_metric_hub`
    for the default sampled mean-rank metric.
    """

    def __init__(
        self,
        kge: KGE,
        loss_fn: RankingRelationLoss,
        optimizer_cls: type[Optimizer] = Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler_cls: type[LRScheduler] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        val_metric_hub: MetricHub | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(loss_fn, RankingRelationLoss):
            raise TypeError(
                f"LitKGE requires loss_fn to be RankingRelationLoss, got {type(loss_fn).__name__}"
            )

        self.kge = kge
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.val_metric_hub = val_metric_hub

        self.loss_fn_name = loss_fn.__class__.__name__
        self.optimizer_name = optimizer_cls.__name__

        self.save_hyperparameters(
            ignore=[
                "kge",
                "data_loaders",
                "loss_fn",
                "optimizer_cls",
                "scheduler_cls",
                "val_metric_hub",
            ],
            logger=False,
        )

    def on_validation_epoch_start(self) -> None:
        if self.val_metric_hub is not None:
            self.val_metric_hub.to(self.device)

    def forward(self, facts_tensor: torch.Tensor) -> torch.Tensor:
        """Score triples ``(s, r, o)``.

        Args:
            facts_tensor: ``[batch_size, 3]`` (subject, relation, object) indices.

        Returns:
            Scores ``[batch_size]`` (higher = more plausible under the score fn).
        """
        return self.kge.forward(facts_tensor)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return optimizer, or optimizer + LR scheduler if configured."""
        optimizer = self.optimizer_cls(params=self.kge.parameters(), **self.optimizer_kwargs)

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

    def forward_negatives(self, neg_facts_tensor: torch.Tensor) -> torch.Tensor:
        """Score corrupted triples.

        Args:
            neg_facts_tensor: ``[batch_size, num_negatives, 3]``.

        Returns:
            Scores ``[batch_size, num_negatives]``.
        """
        num_fact_components = neg_facts_tensor.shape[2]
        num_negatives = neg_facts_tensor.shape[1]
        batch_size = neg_facts_tensor.shape[0]

        neg_facts_flat = neg_facts_tensor.view(-1, num_fact_components)
        neg_scores_flat = self.forward(neg_facts_flat)
        return neg_scores_flat.view(batch_size, num_negatives)

    def _score_and_loss(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor, Tensor | None, Tensor | None]:
        """Score positives/negatives and compute loss.

        Returns:
            Tuple of (loss, loss_dict, pos_scores, neg_scores, rel_logits, rel_labels).
            ``rel_logits`` / ``rel_labels`` are set whenever the batch carries
            ``relation_labels`` (for validation metrics); the loss still only uses
            them when ``rel_loss_weight != 0``.
        """
        pos_facts = batch["positives"]
        neg_facts = batch["negatives"]

        pos_scores = self.forward(pos_facts)
        neg_scores = self.forward_negatives(neg_facts)

        rel_logits: Tensor | None = None
        rel_labels: Tensor | None = batch.get("relation_labels")
        if rel_labels is not None:
            rel_logits = self.kge.score_relations(pos_facts[:, 0].long(), pos_facts[:, 2].long())
            rel_labels = rel_labels.to(device=rel_logits.device, dtype=rel_logits.dtype)

        loss_rel_logits = rel_logits if self.loss_fn.rel_loss_weight != 0 else None
        loss_rel_labels = rel_labels if self.loss_fn.rel_loss_weight != 0 else None
        loss, loss_dict = self.loss_fn(
            pos_scores, neg_scores, rel_logits=loss_rel_logits, rel_labels=loss_rel_labels
        )
        return loss, loss_dict, pos_scores, neg_scores, rel_logits, rel_labels

    def _log_step_metrics(
        self,
        prefix: str,
        loss: Tensor,
        loss_dict: dict[str, Tensor],
        batch_size: int,
    ) -> None:
        """Log loss components with the given prefix (e.g. 'train' or 'val')."""
        log_kw = dict(on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log(f"{prefix}/loss", loss, **log_kw)
        self.log(f"{prefix}/entity_loss", loss_dict["entity"], **log_kw)
        if "relation" in loss_dict:
            self.log(f"{prefix}/rel_loss", loss_dict["relation"], **log_kw)

    def _log_val_metric_hub_results(self, results: dict[str, Any]) -> None:
        log_kw = dict(on_step=False, on_epoch=True, sync_dist=True)
        for name, value in results.items():
            if isinstance(value, Tensor):
                v = value.detach()
                scalar = v.item() if v.numel() == 1 else v.float().mean().item()
            else:
                scalar = float(value)
            prog_bar = name == "mean_rank"
            self.log(f"val/{name}", scalar, prog_bar=prog_bar, **log_kw)

    def training_step(self, batch: dict[str, Tensor], _batch_idx: int) -> Tensor:
        """One training step: positives/negatives (and optional relation labels)."""
        loss, loss_dict, pos_scores, _, _, _ = self._score_and_loss(batch)
        self._log_step_metrics("train", loss, loss_dict, pos_scores.size(0))
        return loss

    def validation_step(self, batch: dict[str, Tensor], _batch_idx: int) -> Tensor:
        """Validation loss; non-loss metrics go through ``val_metric_hub``."""
        loss, loss_dict, pos_scores, neg_scores, rel_logits, rel_labels = self._score_and_loss(
            batch
        )
        self._log_step_metrics("val", loss, loss_dict, pos_scores.size(0))
        if self.val_metric_hub is not None:
            hub_kw: dict[str, Any] = {
                "pos_scores": pos_scores.detach(),
                "neg_scores": neg_scores.detach(),
            }
            if rel_logits is not None and rel_labels is not None:
                hub_kw["rel_logits"] = rel_logits.detach()
                hub_kw["rel_labels"] = rel_labels.detach()
            self.val_metric_hub.update(**hub_kw)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.val_metric_hub is None:
            return
        results = self.val_metric_hub.compute_and_reset()
        self._log_val_metric_hub_results(results)


def create_lit_kge(
    kge: KGE,
    loss_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    scheduler_config: dict[str, Any] | None = None,
    val_metric_hub: MetricHub | None = None,
) -> LitKGE:
    """Construct :class:`LitKGE` from Hydra-style loss/optimizer dicts.

    Args:
        kge: KGE model (embeddings + score function).
        loss_config: Must be ``{"name": "RankingRelationLoss", "kwargs": {...}}``.
            Use :meth:`~hakken_models.core.configs.train_common.LossConfig.with_kge_negative_strategy`
            when building from :class:`~hakken_models.core.configs.train_kge.TrainKGEConfig`.
        optimizer_config: ``{"name": "Adam", "kwargs": {...}}`` etc.
        scheduler_config: Optional ``{"name": "...", "kwargs": {...}}``.

    Returns:
        Configured :class:`LitKGE`.

    Raises:
        ValueError: If ``loss_config["name"]`` is not ``RankingRelationLoss``.
    """
    loss_name = loss_config["name"]
    loss_kwargs = dict(loss_config.get("kwargs", {}))

    if loss_name != "RankingRelationLoss":
        raise ValueError(
            f"KGE loss must be RankingRelationLoss (got {loss_name!r}). "
            "Use entity_loss / rel_loss_weight in kwargs."
        )
    loss_fn = loss_fn_registry.create("RankingRelationLoss", **loss_kwargs)

    optimizer_cls = getattr(torch.optim, optimizer_config["name"])

    scheduler_cls = None
    if scheduler_config:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config["name"])

    return LitKGE(
        kge=kge,
        loss_fn=loss_fn,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_config["kwargs"],
        scheduler_cls=scheduler_cls,
        scheduler_kwargs=scheduler_config.get("kwargs", {}) if scheduler_config else None,
        val_metric_hub=val_metric_hub,
    )
