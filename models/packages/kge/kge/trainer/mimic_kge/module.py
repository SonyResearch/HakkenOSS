from typing import Any

import pytorch_lightning as pl
import torch
from hakken_ml_toolkit.metrics import MetricsDict, MetricsDictConfig
from hakken_ml_toolkit.tracker import TrackerI
from torch import Tensor, nn
from torchmetrics import MeanMetric
from tqdm import tqdm

from kge.common.entities import KGPredictionSubgraph
from kge.data_loaders.mimic_kge import MimicKGEDataLoader
from kge.models.gnn import GNNKGE
from kge.optim.factory import (
    LRSchedulerInfo,
    OptimizerInfo,
    lr_sched_factory,
    optim_factory,
)
from kge.utils import remove_overlapping_edges

TRAINING_LOSS_KEY = "training/loss"
VALIDATION_LOSS_KEY = "validation/loss"
VALIDATION_LOSS_POS_KEY = VALIDATION_LOSS_KEY + "_pos"
VALIDATION_LOSS_NEG_KEY = VALIDATION_LOSS_KEY + "_neg"
LEARNING_RATE_KEY = "training/learning_rate"


BatchType = tuple[KGPredictionSubgraph, Tensor, Tensor, Tensor, Tensor]


class MimicKGELightning(pl.LightningModule):
    def __init__(
        self,
        optimizer_info: OptimizerInfo,
        lr_sched_info: LRSchedulerInfo,
        model: GNNKGE,
        tracker: TrackerI,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_info = optimizer_info
        self.lr_sched_info = lr_sched_info
        self.tracker = tracker

        self.metrics_training = MetricsDict(MetricsDictConfig(reduce="mean"))
        self.metrics_validation = MetricsDict(MetricsDictConfig(reduce="mean"))

        self.metrics_training.add(TRAINING_LOSS_KEY)
        self.metrics_validation.add(VALIDATION_LOSS_KEY)

        self.loss_fn = nn.MSELoss()

    def on_train_epoch_start(self):
        self.metrics_training.reset()
        return super().on_train_epoch_start()

    def on_train_epoch_end(self):
        metrics = self.metrics_training.compute()
        self.tracker.track_data(metrics)
        self.tracker.increment_step()
        return super().on_train_epoch_end()

    def training_step(self, batch: BatchType, _batch_idx: int) -> dict[str, Any] | Tensor:
        # Unpack the batch (assuming same structure as in the original script)
        batch_graph, facts_pos, facts_neg, target_scores_pos, target_scores_neg = batch

        edge_index = batch_graph.edge_index
        edge_type = batch_graph.edge_type

        edge_index, edge_type = remove_overlapping_edges(  # type: ignore[assignment]
            edges_to_exclude=batch_graph.edge_label_index,
            target_edge_index=edge_index,
            target_edge_labels=edge_type,
        )

        # Encode subgraph
        z = self.model.encode_subgraph(
            node_ids=batch_graph.node_ids,
            edge_index=edge_index,
            edge_type=edge_type,
        )  # [N_sub, H]

        # Compute scores for positive and negative triples
        pos_scores = self.model.score_from_z(
            z,
            subject_ids=facts_pos[:, 0],
            relation_ids=facts_pos[:, 1],
            object_ids=facts_pos[:, 2],
        )
        neg_scores = self.model.score_from_z(
            z,
            subject_ids=facts_neg[:, 0],
            relation_ids=facts_neg[:, 1],
            object_ids=facts_neg[:, 2],
        )

        # Concatenate scores and targets
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        target = torch.cat([target_scores_pos, target_scores_neg], dim=0)

        # Compute loss
        loss = self.loss_fn.forward(scores, target)

        loss_item = loss.detach().cpu().item()

        # Log metrics
        self.metrics_training.update(TRAINING_LOSS_KEY, loss_item)
        # Return dictionary for logging
        return loss

    def validation_step(self, batch: BatchType, _batch_idx: int) -> dict[str, Any]:
        # Unpack the batch
        batch_graph, facts_pos, facts_neg, target_scores_pos, target_scores_neg = batch

        # Encode subgraph
        z = self.model.encode_subgraph(
            node_ids=batch_graph.node_ids, edge_index=batch_graph.edge_index
        )  # [N_sub, H]

        # Compute scores for positive and negative triples
        pos_scores = self.model.score_from_z(
            z,
            subject_ids=facts_pos[:, 0],
            relation_ids=facts_pos[:, 1],
            object_ids=facts_pos[:, 2],
        )
        neg_scores = self.model.score_from_z(
            z,
            subject_ids=facts_neg[:, 0],
            relation_ids=facts_neg[:, 1],
            object_ids=facts_neg[:, 2],
        )

        # Concatenate scores and targets
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        target = torch.cat([target_scores_pos, target_scores_neg], dim=0)

        # Compute loss
        loss = self.loss_fn.forward(scores, target).detach().cpu().item()
        loss_pos = self.loss_fn.forward(pos_scores, target_scores_pos).detach().cpu().item()
        loss_neg = self.loss_fn.forward(neg_scores, target_scores_neg).detach().cpu().item()

        # Log metrics
        self.metrics_validation.update(VALIDATION_LOSS_KEY, loss)
        self.metrics_validation.update(VALIDATION_LOSS_POS_KEY, loss_pos)
        self.metrics_validation.update(VALIDATION_LOSS_NEG_KEY, loss_neg)

        return {VALIDATION_LOSS_KEY: loss}

    def on_validation_epoch_start(self):
        self.metrics_validation.reset()
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        metrics = self.metrics_validation.compute()
        loss = metrics[VALIDATION_LOSS_KEY]

        opt = self.optimizers()
        if isinstance(opt, list):
            msg = "Only one optimizer is allowed"
            raise NotImplementedError(msg)

        current_lr = opt.param_groups[0]["lr"]

        self.tracker.track_value(LEARNING_RATE_KEY, current_lr)
        self.log(VALIDATION_LOSS_KEY, loss, on_step=False, on_epoch=True)

        self.tracker.track_data(metrics)

    def configure_optimizers(self) -> torch.optim.Optimizer | Any:
        optimizer = optim_factory(
            parameters=self.model.parameters(), optim_info=self.optimizer_info
        )

        scheduler = lr_sched_factory(optimizer, lr_sched_info=self.lr_sched_info)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": VALIDATION_LOSS_KEY,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def compute_objective_from_dataloader(
        self, dataloader: MimicKGEDataLoader, device: str | torch.device = "cpu"
    ) -> float:
        self.model.eval()
        self.to(device)

        loss_fn = nn.MSELoss(reduce="none")

        metric = MeanMetric()

        batch: BatchType
        for batch in tqdm(dataloader, desc="Processing batches", total=len(dataloader)):
            batch_graph: KGPredictionSubgraph
            batch_graph, facts_pos, facts_neg, target_scores_pos, target_scores_neg = batch

            batch_graph = batch_graph.to(device)

            facts_pos = facts_pos.to(device)
            if facts_neg is not None:
                facts_neg = facts_neg.to(device)
            target_scores_pos = target_scores_pos.to(device)
            if target_scores_neg is not None:
                target_scores_neg = target_scores_neg.to(device)

            # Encode subgraph
            z = self.model.encode_subgraph(
                node_ids=batch_graph.node_ids, edge_index=batch_graph.edge_index
            )  # [N_sub, H]

            # Compute scores for positive and negative triples
            pos_scores = self.model.score_from_z(
                z,
                subject_ids=facts_pos[:, 0],
                relation_ids=facts_pos[:, 1],
                object_ids=facts_pos[:, 2],
            )
            neg_scores = self.model.score_from_z(
                z,
                subject_ids=facts_neg[:, 0],
                relation_ids=facts_neg[:, 1],
                object_ids=facts_neg[:, 2],
            )

            # Concatenate scores and targets
            scores = torch.cat([pos_scores, neg_scores], dim=0)
            target = torch.cat([target_scores_pos, target_scores_neg], dim=0)

            # Compute loss
            loss = loss_fn.forward(scores, target).detach().cpu()
            metric.update(loss)

        return metric.compute().item()
