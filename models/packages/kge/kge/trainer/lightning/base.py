from typing import Any, cast

import pytorch_lightning as pl
import torch
import torchmetrics as tm
from hakken_ml_toolkit.losses.common.constants import ReduceType
from hakken_ml_toolkit.metrics import MeanReciprocalRank, MetricsDict, MetricsDictConfig
from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError
from hakken_ml_toolkit.ml_utils.extras import FactBatchUtils
from hakken_ml_toolkit.tracker import TrackerI
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kge.common.types import LongTensor2D
from kge.evaluator.base import KGEEvaluator
from kge.evaluator.utils import KGEEvalUtils
from kge.models.base import KGEI
from kge.negative_sampler.base import NegativeSamplerI
from kge.optim.factory import (
    LRSchedulerInfo,
    OptimizerInfo,
    lr_sched_factory,
    optim_factory,
)

TRAINING_LOSS_KEY = "training/loss"
VALIDATION_LOSS_KEY = "validation/loss"
LEARNING_RATE_KEY = "training/learning_rate"


class KGELightning(pl.LightningModule):
    def __init__(
        self,
        optimizer_info: OptimizerInfo,
        lr_sched_info: LRSchedulerInfo,
        model: KGEI,
        evaluator: KGEEvaluator,
        tracker: TrackerI,
        negative_sampler: NegativeSamplerI | None = None,
    ):
        super().__init__()

        self.model = model
        self.optimizer_info = optimizer_info
        self.lr_sched_info = lr_sched_info
        self.tracker = tracker
        self.evaluator = evaluator
        self.automatic_optimization = False

        self.negative_sampler = negative_sampler

        self.metrics_training = MetricsDict(MetricsDictConfig(reduce="mean"))
        self.metrics_validation = MetricsDict(MetricsDictConfig(reduce="mean"))

        self.metrics_training.add(TRAINING_LOSS_KEY)
        self.metrics_validation.add(VALIDATION_LOSS_KEY)

    def to(self, device: str | torch.device) -> "KGELightning":  # type: ignore[override]
        """
        Move the model to the specified device.
        """
        self.model.to(device)
        self.negative_sampler.to_device(device)
        return super().to(device)

    def get_sro_batch_neg(
        self, sro_batch_pos: LongTensor2D, num_negatives: int | None = None
    ) -> tuple[LongTensor2D, int]:
        sro_tensor_neg = self.negative_sampler.corrupt_batch(
            sro_batch=sro_batch_pos, num_negatives=num_negatives
        )
        num_negatives = sro_tensor_neg.shape[1]
        sro_batch_neg = sro_tensor_neg.view(-1, 3)
        return sro_batch_neg, num_negatives

    def on_train_epoch_start(self):
        self.metrics_training.reset()
        return super().on_train_epoch_start()

    def on_train_epoch_end(self):
        metrics = self.metrics_training.compute()

        self.tracker.track_data(metrics)
        self.tracker.increment_step()
        return super().on_train_epoch_end()

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> dict[str, Any]:
        raise NotImplementedError()

    def on_validation_epoch_start(self):
        self.evaluator.reset()
        self.metrics_validation.reset()
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        metrics = self.metrics_validation.compute()
        loss = metrics[VALIDATION_LOSS_KEY]
        sched = self.lr_schedulers()

        opt = self.optimizers()
        if isinstance(opt, list):
            msg = "Only one optimizer is allowed"
            raise NotImplementedError(msg)

        sched.step(loss)

        current_lr = opt.param_groups[0]["lr"]

        self.tracker.track_value(LEARNING_RATE_KEY, current_lr)
        self.log(VALIDATION_LOSS_KEY, loss, on_step=False, on_epoch=True)

        try:
            metrics_evaluator = self.evaluator.compute_metrics()
            metrics_evaluator = KGEEvalUtils.prefix_metrics(
                prefix="validation", metrics=metrics_evaluator
            )
            metrics.update(metrics_evaluator)
        except NoSamplesError:
            pass

        self.tracker.track_data(metrics)

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> dict[str, Any]:
        raise NotImplementedError()

    def test_step(self, batch: list[torch.Tensor], batch_idx: int):
        self.log_dict(self.validation_step(batch, batch_idx))

    def configure_optimizers(self) -> torch.optim.Optimizer | Any:
        optimizer = optim_factory(
            parameters=self.model.parameters(), optim_info=self.optimizer_info
        )

        sched = lr_sched_factory(optimizer, lr_sched_info=self.lr_sched_info)

        return [optimizer], [sched]

    def compute_loss(
        self, sro_batch_pos: torch.Tensor, reduce: ReduceType | None = None
    ) -> torch.Tensor:
        raise NotImplementedError()

    def compute_objective_from_dataset(
        self, dataset: Dataset, device: str = "cuda", loader_kwargs: dict | None = None
    ) -> float:
        """
        Computes the objective (Mean Reciprocal Rank) for the given dataset using the model.

        Args:
            dataset (Dataset): The dataset to evaluate the model on.
            device (str, optional): The device to move the model and data to, default is "cuda".
            loader_kwargs (dict, optional): Additional arguments for the DataLoader,
                like batch_size and shuffle.

        Returns:
            float: The computed Mean Reciprocal Rank (MRR) for the dataset.

        The function evaluates the model on the provided dataset by iterating through batches
        and computing the MRR metric. The model must be in evaluation mode, and the data will
        be processed on the specified device (e.g., CUDA).
        """

        self.model.eval()
        self.to(device)

        if loader_kwargs is None:
            loader_kwargs = {"batch_size": 1024, "shuffle": False}

        data_loader = DataLoader(dataset=dataset, **loader_kwargs)

        metric = MeanReciprocalRank()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing batches", total=len(data_loader)):
                sro_batch_pos: torch.Tensor = batch[0]
                sr_batch = FactBatchUtils.to_sr_batch(sro_batch_pos.to(device))
                targets: torch.Tensor = sro_batch_pos[:, 2]

                scores = self.model.score_objects(sr_batch)

                metric.update(scores=scores, targets=targets.to(device))

        return cast("float", metric.compute().item())

    def compute_loss_from_loader(self, data_loader: DataLoader, device: str = "cuda") -> float:
        """
        Compute the objective value for the validation dataset.

        Args:
            data_loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            float: The computed objective value.
        """
        self.model.eval()
        self.to(device)

        metric = tm.MeanMetric(nan_strategy="error").to(device)

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing batches", total=len(data_loader)):
                sro_batch_pos: torch.Tensor = batch[0]
                objective = self.compute_loss(
                    sro_batch_pos=sro_batch_pos.to(device), reduce=ReduceType.NONE
                )
                metric.update(objective)

        return metric.compute().item()
