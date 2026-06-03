from typing import Any, cast

import torch
from hakken_ml_toolkit.losses import RankingLossI
from hakken_ml_toolkit.losses.common.constants import ReduceType
from hakken_ml_toolkit.ml_utils.extras import FactBatchUtils, PyTorchUtils
from hakken_ml_toolkit.tracker import TrackerI

from kge.evaluator.base import KGEEvaluator
from kge.models.base import KGEI
from kge.negative_sampler.base import NegativeSamplerI
from kge.optim.factory import LRSchedulerInfo, OptimizerInfo
from kge.trainer.lightning.base import (
    TRAINING_LOSS_KEY,
    VALIDATION_LOSS_KEY,
    KGELightning,
)


class KGERankingLightning(KGELightning):
    def __init__(
        self,
        model: KGEI,
        negative_sampler: NegativeSamplerI,
        optimizer_info: OptimizerInfo,
        lr_sched_info: LRSchedulerInfo,
        loss_fn: RankingLossI,
        tracker: TrackerI,
        evaluator: KGEEvaluator,
        remove_triples_path: str | None = None,
    ):
        super().__init__(
            optimizer_info=optimizer_info,
            lr_sched_info=lr_sched_info,
            model=model,
            evaluator=evaluator,
            tracker=tracker,
            negative_sampler=negative_sampler,
        )
        self.loss_fn = loss_fn
        self.sro_remove: torch.Tensor | None = None
        if remove_triples_path is not None:
            self.sro_remove = PyTorchUtils.load(remove_triples_path)

    def training_step(self, batch: list[torch.Tensor], _batch_idx: int) -> dict[str, Any]:
        sro_batch_pos = batch[0]

        device = batch[0].device

        batch_size = sro_batch_pos.size(0)

        sro_batch_neg, num_negatives = self.get_sro_batch_neg(sro_batch_pos)

        pos_output = self.model(sro_batch_pos)
        neg_output = self.model(sro_batch_neg)

        negative_scores = neg_output.scores.view(batch_size, num_negatives)

        if self.sro_remove is not None:
            mask_remove = torch.BoolTensor([False] * batch_size).to(device)

            for sro_remove_i in self.sro_remove:
                subject_pos = FactBatchUtils.subject(sro_batch_pos)
                relation_pos = FactBatchUtils.relation(sro_batch_pos)
                object_pos = FactBatchUtils.object(sro_batch_pos)
                mask_pos = (
                    (subject_pos == sro_remove_i[0])
                    & (relation_pos == sro_remove_i[1])
                    & (object_pos == sro_remove_i[2])
                )
                mask_remove |= mask_pos

            pos_output.scores = pos_output.scores[~mask_remove]
            negative_scores = negative_scores[~mask_remove]

        loss = self.loss_fn.compute(pos_output.scores, negative_scores)

        opt = cast("torch.optim.Optimizer", self.optimizers())
        if isinstance(opt, list):
            msg = "Only one optimizer is allowed"
            raise NotImplementedError(msg)

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.metrics_training.update(TRAINING_LOSS_KEY, loss.detach().cpu())
        return {TRAINING_LOSS_KEY: loss}

    def compute_loss(
        self, sro_batch_pos: torch.Tensor, reduce: ReduceType | None = None
    ) -> torch.Tensor:
        if reduce is not None:
            self.loss_fn.set_reduce(reduce=reduce)

        batch_size = sro_batch_pos.size(0)

        sro_batch_neg, num_negatives = self.get_sro_batch_neg(sro_batch_pos)

        pos_output = self.model(sro_batch_pos)
        neg_output = self.model(sro_batch_neg)

        negative_scores = neg_output.scores.view(batch_size, num_negatives)

        loss = self.loss_fn.compute(pos_output.scores, negative_scores)

        self.loss_fn.reset_reduce()
        return loss

    def validation_step(self, batch: list[torch.Tensor], _batch_idx: int) -> dict[str, Any]:
        sro_batch_pos = batch[0]

        self.evaluator.set_model(self.model)

        self.evaluator.update_in_batches(sro_batch=sro_batch_pos, batch_size=128)

        loss = self.compute_loss(sro_batch_pos=sro_batch_pos)

        self.metrics_validation.update(VALIDATION_LOSS_KEY, loss.detach().cpu())

        return {VALIDATION_LOSS_KEY: loss}
