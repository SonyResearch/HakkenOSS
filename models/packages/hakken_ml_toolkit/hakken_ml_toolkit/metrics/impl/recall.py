from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from hakken_ml_toolkit.metrics.core.contracts.metric import MetricConfig, MetricI
from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError, UnknownAverageError

if TYPE_CHECKING:
    from hakken_ml_toolkit.metrics.core.entities import FloatTensor1D, FloatTensor2D, LongTensor1D


class RecallConfig(MetricConfig):
    num_classes: int = 1
    average: str = "macro"  # Options: 'macro', 'micro', 'weighted', 'none'


class Recall(MetricI[RecallConfig]):
    DEFAULT_CONFIG = RecallConfig()
    name = "recall"

    def __init__(self, config: RecallConfig):
        super().__init__(config=config)
        self.num_classes = config.num_classes
        self.average = config.average
        self.true_positives = torch.zeros(self.num_classes, dtype=torch.float32)
        self.false_negatives = torch.zeros(self.num_classes, dtype=torch.float32)
        self.total_samples = 0

    def reset(self) -> None:
        self.true_positives.zero_()
        self.false_negatives.zero_()
        self.total_samples = 0

    def update(self, scores: FloatTensor2D, targets: LongTensor1D) -> None:
        predictions = torch.argmax(scores, dim=1)  # Shape: [batch_size]
        self.total_samples += targets.size(0)

        for cls in range(self.num_classes):
            cls_pred = predictions == cls
            cls_target = targets == cls

            tp = torch.sum(cls_pred & cls_target).item()
            fn = torch.sum(~cls_pred & cls_target).item()

            self.true_positives[cls] += tp
            self.false_negatives[cls] += fn

    def compute(self) -> FloatTensor1D:
        if self.total_samples == 0:
            raise NoSamplesError()

        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        recall[torch.isnan(recall)] = 0.0  # Handle division by zero

        if self.average == "macro":
            recall_score = recall.mean().item()
            return torch.tensor([recall_score])
        if self.average == "micro":
            tp_total = self.true_positives.sum()
            fn_total = self.false_negatives.sum()

            recall_micro = tp_total / (tp_total + fn_total + 1e-8)
            return torch.tensor([recall_micro.item()])
        if self.average == "weighted":
            support = self.true_positives + self.false_negatives
            weights = support / support.sum()
            recall_score = (recall * weights).sum().item()
            return torch.tensor([recall_score])
        if self.average == "none":
            return recall
        raise UnknownAverageError(self.average)
