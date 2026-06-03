from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from hakken_ml_toolkit.metrics.core.contracts.metric import MetricConfig, MetricI
from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError, UnknownAverageError

if TYPE_CHECKING:
    from hakken_ml_toolkit.metrics.core.entities import FloatTensor1D, FloatTensor2D, LongTensor1D


class PrecisionConfig(MetricConfig):
    num_classes: int = 1
    average: str = "macro"  # Options: 'macro', 'micro', 'weighted', 'none'


class Precision(MetricI[PrecisionConfig]):
    DEFAULT_CONFIG = PrecisionConfig()
    name = "precision"

    def __init__(self, config: PrecisionConfig):
        super().__init__(config=config)

        self.true_positives = torch.zeros(self.config.num_classes, dtype=torch.float32)
        self.false_positives = torch.zeros(self.config.num_classes, dtype=torch.float32)
        self.total_samples = 0
        self.class_counts = torch.zeros(self.config.num_classes, dtype=torch.float32)

    def reset(self) -> None:
        self.true_positives.zero_()
        self.false_positives.zero_()
        self.total_samples = 0
        self.class_counts.zero_()

    def update(self, scores: FloatTensor2D, targets: LongTensor1D) -> None:
        predictions = torch.argmax(scores, dim=1)  # Shape: [batch_size]
        self.total_samples += targets.size(0)

        for cls in range(self.config.num_classes):
            cls_pred = predictions == cls
            cls_target = targets == cls

            self.class_counts[cls] += torch.sum(cls_target).item()

            tp = torch.sum(cls_pred & cls_target).item()
            fp = torch.sum(cls_pred & ~cls_target).item()

            self.true_positives[cls] += tp
            self.false_positives[cls] += fp

    def compute(self) -> FloatTensor1D:
        if self.total_samples == 0:
            raise NoSamplesError()

        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        precision[torch.isnan(precision)] = 0.0  # Handle division by zero

        if self.config.average == "macro":
            precision_score = precision.mean().item()
            return torch.tensor([precision_score])
        if self.config.average == "micro":
            tp_total = self.true_positives.sum()
            fp_total = self.false_positives.sum()

            precision_micro = tp_total / (tp_total + fp_total + 1e-8)
            return torch.tensor([precision_micro.item()])
        if self.config.average == "weighted":
            weights = self.class_counts / self.class_counts.sum()
            precision_score = (precision * weights).sum().item()
            return torch.tensor([precision_score])
        if self.config.average == "none":
            return precision
        raise UnknownAverageError(self.config.average)
