from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from pydantic import Field

from hakken_ml_toolkit.metrics.core.contracts.metric import MetricConfig, MetricI
from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError, UnknownAverageError

if TYPE_CHECKING:
    from hakken_ml_toolkit.metrics.core.entities import FloatTensor1D, FloatTensor2D, LongTensor1D


class F1Config(MetricConfig):
    reduce: str = "mean"
    num_classes: int = Field(default=2, description="Number of classes")
    average: str = Field(
        default="macro",
        description="Averaging method: 'macro', 'micro', 'weighted', or 'none'",
    )


EPSILON = 1e-8


class F1(MetricI[F1Config]):
    DEFAULT_CONFIG = F1Config()
    name = "f1"

    def __init__(self, config: F1Config | dict[str, Any] | None = None):
        super().__init__(config=config)
        self.reset()

    def reset(self) -> None:
        self._true_positives = torch.zeros(self.config.num_classes, dtype=torch.float64)
        self._false_positives = torch.zeros(self.config.num_classes, dtype=torch.float64)
        self._false_negatives = torch.zeros(self.config.num_classes, dtype=torch.float64)
        self._total = 0

    def update(self, scores: FloatTensor2D, targets: LongTensor1D) -> None:
        predictions = torch.argmax(scores, dim=1)
        self._total += targets.size(0)

        for cls in range(self.config.num_classes):
            cls_pred = predictions == cls
            cls_target = targets == cls

            self._true_positives[cls] += torch.sum(cls_pred & cls_target).item()
            self._false_positives[cls] += torch.sum(cls_pred & ~cls_target).item()
            self._false_negatives[cls] += torch.sum(~cls_pred & cls_target).item()

    def compute(self) -> FloatTensor1D:
        """
        Computes F1 score based on accumulated true positives, false positives, and false
        negatives.

        NaN values in the F1 calculation (which occur when a class has no true positives)
        are set to 0.0.

        Returns:
            FloatTensor1D: F1 score(s) based on the configured averaging method
            ('macro', 'micro', 'weighted', or 'none').
        """

        if self._total == 0:
            raise NoSamplesError()

        precision = self._true_positives / (self._true_positives + self._false_positives + EPSILON)
        recall = self._true_positives / (self._true_positives + self._false_negatives + EPSILON)
        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_per_class[torch.isnan(f1_per_class)] = 0.0

        if self.config.average == "macro":
            f1_score = f1_per_class.mean()
        elif self.config.average == "micro":
            tp_total = self._true_positives.sum()
            fp_total = self._false_positives.sum()
            fn_total = self._false_negatives.sum()

            precision_micro = tp_total / (tp_total + fp_total + EPSILON)
            recall_micro = tp_total / (tp_total + fn_total + EPSILON)
            f1_score = (
                2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + EPSILON)
            )
        elif self.config.average == "weighted":
            support = self._true_positives + self._false_negatives
            weights = support / support.sum()
            f1_score = (f1_per_class * weights).sum()
        elif self.config.average == "none":
            f1_score = f1_per_class
        else:
            raise UnknownAverageError(self.config.average)

        return f1_score if self.config.average == "none" else torch.tensor([f1_score.item()])
