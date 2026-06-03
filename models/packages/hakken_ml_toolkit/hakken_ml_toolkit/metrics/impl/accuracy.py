from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from hakken_ml_toolkit.metrics.core.contracts.metric import MetricConfig, MetricI
from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError

if TYPE_CHECKING:
    from hakken_ml_toolkit.metrics.core.entities import FloatTensor1D, FloatTensor2D, LongTensor1D


class AccuracyConfig(MetricConfig):
    reduce: str = "mean"


class Accuracy(MetricI[AccuracyConfig]):
    DEFAULT_CONFIG = AccuracyConfig()
    name = "accuracy"

    def __init__(self, config: AccuracyConfig | dict[str, Any] | None = None):
        super().__init__(config=config)
        self.reset()

    def reset(self) -> None:
        self._correct = 0.0
        self._total = 0.0

    def update(self, scores: FloatTensor2D, targets: LongTensor1D) -> None:
        # Get the predicted classes
        predictions = torch.argmax(scores, dim=1)

        self._correct += (predictions == targets).sum().item()
        self._total += targets.size(0)

    def compute(self) -> FloatTensor1D:
        if self._total == 0:
            raise NoSamplesError()

        accuracy = self._correct / self._total
        return self.reduce(torch.tensor([accuracy]))
