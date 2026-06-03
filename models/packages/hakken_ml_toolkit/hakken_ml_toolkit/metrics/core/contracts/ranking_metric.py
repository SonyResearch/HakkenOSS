from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

from hakken_ml_toolkit.metrics.core.contracts.metric import MetricConfig, MetricI

if TYPE_CHECKING:
    from hakken_ml_toolkit.metrics.core.entities import FloatTensor2D, LongTensor1D


class RankingMetricConfig(MetricConfig):
    pass


T = TypeVar("T", bound=RankingMetricConfig)


class RankingMetricI(MetricI[T]):
    @abstractmethod
    def update(self, scores: FloatTensor2D, targets: LongTensor1D) -> None:
        """
        Update metric state with batch of entity/relation predictions and targets.

        Args:
            scores: Predicted scores for each entity/relation candidate,
                    shape (batch_size, num_entities/relations)
            targets: Ground truth entity/relation indices, shape (batch_size,)
        """
        pass
