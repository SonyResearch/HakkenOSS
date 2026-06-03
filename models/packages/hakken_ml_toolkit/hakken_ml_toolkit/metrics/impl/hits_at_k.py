from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from hakken_ml_toolkit.metrics.core.contracts.ranking_metric import (
    RankingMetricConfig,
    RankingMetricI,
)
from hakken_ml_toolkit.metrics.core.exceptions import (
    NoSamplesError,
    TopKLargerThanNumberOfItemsError,
)

if TYPE_CHECKING:
    from hakken_ml_toolkit.metrics.core.entities import FloatTensor1D, FloatTensor2D, LongTensor1D


class HitsAtKConfig(RankingMetricConfig):
    top_k: int = 1


class HitsAtK(RankingMetricI[HitsAtKConfig]):
    """
    Computes the Hits@K metric.

    This metric calculates the proportion of samples where the correct target
    item appears within the top-K predicted items. A hit is counted when the
    target index is present in the top-K indices of the score matrix.
    """

    DEFAULT_CONFIG = HitsAtKConfig()
    name = "hits_at_k"

    def __init__(self, top_k: int = 1, **kwargs: Any):
        config = HitsAtKConfig(top_k=top_k, **kwargs)
        super().__init__(config=config)
        self.reset()

    def reset(self) -> None:
        self._total_hits = 0.0
        self._total = 0.0

    def update(self, scores: FloatTensor2D, targets: LongTensor1D) -> None:
        if self.config.top_k > scores.size(1):
            raise TopKLargerThanNumberOfItemsError(
                top_k=self.config.top_k, number_of_items=scores.size(1)
            )

        _, top_k_indices = torch.topk(scores, k=self.config.top_k, dim=1)

        hits = torch.any(top_k_indices == targets.unsqueeze(1), dim=1)
        self._total_hits += hits.sum().item()
        self._total += targets.size(0)

    def compute(self) -> FloatTensor1D:
        if self._total == 0:
            raise NoSamplesError()

        hits_at_k = self._total_hits / self._total
        return torch.tensor([hits_at_k])
