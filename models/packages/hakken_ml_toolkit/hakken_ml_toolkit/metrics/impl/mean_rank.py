from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from hakken_ml_toolkit.metrics.core.contracts.metric import MetricConfig, MetricI
from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError

if TYPE_CHECKING:
    from hakken_ml_toolkit.metrics.core.entities import FloatTensor1D, FloatTensor2D, LongTensor1D


class MeanRankConfig(MetricConfig):
    pass


class MeanRank(MetricI[MeanRankConfig]):
    DEFAULT_CONFIG = MeanRankConfig()
    name = "mean_rank"

    def __init__(self, config: MeanRankConfig):
        super().__init__(config=config)
        self.reset()

    def reset(self) -> None:
        self.ranks_sum = 0.0
        self.num_samples = 0

    def update(self, scores: FloatTensor2D, targets: LongTensor1D) -> None:
        target_scores = scores[torch.arange(len(targets)), targets].unsqueeze(1)
        mask = scores >= target_scores

        target_ranks = mask.sum(dim=1)

        self.ranks_sum += target_ranks.sum().item()

        self.num_samples += len(targets)

    def compute(self) -> FloatTensor1D:
        if self.num_samples == 0:
            raise NoSamplesError()

        return torch.tensor([self.ranks_sum / self.num_samples])
