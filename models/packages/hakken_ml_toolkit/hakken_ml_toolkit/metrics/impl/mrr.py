from __future__ import annotations

from typing import TYPE_CHECKING

from hakken_ml_toolkit.metrics.core.contracts.ranking_metric import (
    RankingMetricConfig,
    RankingMetricI,
)
from hakken_ml_toolkit.metrics.core.exceptions import NoSamplesError

if TYPE_CHECKING:
    from hakken_ml_toolkit.metrics.core.entities import FloatTensor1D, FloatTensor2D, LongTensor1D


from torcheval.metrics import ReciprocalRank


class MRRConfig(RankingMetricConfig):
    pass


class MeanReciprocalRank(RankingMetricI[MRRConfig]):
    """
    The Mean Reciprocal Rank (MRR) computes the mean of the reciprocal ranks
    across all queries.

    For a single query, the reciprocal rank is 1/r, where r is the rank position
    of the first relevant item.

     Note:
        In case of tied scores, this implementation assigns the higher rank (better performance)
        to the tied items. For example, if two items have the same score, they both receive
        the higher rank position.
    """

    DEFAULT_CONFIG = MRRConfig()
    name = "mrr"

    def __init__(self, config: MRRConfig = DEFAULT_CONFIG):
        super().__init__(config=config)
        self.reciprocal_rank = ReciprocalRank()
        self.reset()

    def reset(self) -> None:
        self.reciprocal_rank.reset()

    def update(self, scores: FloatTensor2D, targets: LongTensor1D) -> None:
        self.reciprocal_rank.update(scores, targets)

    def compute(self) -> FloatTensor1D:
        reciprocal_ranks = self.reciprocal_rank.compute()

        if reciprocal_ranks.numel() == 0:
            raise NoSamplesError()

        return reciprocal_ranks.mean()
