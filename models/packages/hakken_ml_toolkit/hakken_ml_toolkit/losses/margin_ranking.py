import torch

from hakken_ml_toolkit.losses.base.ranking_loss import RankingLossConfig, RankingLossI
from hakken_ml_toolkit.losses.common.domain import FloatTensor1D, FloatTensor2D


class MarginRankingLossConfig(RankingLossConfig):
    margin: float = 1.0


class MarginRankingLoss(RankingLossI[MarginRankingLossConfig]):
    """
    Formula: loss = max(0, margin + negative_score - positive_score)
    """

    def __init__(self, config: MarginRankingLossConfig):
        super().__init__(config)

    def _compute(
        self, positive_scores: FloatTensor2D, negative_scores: FloatTensor2D
    ) -> FloatTensor1D:
        expanded_positive_scores = positive_scores.expand_as(negative_scores)

        losses_all = torch.relu(self.config.margin + negative_scores - expanded_positive_scores)

        return losses_all.mean(dim=1)
