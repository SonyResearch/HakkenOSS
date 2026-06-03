from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel

from hakken_ml_toolkit.losses.common.constants import ReduceType
from hakken_ml_toolkit.losses.common.exceptions import (
    ShapeMismatchError,
    UnknownReductionError,
)

if TYPE_CHECKING:
    from hakken_ml_toolkit.losses.common.domain import (
        FloatTensor1D,
        FloatTensor2D,
        FloatTensorScalar,
    )


class RankingLossConfig(BaseModel):
    reduce: ReduceType = ReduceType.SUM


T = TypeVar("T", bound=RankingLossConfig)


class RankingLossI(ABC, Generic[T]):
    """
    Abstract interface for ranking loss functions.

    Ranking losses compare positive scores against negative scores,  typically used in
    contrastive learning.
    """

    def __init__(self, config: T):
        self.config: T = config
        self._reduce: ReduceType = self.config.reduce

    @abstractmethod
    def _compute(
        self, positive_scores: FloatTensor2D, negative_scores: FloatTensor2D
    ) -> FloatTensor1D:
        pass

    def set_reduce(self, reduce: ReduceType) -> None:
        self._reduce = reduce

    def reset_reduce(self) -> None:
        self._reduce = self.config.reduce

    def reduce(
        self, losses: FloatTensor1D, keepdim: bool = True
    ) -> FloatTensor1D | FloatTensorScalar:
        if self._reduce == ReduceType.SUM:
            data = losses.sum()
            return data.unsqueeze(0) if keepdim else data
        if self._reduce == ReduceType.MEAN:
            data = losses.mean()
            return data.unsqueeze(0) if keepdim else data
        if self._reduce == ReduceType.NONE:
            data = losses
        else:
            raise UnknownReductionError(self._reduce)

        return data

    def compute(
        self, positive_scores: FloatTensor2D, negative_scores: FloatTensor2D
    ) -> FloatTensor1D:
        """
        Computes the ranking loss between positive and negative scores.

        Args:
            positive_scores (FloatTensor2D): A tensor of shape [batch_size, 1] containing
            scores for positive samples.
            negative_scores (FloatTensor2D): A tensor of shape [batch_size, num_negatives] that
                contains scores for one/many negative samples per corresponding positive sample.

        Returns:
            FloatTensor1D: A 1D tensor of losses, reduced using the class's `reduce` method.

        Raises:
            ShapeMismatchError: If `positive_scores` and `negative_scores` have mismatched
                batch sizes or if `positive_scores` does not have shape [*, 1].
        """
        if positive_scores.shape[0] != negative_scores.shape[0]:
            raise ShapeMismatchError()
        if positive_scores.shape[1] != 1:
            raise ShapeMismatchError()

        losses = self._compute(positive_scores=positive_scores, negative_scores=negative_scores)

        return self.reduce(losses)
