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
        LongTensor2D,
    )


class ClfLossConfig(BaseModel):
    reduce: ReduceType = ReduceType.SUM


T = TypeVar("T", bound=ClfLossConfig)


class ClfLossI(ABC, Generic[T]):
    """
    Abstract interface for classification loss functions.

    Classification losses measure the error between predicted class unnormalized
    probabilities (logits) and target class labels.
    """

    def __init__(self, config: T):
        self.config = config

    @abstractmethod
    def _compute(self, logits: FloatTensor2D, target: LongTensor2D) -> FloatTensor1D:
        pass

    def reduce(
        self, losses: FloatTensor1D, keepdim: bool = True
    ) -> FloatTensor1D | FloatTensorScalar:
        if self.config.reduce == ReduceType.SUM:
            data = losses.sum()
            return data.unsqueeze(0) if keepdim else data
        if self.config.reduce == ReduceType.MEAN:
            data = losses.mean()
            return data.unsqueeze(0) if keepdim else data
        if self.config.reduce == ReduceType.NONE:
            data = losses
        else:
            raise UnknownReductionError(self.config.reduce)

        return data

    def compute(self, logits: FloatTensor2D, target: LongTensor2D, **kwargs) -> FloatTensor1D:
        if logits.shape[0] != target.shape[0]:
            raise ShapeMismatchError()

        losses = self._compute(logits=logits, target=target, **kwargs)

        return self.reduce(losses)
