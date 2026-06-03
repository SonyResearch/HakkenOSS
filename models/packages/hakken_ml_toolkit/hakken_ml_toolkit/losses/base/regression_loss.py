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


class RegressionLossConfig(BaseModel):
    reduce: ReduceType = ReduceType.SUM


T = TypeVar("T", bound=RegressionLossConfig)


class RegressionLossI(ABC, Generic[T]):
    """
    Abstract interface for regression loss functions.

    Regression losses measure the error between predicted continuous values and target values.
    """

    def __init__(self, config: T):
        self.config = config
        self._reduce: ReduceType = self.config.reduce

    @abstractmethod
    def _compute(self, prediction: FloatTensor2D, target: FloatTensor2D) -> FloatTensor1D:
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
            raise UnknownReductionError(self.config.reduce)

        return data

    def compute(self, prediction: FloatTensor2D, target: FloatTensor2D, **kwargs) -> FloatTensor1D:
        if prediction.shape[0] != target.shape[0]:
            raise ShapeMismatchError()

        losses = self._compute(prediction=prediction, target=target, **kwargs)

        return self.reduce(losses)
