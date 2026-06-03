import torch

from hakken_ml_toolkit.losses.base.regression_loss import (
    RegressionLossConfig,
    RegressionLossI,
)
from hakken_ml_toolkit.losses.common.domain import FloatTensor1D, FloatTensor2D


class MSELossConfig(RegressionLossConfig):
    pass


class MSELoss(RegressionLossI[MSELossConfig]):
    def __init__(self, config: MSELossConfig):
        super().__init__(config)

    def _compute(self, prediction: FloatTensor2D, target: FloatTensor2D) -> FloatTensor1D:
        squared_diff = (prediction - target) ** 2

        return torch.mean(squared_diff, dim=-1)
