import torch
from torch import nn

from hakken_ml_toolkit.losses.base.clf_loss import ClfLossConfig, ClfLossI
from hakken_ml_toolkit.losses.common.domain import (
    FloatTensor1D,
    FloatTensor2D,
    LongTensor2D,
)


class BCEWithLogitsLossConfig(ClfLossConfig):
    pass


class BCEWithLogitsLoss(ClfLossI[BCEWithLogitsLossConfig]):
    """
    Implements Binary Cross Entropy with Logits loss.

    It can be used for both binary classification
    and multi-label classification tasks.
    """

    def __init__(self, config: BCEWithLogitsLossConfig):
        super().__init__(config)

    def _compute(
        self,
        logits: FloatTensor2D,
        target: LongTensor2D,
        samples_weight: torch.Tensor | None = None,
    ) -> FloatTensor1D:
        """
        Computes BCE with logits loss between predicted logits and target labels.

        Args:
            logits: Tensor of shape [batch_size, num_classes] containing predicted logits
            target: Tensor of shape [batch_size, num_classes] containing target labels (0 or 1)
            samples_weight: Optional tensor of weights to apply to each sample

        Returns:
            Tensor of shape [batch_size] containing per-sample BCE losses summed across classes
        """
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", weight=samples_weight)

        return loss_fn(logits, target.float()).sum(-1)  # type: ignore[no-any-return]
