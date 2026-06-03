"""Asymmetric Loss for multi-label classification (PU-friendly)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module

from hakken_models.losses.utils import reduce


class AsymmetricLoss(Module):
    """Asymmetric Loss for multi-label classification.

    Addresses positive-negative imbalance and PU (Positive-Unlabeled) settings
    by down-weighting easy negatives. Designed for scenarios with many easy
    negatives (unobserved relations) and few positives.

    Reference: Ben-Baruch et al. (2020)
    "Asymmetric Loss For Multi-Label Classification"
    https://arxiv.org/abs/2009.14119

    Formula:
        - Positive: L+ = (1 - p)^γ+ * log(p)
        - Negative: L- = (p + clip)^γ- * log(1 - p - clip)  [clipped]
        - Asymmetric focusing: weight = (1 - pt)^γ per label type

    Args:
        gamma_neg: Exponent for negative samples. Higher values (2-5) down-weight
            easy negatives more. Default 4.
        gamma_pos: Exponent for positive samples. Default 1.
        clip: Probability margin for negatives (asymmetric probability shifting).
            Shifts p for negatives: p_neg = (1 - p + clip).clamp(max=1).
            Higher clip reduces easy-negative contribution. Default 0.05.
        eps: Small constant for numerical stability. Default 1e-8.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        dim: int | None = None,
    ) -> Tensor:
        """Compute asymmetric loss.

        Args:
            logits: Raw predictions [N, C].
            labels: Multi-hot targets [N, C], values in {0, 1}.

        Returns:
            Scalar loss (after reduction).
        """
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = labels * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - labels) * torch.log(xs_neg.clamp(min=self.eps))
        loss = -(los_pos + los_neg)

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * labels
            pt1 = xs_neg * (1 - labels)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * labels + self.gamma_neg * (1 - labels)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return reduce(loss.sum(dim=-1), self.reduction, dim=dim)
