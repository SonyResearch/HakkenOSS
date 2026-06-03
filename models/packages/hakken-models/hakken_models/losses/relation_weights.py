"""Utilities for relation-label weighting in imbalanced classification."""

from __future__ import annotations

from torch import Tensor


def compute_pos_weight_from_relation_labels(
    labels: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Compute pos_weight for BCE from multi-hot relation labels.

    For each relation r: pos_weight[r] = n_neg[r] / (n_pos[r] + eps).
    Boosts rare relations when used with BCEWithLogitsLoss or FLWithLogitsLoss.

    Args:
        labels: ``[N, R]`` float tensor, multi-hot (0 or 1).
        eps: Small constant to avoid division by zero for rare relations.

    Returns:
        ``[R]`` float tensor suitable for pos_weight.
    """
    n_pos = labels.sum(dim=0)
    n_neg = labels.shape[0] - n_pos
    pos_weight = n_neg / (n_pos + eps)
    return pos_weight.float()
