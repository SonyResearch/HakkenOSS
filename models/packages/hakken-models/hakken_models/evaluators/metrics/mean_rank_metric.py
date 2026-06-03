"""Mean rank metric for entity-ranking evaluation (pos_scores vs neg_scores)."""

from typing import Self

import torch
from torch import Tensor


class MeanRankMetric:
    """Accumulates approximate mean rank over batches (sampled negatives).

    For each row, ``rank = 1 + #(negatives with score > positive)``. Across
    batches, the metric reports the mean rank over all examples (weighted by
    count), not the mean of per-batch means.
    """

    def __init__(self) -> None:
        self._sum_rank: float = 0.0
        self._count: int = 0
        self._device: torch.device | str = torch.device("cpu")

    def update(self, pos_scores: Tensor, neg_scores: Tensor) -> None:
        """Update with a batch of positive and negative scores.

        Args:
            pos_scores: [B] scores for positive triples.
            neg_scores: [B, K] scores for corrupted negatives.
        """
        if not torch.isfinite(pos_scores).all() or not torch.isfinite(neg_scores).all():
            return
        num_better = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1).float()
        ranks = 1.0 + num_better
        self._sum_rank += ranks.sum().item()
        self._count += ranks.numel()

    def compute(self) -> Tensor:
        """Return mean rank over all accumulated batches."""
        device = (
            self._device if isinstance(self._device, torch.device) else torch.device(self._device)
        )
        if self._count == 0:
            return torch.tensor(float("inf"), device=device)
        return torch.tensor(self._sum_rank / self._count, device=device)

    def reset(self) -> None:
        """Reset accumulated state."""
        self._sum_rank = 0.0
        self._count = 0

    def to(self, device: torch.device | str) -> Self:
        """Move metric to device."""
        self._device = device
        return self
