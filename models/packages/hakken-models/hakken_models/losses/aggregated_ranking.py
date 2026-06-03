"""KGE-style ranking loss: pairwise loss over positives vs multiple negatives per row."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn

from hakken_models.losses._registry import loss_fn_registry


class AggregatedRankingLoss(nn.Module):
    """Pairwise ranking loss with built-in negative aggregation (KGE / triple scoring).

    Expects ``pos_scores`` of shape ``[B]`` and ``neg_scores`` of shape ``[B, K]``.
    Aggregates the ``K`` negatives per row (``hardest`` = max score, ``mean`` = average
    over pairwise terms), then applies a registry pairwise loss (typically
    :class:`torch.nn.MarginRankingLoss`).

    This is the intended ``loss_fn`` for plain :class:`~hakken_models.models.kge.LitKGE`
    training when using multiple sampled negatives per positive triple.
    """

    def __init__(
        self,
        pair_loss: str = "MarginRankingLoss",
        pair_loss_kwargs: dict[str, Any] | None = None,
        neg_strategy: str = "hardest",
    ) -> None:
        super().__init__()
        self.pairwise = loss_fn_registry.create(pair_loss, **(pair_loss_kwargs or {}))
        self.neg_strategy = neg_strategy

    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        num_negatives = neg_scores.shape[1]
        target = torch.ones_like(pos_scores)
        if num_negatives == 1:
            return cast(
                Tensor,
                self.pairwise(pos_scores, neg_scores.squeeze(-1), target=target),
            )
        if self.neg_strategy == "hardest":
            neg_agg = neg_scores.max(dim=1)[0]
            return cast(Tensor, self.pairwise(pos_scores, neg_agg, target=target))
        if self.neg_strategy == "mean":
            pos_expanded = pos_scores.unsqueeze(1)
            loss = self.pairwise(pos_expanded, neg_scores, target=target.unsqueeze(1))
            return loss.mean()
        raise NotImplementedError(f"Unknown neg_strategy: {self.neg_strategy!r}")
