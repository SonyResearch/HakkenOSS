"""Ranking + optional multi-label relation loss (KGE, SeGAL, etc.)."""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor, nn

from hakken_models.losses._registry import loss_fn_registry
from hakken_models.losses.aggregated_ranking import AggregatedRankingLoss


class RankingRelationLoss(nn.Module):
    """Entity/triple ranking plus optional multi-label relation loss.

    Entity scores: ``pos_scores [B]``, ``neg_scores [B, K]``. When
    ``rel_logits`` / ``rel_labels`` are set and ``rel_loss_weight != 0``,
    adds a weighted relation term (e.g. BCE over relations for (s, o)).

    For **margin-only KGE**, use ``rel_loss_weight=0`` (relation head is skipped).

    Args:
        entity_loss: Registry name for pairwise ranking (e.g. ``MarginRankingLoss``).
        entity_loss_kwargs: Kwargs for entity loss.
        relation_loss: Registry name when relation term is used.
        relation_loss_kwargs: Kwargs for relation loss.
        neg_strategy: Aggregation for multiple negatives (``hardest`` / ``mean``).
        rel_loss_weight: Weight on relation loss; use ``0`` to disable.
        entity_loss_multi_neg: If True, pass full ``[B, K]`` to entity loss (e.g. NSSA).
    """

    def __init__(
        self,
        entity_loss: str = "MarginRankingLoss",
        entity_loss_kwargs: dict[str, Any] | None = None,
        relation_loss: str = "BCEWithLogitsLoss",
        relation_loss_kwargs: dict[str, Any] | None = None,
        neg_strategy: str = "hardest",
        rel_loss_weight: float = 1.0,
        entity_loss_multi_neg: bool = False,
    ) -> None:
        super().__init__()
        if entity_loss_multi_neg:
            self._entity_ranking: nn.Module = loss_fn_registry.create(
                entity_loss, **(entity_loss_kwargs or {})
            )
        else:
            self._entity_ranking = AggregatedRankingLoss(
                pair_loss=entity_loss,
                pair_loss_kwargs=entity_loss_kwargs or {},
                neg_strategy=neg_strategy,
            )
        self.relation_loss_fn = loss_fn_registry.create(
            relation_loss, **(relation_loss_kwargs or {})
        )
        self.neg_strategy = neg_strategy
        self.rel_loss_weight = rel_loss_weight
        self.entity_loss_multi_neg = entity_loss_multi_neg

    def forward(
        self,
        pos_scores: Tensor,
        neg_scores: Tensor,
        rel_logits: Tensor | None = None,
        rel_labels: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute loss and return total plus component dict for logging.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains detached
            values for logging (gradients flow only through total_loss):
            - ``entity``: Entity-ranking term.
            - ``relation``: Relation multi-label term (only when rel_logits/
              rel_labels are provided and rel_loss_weight != 0).
        """
        entity = self.entity_loss(pos_scores, neg_scores)
        loss_dict: dict[str, Tensor] = {"entity": entity.detach()}
        if self.rel_loss_weight != 0 and rel_logits is not None and rel_labels is not None:
            rel = self.relation_loss_fn(rel_logits, rel_labels)
            loss_dict["relation"] = rel.detach()
            return entity + self.rel_loss_weight * rel, loss_dict
        return entity, loss_dict

    def entity_loss(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        """Entity-ranking term (aggregated margin, NSSA, etc.)."""
        return cast(Tensor, self._entity_ranking(pos_scores, neg_scores))
