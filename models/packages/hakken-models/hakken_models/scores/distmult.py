from __future__ import annotations

import torch

from .base import ScoreFn


class DistMultScore(ScoreFn):
    """DistMult computes the plausibility of a triple using element-wise
    multiplication and summation:

    score(s, r, o) = sum(s ⊙ r ⊙ o)

    where ⊙ denotes element-wise multiplication. This scoring function assumes
    that entity and relation embeddings have the same dimensionality.
    """

    def subjects(
        self,
        r_emb: torch.Tensor,
        o_emb: torch.Tensor,
        entity_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        ro_product = r_emb * o_emb

        return torch.mm(entity_embeddings, ro_product.t()).t()

    def relations(
        self,
        s_emb: torch.Tensor,
        o_emb: torch.Tensor,
        relation_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        so_product = s_emb * o_emb

        return torch.mm(relation_embeddings, so_product.t()).t()

    def objects(
        self,
        s_emb: torch.Tensor,
        r_emb: torch.Tensor,
        entity_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sr_product = s_emb * r_emb  # (batch_size, embedding_dim)

        return torch.mm(entity_embeddings, sr_product.t()).t()

    def all(self, s_emb: torch.Tensor, r_emb: torch.Tensor, o_emb: torch.Tensor) -> torch.Tensor:
        return torch.sum(s_emb * r_emb * o_emb, dim=1, keepdim=True)
