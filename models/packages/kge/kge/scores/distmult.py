from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from kge.common.types import FloatTensor2D


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
        r_emb: FloatTensor2D,
        o_emb: FloatTensor2D,
        entity_embeddings: FloatTensor2D,
    ) -> FloatTensor2D:
        ro_product = r_emb * o_emb

        return torch.mm(entity_embeddings, ro_product.t()).t()

    def relations(
        self,
        s_emb: FloatTensor2D,
        o_emb: FloatTensor2D,
        relation_embeddings: FloatTensor2D,
    ) -> FloatTensor2D:
        so_product = s_emb * o_emb

        return torch.mm(relation_embeddings, so_product.t()).t()

    def objects(
        self,
        s_emb: FloatTensor2D,
        r_emb: FloatTensor2D,
        entity_embeddings: FloatTensor2D,
    ) -> FloatTensor2D:
        sr_product = s_emb * r_emb  # (batch_size, embedding_dim)

        return torch.mm(entity_embeddings, sr_product.t()).t()

    def all(
        self, s_emb: FloatTensor2D, r_emb: FloatTensor2D, o_emb: FloatTensor2D
    ) -> FloatTensor2D:
        return torch.sum(s_emb * r_emb * o_emb, dim=1, keepdim=True)
