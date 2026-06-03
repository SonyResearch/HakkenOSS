from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from kge.models.er_model import ERModel, ERModelConfig

if TYPE_CHECKING:
    from kge.common.types import FloatTensor2D


class DistMultConfig(ERModelConfig):
    name: str = "distmult"


class DistMult(ERModel[DistMultConfig]):
    def __init__(self, config: DistMultConfig):
        super().__init__(config)

    @classmethod
    def get_config_class(cls) -> type[DistMultConfig]:
        return DistMultConfig

    def _score_subjects(self, r_emb: FloatTensor2D, o_emb: FloatTensor2D) -> FloatTensor2D:
        r_o = r_emb * o_emb

        return torch.matmul(r_o, self._entity_embeddings.weight.t())

    def _score_relations(self, s_emb: FloatTensor2D, o_emb: FloatTensor2D) -> FloatTensor2D:
        s_o = s_emb * o_emb

        return torch.matmul(s_o, self._relation_embeddings.weight.t())

    def _score_objects(self, s_emb: FloatTensor2D, r_emb: FloatTensor2D) -> FloatTensor2D:
        s_r = s_emb * r_emb

        return torch.matmul(s_r, self._entity_embeddings.weight.t())

    def _score(
        self, s_emb: FloatTensor2D, r_emb: FloatTensor2D, o_emb: FloatTensor2D
    ) -> FloatTensor2D:
        return torch.sum(s_emb * r_emb * o_emb, dim=-1, keepdim=True)
