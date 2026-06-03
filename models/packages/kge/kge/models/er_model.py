from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import torch
from torch import nn

from kge.common.entities import KGEForwardOutput
from kge.common.types import FloatTensor2D, LongTensor1D, LongTensor2D
from kge.models.base import KGEI, KGEConfig

if TYPE_CHECKING:
    from collections.abc import Iterator


class ERModelConfig(KGEConfig):
    pass


ERModelType = TypeVar("ERModelType", bound="ERModel")
ERModelConfigType = TypeVar("ERModelConfigType", bound="ERModelConfig")


class ERModel(KGEI[ERModelConfigType], nn.Module):
    """
    Standard entity-relation models, with one independent embedding
    representation per entity and one embedding representation per relation.
    """

    def __init__(self, config: ERModelConfigType):
        super().__init__(config)
        nn.Module.__init__(self)
        self._entity_embeddings = self._get_entity_embeddings(
            config.num_entities, config.embedding_dim
        )
        self._relation_embeddings = self._get_relation_embeddings(
            config.num_relations, config.embedding_dim
        )

    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def eval(self: ERModelType) -> ERModelType:
        return nn.Module.eval(self)

    def train(self: ERModelType, mode: bool = True) -> ERModelType:
        return nn.Module.train(self, mode)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return nn.Module.parameters(self, recurse)

    def to_device(self: ERModelType, device: str | torch.device) -> ERModelType:
        return nn.Module.to(self, device)

    def _get_entity_embeddings(self, num_entities: int, embedding_dim: int) -> nn.Embedding:
        return nn.Embedding(num_entities, embedding_dim)

    def _get_relation_embeddings(self, num_entities: int, embedding_dim: int) -> nn.Embedding:
        return nn.Embedding(num_entities, embedding_dim)

    def forward(self, sro_batch: LongTensor2D) -> KGEForwardOutput:
        s_emb = self.entity_embeddings(sro_batch[:, 0])
        r_emb = self.relation_embeddings(sro_batch[:, 1])
        o_emb = self.entity_embeddings(sro_batch[:, 2])

        scores = self._score(s_emb, r_emb, o_emb)
        return KGEForwardOutput(scores=scores)

    def entity_embeddings(self, entity_batch: LongTensor1D) -> FloatTensor2D:
        return self._entity_embeddings.forward(entity_batch)

    def relation_embeddings(self, relation_batch: LongTensor1D) -> FloatTensor2D:
        return self._relation_embeddings.forward(relation_batch)
