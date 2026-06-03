# ruff: noqa: ARG002

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from kge.common.entities import KGEForwardOutput
from kge.common.types import (
    FloatTensor2D,
    LongTensor1D,
    LongTensor2D,
)
from kge.models.base import KGEI
from kge.models.config import KGEConfig

if TYPE_CHECKING:
    from collections.abc import Iterator


class RandomKGE(KGEI[KGEConfig]):
    """Random KGE implementation for testing and baseline purposes."""

    def __init__(self, config: KGEConfig):
        super().__init__(config)
        nn.Module.__init__(self)
        self.dummy_param = nn.Parameter(torch.tensor(0.0))

    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def eval(self: RandomKGE) -> RandomKGE:
        return nn.Module.eval(self)

    def train(self: RandomKGE, mode: bool = True) -> RandomKGE:
        return nn.Module.train(self, mode)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return nn.Module.parameters(self, recurse)

    def to_device(self: RandomKGE, device: str | torch.device) -> RandomKGE:
        return nn.Module.to(self, device)

    def forward(self, sro_batch: LongTensor2D) -> KGEForwardOutput:
        """Forward pass returning random scores."""
        batch_size = sro_batch.size(0)
        scores = torch.rand(batch_size, 1, device=self.device)
        return KGEForwardOutput(scores=scores)

    def _score_objects(self, s_emb: FloatTensor2D, r_emb: FloatTensor2D) -> FloatTensor2D:
        """Score all objects for given subject-relation pairs."""
        batch_size = s_emb.size(0)
        return torch.rand(batch_size, self.config.num_entities, device=self.device)

    def _score_subjects(self, r_emb: FloatTensor2D, o_emb: FloatTensor2D) -> FloatTensor2D:
        """Score all subjects for given relation-object pairs."""
        batch_size = o_emb.size(0)
        return torch.rand(batch_size, self.config.num_entities, device=self.device)

    def _score_relations(self, s_emb: FloatTensor2D, o_emb: FloatTensor2D) -> FloatTensor2D:
        """Score all relations for given subject-object pairs."""
        batch_size = s_emb.size(0)
        return torch.rand(batch_size, self.config.num_relations, device=self.device)

    def _score(
        self, s_emb: FloatTensor2D, r_emb: FloatTensor2D, o_emb: FloatTensor2D
    ) -> FloatTensor2D:
        """Score specific subject-relation-object triples."""
        batch_size = s_emb.size(0)
        return torch.rand(batch_size, 1, device=self.device)

    def entity_embeddings(self, entity_batch: LongTensor1D) -> FloatTensor2D:
        """Return random entity embeddings."""
        batch_size = entity_batch.size(0)
        return torch.rand(batch_size, self.config.embedding_dim, device=self.device)

    def relation_embeddings(self, relation_batch: LongTensor1D) -> FloatTensor2D:
        """Return random relation embeddings."""
        batch_size = relation_batch.size(0)
        return torch.rand(batch_size, self.config.embedding_dim, device=self.device)

    @classmethod
    def get_config_class(cls) -> type[KGEConfig]:
        return KGEConfig

    @torch.no_grad()
    def normalize_scores(self, scores: FloatTensor2D) -> FloatTensor2D:
        """Normalize scores using the provided scaler."""

        return torch.sigmoid(scores)
