from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from hakken_ml_toolkit.ml_utils.extras import FactBatchUtils
from torch import nn

from kge.common.entities import KGEForwardOutput
from kge.common.exceptions import WrongDimensionsError
from kge.common.types import FloatTensor2D, LongTensor1D, LongTensor2D
from kge.models.base import KGEI, KGEConfig
from kge.models.conv_e.module import ConvEModule

if TYPE_CHECKING:
    from collections.abc import Iterator


class ConvEConfig(KGEConfig):
    conv_out_channels: int = 32
    conv_kernel_size: int = 3
    embedding_dropout: float = 0.2
    feature_map_dropout: float = 0.2
    projection_dropout: float = 0.3


class ConvE(KGEI[ConvEConfig], nn.Module):
    def __init__(self, config: ConvEConfig):
        nn.Module.__init__(self)
        self.config = config
        embedding_height, embedding_width = self.compute_embedding_dimensions(
            self.config.embedding_dim
        )

        self.model = ConvEModule(
            self.config.num_entities,
            self.config.num_relations,
            embedding_height=embedding_height,
            embedding_width=embedding_width,
            conv_out_channels=self.config.conv_out_channels,
            conv_kernel_size=self.config.conv_kernel_size,
            embedding_dropout=self.config.embedding_dropout,
            feature_map_dropout=self.config.feature_map_dropout,
            projection_dropout=self.config.projection_dropout,
        )

    @classmethod
    def get_config_class(cls) -> type[ConvEConfig]:
        return ConvEConfig

    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    @staticmethod
    def compute_embedding_dimensions(embedding_dim: int) -> tuple[int, int]:
        """
        Compute the height and width of the embedding based on the total embedding dimension.
        The dimensions are chosen to be as close to a square as possible, with height >= width.
        """
        if embedding_dim < 10:
            msg = "Embedding dimension must be greater than 10"
            raise WrongDimensionsError(msg)
        embedding_height = 10  # int(math.ceil(math.sqrt(embedding_dim)))
        embedding_width = math.ceil(embedding_dim / embedding_height)

        # Ensure that height * width == embedding_dim
        while embedding_height * embedding_width > embedding_dim:
            if embedding_height > embedding_width:
                embedding_height -= 1
            else:
                embedding_width -= 1

        if embedding_height * embedding_width != embedding_dim:
            expected_value = embedding_height * embedding_width
            msg = f"""
            Cannot find valid dimensions for embedding_dim {embedding_dim}. 
            Closest match: {embedding_height} * {embedding_width} = {expected_value}"""
            raise WrongDimensionsError(msg)

        return embedding_height, embedding_width

    def eval(self) -> ConvE:
        return nn.Module.eval(self)

    def train(self, mode: bool = True) -> ConvE:
        return nn.Module.train(self, mode)

    def to_device(self, device: str | torch.device) -> ConvE:
        return nn.Module.to(self, device)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return nn.Module.parameters(self, recurse)

    def forward(self, sro_batch: LongTensor2D) -> KGEForwardOutput:
        scores = self.model.forward(
            subject_indices=sro_batch[:, 0],
            relation_indices=sro_batch[:, 1],
            object_indices=sro_batch[:, 2],
        )
        return KGEForwardOutput(scores=scores)

    def score_relations(self, so_batch: LongTensor2D) -> FloatTensor2D:
        batch_size = so_batch.size(0)

        sro_batch = FactBatchUtils.so_to_sro_batch(
            so_batch, num_relations=self.config.num_relations
        )

        scores = self.score_by_batches(sro_batch=sro_batch, batch_size=batch_size * 100)

        return scores.view(batch_size, -1)

    def _score_relations(self, _s_emb: FloatTensor2D, _o_emb: FloatTensor2D) -> FloatTensor2D:
        raise NotImplementedError()

    @torch.no_grad()
    def score_by_batches(self, sro_batch: LongTensor2D, batch_size: int) -> FloatTensor2D:
        total_size = sro_batch.size(0)
        num_batches = (total_size + batch_size - 1) // batch_size

        scores_list = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_size)
            sro_batch_i = sro_batch[start_idx:end_idx]

            scores_i = self.score(sro_batch_i)
            scores_list.append(scores_i)

        return torch.cat(scores_list, dim=0)

    def score_subjects(self, ro_batch: FloatTensor2D) -> FloatTensor2D:
        batch_size = ro_batch.shape[0]

        sro_batch = FactBatchUtils.ro_to_sro_batch(ro_batch, num_entities=self.config.num_entities)

        scores = self.score_by_batches(sro_batch=sro_batch, batch_size=batch_size * 100)

        return scores.view(batch_size, -1)

    def _score_subjects(self, _r_emb: FloatTensor2D, _o_emb: FloatTensor2D) -> FloatTensor2D:
        raise NotImplementedError()

    def score_objects(self, sr_batch: LongTensor2D) -> FloatTensor2D:
        return self.model.forward(
            subject_indices=sr_batch[:, 0],
            relation_indices=sr_batch[:, 1],
        )

    def _score_objects(self, _s_emb: FloatTensor2D, _r_emb: FloatTensor2D) -> FloatTensor2D:
        raise NotImplementedError()

    def _score(
        self, _s_emb: FloatTensor2D, _r_emb: FloatTensor2D, _o_emb: FloatTensor2D
    ) -> FloatTensor2D:
        raise NotImplementedError()

    def score(self, sro_batch: LongTensor2D) -> FloatTensor2D:
        return self.model.forward(
            subject_indices=sro_batch[:, 0],
            relation_indices=sro_batch[:, 1],
            object_indices=sro_batch[:, 2],
        )

    def entity_embeddings(self, entity_batch: LongTensor1D) -> FloatTensor2D:
        return self.model.get_entity_embeddings(entity_batch)

    def relation_embeddings(self, relation_batch: LongTensor1D) -> FloatTensor2D:
        return self.model.get_relation_embeddings(relation_indices=relation_batch)
