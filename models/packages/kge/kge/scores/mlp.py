from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from gnn import MLP, MLPConfig
from hakken_ml_toolkit.ml_utils.extras import PyTorchUtils
from loguru import logger

from kge.common.types import FloatTensor2D

from .base import ScoreFn

if TYPE_CHECKING:
    from collections.abc import Callable


class MLPScore(ScoreFn):
    def __init__(self, config: MLPConfig):
        super().__init__()

        logger.info("Initializing MLPScore")

        if config.output_dim != 1:
            msg = "output_dim must be 1"
            raise ValueError(msg)
        if config.drop_last:
            msg = "drop_last must be False"
            raise ValueError(msg)
        if config.use_activation_output:
            msg = "use_activation_output must be False"
            raise ValueError(msg)
        self.mlp = MLP(config)

        self._entity_batch_size: int | None = None

    def set_entity_batch_size(self, batch_size: int) -> None:
        """Set the batch size for entity operations to control memory usage."""
        self._entity_batch_size = batch_size

    def _get_entity_batch_size(self, num_entities: int) -> int:
        """Get the effective batch size for entity operations."""
        if self._entity_batch_size is None:
            return num_entities  # No batching
        return min(self._entity_batch_size, num_entities)

    def _batched_entity_scoring(
        self,
        batch_size: int,
        num_entities: int,
        combine_fn: Callable[[int, int], FloatTensor2D],
    ) -> FloatTensor2D:
        """
        Generic batched entity scoring function.

        Args:
            batch_size: Number of query batches
            num_entities: Total number of entities to score against
            combine_fn: Function that takes (start_idx, end_idx) and returns
                       combined embeddings for entities in that range

        Returns:
            scores: [batch_size, num_entities]
        """
        entity_batch_size = self._get_entity_batch_size(num_entities)

        if entity_batch_size >= num_entities:
            # No batching needed
            combined = combine_fn(0, num_entities)
            combined = combined.reshape(-1, combined.shape[-1])
            scores = self.mlp.forward(combined)
            return scores.reshape(batch_size, num_entities)

        # Batched processing
        all_scores = []

        for start_idx in range(0, num_entities, entity_batch_size):
            end_idx = min(start_idx + entity_batch_size, num_entities)
            current_batch_size = end_idx - start_idx

            # Get combined embeddings for current entity batch
            combined = combine_fn(start_idx, end_idx)
            combined = combined.reshape(-1, combined.shape[-1])

            # Score current batch
            batch_scores = self.mlp.forward(combined)
            batch_scores = batch_scores.reshape(batch_size, current_batch_size)

            all_scores.append(batch_scores)

        return torch.cat(all_scores, dim=1)

    def subjects(
        self,
        r_emb: FloatTensor2D,
        o_emb: FloatTensor2D,
        entity_embeddings: FloatTensor2D,
    ) -> FloatTensor2D:
        batch_size = r_emb.shape[0]
        num_entities = entity_embeddings.shape[0]

        def combine_subjects(start_idx: int, end_idx: int) -> FloatTensor2D:
            # Get entity batch
            entity_batch = entity_embeddings[start_idx:end_idx]
            current_batch_size = entity_batch.shape[0]

            # Expand relation and object embeddings
            r_expanded = r_emb.unsqueeze(1).expand(-1, current_batch_size, -1)
            o_expanded = o_emb.unsqueeze(1).expand(-1, current_batch_size, -1)

            # Combine embeddings: [batch_size, current_batch_size, 3*embed_dim]
            return torch.cat(
                [
                    entity_batch.unsqueeze(0).expand(batch_size, -1, -1),
                    r_expanded,
                    o_expanded,
                ],
                dim=-1,
            )

        return self._batched_entity_scoring(batch_size, num_entities, combine_subjects)

    def objects(
        self,
        s_emb: FloatTensor2D,
        r_emb: FloatTensor2D,
        entity_embeddings: FloatTensor2D,
    ) -> FloatTensor2D:
        batch_size = s_emb.shape[0]
        num_entities = entity_embeddings.shape[0]

        def combine_objects(start_idx: int, end_idx: int) -> FloatTensor2D:
            # Get entity batch
            entity_batch = entity_embeddings[start_idx:end_idx]
            current_batch_size = entity_batch.shape[0]

            # Expand subject and relation embeddings
            s_expanded = s_emb.unsqueeze(1).expand(-1, current_batch_size, -1)
            r_expanded = r_emb.unsqueeze(1).expand(-1, current_batch_size, -1)

            # Combine embeddings: [batch_size, current_batch_size, 3*embed_dim]
            return torch.cat(
                [
                    s_expanded,
                    r_expanded,
                    entity_batch.unsqueeze(0).expand(batch_size, -1, -1),
                ],
                dim=-1,
            )

        return self._batched_entity_scoring(batch_size, num_entities, combine_objects)

    def relations(
        self,
        s_emb: FloatTensor2D,
        o_emb: FloatTensor2D,
        relation_embeddings: FloatTensor2D,
    ) -> FloatTensor2D:
        """Relations method - typically doesn't need batching as num_relations is usually small."""
        batch_size = s_emb.shape[0]
        num_relations = relation_embeddings.shape[0]

        # Expand subject and object embeddings
        s_expanded = s_emb.unsqueeze(1).expand(-1, num_relations, -1)
        o_expanded = o_emb.unsqueeze(1).expand(-1, num_relations, -1)

        # Combine all embeddings
        combined = torch.cat(
            [
                s_expanded,
                relation_embeddings.unsqueeze(0).expand(batch_size, -1, -1),
                o_expanded,
            ],
            dim=-1,
        )

        # Reshape and score
        combined = combined.reshape(-1, 3 * s_emb.shape[-1])
        scores = self.mlp.forward(combined)
        return scores.reshape(batch_size, num_relations)

    def all(
        self, s_emb: FloatTensor2D, r_emb: FloatTensor2D, o_emb: FloatTensor2D
    ) -> FloatTensor2D:
        embedding = PyTorchUtils.concat_tensors([s_emb, r_emb, o_emb])
        return self.mlp.forward(embedding)
