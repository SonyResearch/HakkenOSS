from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class ScoreFn(nn.Module, ABC):
    """Abstract base class for scoring functions in knowledge graph embedding models.

    This class defines the interface for scoring functions that compute compatibility
    scores for knowledge graph triples (subject, relation, object). Subclasses must
    implement methods to score different prediction scenarios.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def subjects(
        self,
        r_emb: torch.Tensor,
        o_emb: torch.Tensor,
        entity_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scores for all possible subjects given relation and object.

        This method scores all entities as potential subjects for triples of the
        form (?, relation, object), where ? represents all possible entities.

        Args:
            r_emb: Relation embeddings with shape (batch_size, relation_dim).
            o_emb: Object embeddings with shape (batch_size, entity_dim).
            entity_embeddings: All entity embeddings with shape (num_entities, entity_dim).

        Returns:
            Scores for all possible subjects with shape (batch_size, num_entities).
            Higher scores indicate more plausible triples.
        """
        pass

    @abstractmethod
    def relations(
        self,
        s_emb: torch.Tensor,
        o_emb: torch.Tensor,
        relation_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scores for all possible relations given subject and object.

        This method scores all relations as potential predicates for triples of the
        form (subject, ?, object), where ? represents all possible relations.

        Args:
            s_emb: Subject embeddings with shape (batch_size, entity_dim).
            o_emb: Object embeddings with shape (batch_size, entity_dim).
            relation_embeddings: All relation embeddings with shape (num_relations, relation_dim).

        Returns:
            Scores for all possible relations with shape (batch_size, num_relations).
            Higher scores indicate more plausible triples.
        """
        pass

    @abstractmethod
    def objects(
        self,
        s_emb: torch.Tensor,
        r_emb: torch.Tensor,
        entity_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scores for all possible objects given subject and relation.

        This method scores all entities as potential objects for triples of the
        form (subject, relation, ?), where ? represents all possible entities.

        Args:
            s_emb: Subject embeddings with shape (batch_size, entity_dim).
            r_emb: Relation embeddings with shape (batch_size, relation_dim).
            entity_embeddings: All entity embeddings with shape (num_entities, entity_dim).

        Returns:
            Scores for all possible objects with shape (batch_size, num_entities).
            Higher scores indicate more plausible triples.
        """
        pass

    @abstractmethod
    def all(self, s_emb: torch.Tensor, r_emb: torch.Tensor, o_emb: torch.Tensor) -> torch.Tensor:
        """Compute scores for specific subject-relation-object triples.

        This method computes compatibility scores for complete triples where all
        three components (subject, relation, object) are specified.

        Args:
            s_emb: Subject embeddings with shape (batch_size, entity_dim).
            r_emb: Relation embeddings with shape (batch_size, relation_dim).
            o_emb: Object embeddings with shape (batch_size, entity_dim).

        Returns:
            Scores for the given triples with shape (batch_size, 1)
            Higher scores indicate more plausible triples.
        """
        pass
